import argparse
import json
import os
import sys
from enum import IntEnum
import yaml
import time
from collections import Counter

sys.path.append('./src/diffusion-policy')
import copy
import itertools
import random
import re
from collections import OrderedDict
import requests

import cv2
import habitat
import imageio
import numpy as np
import quaternion
import torch
import tqdm
from depth_camera_filtering import filter_depth
from habitat.config.default import get_agent_config
from habitat.config.default_structured_configs import (
    CollisionsMeasurementConfig,
    FogOfWarConfig,
    TopDownMapMeasurementConfig,
)
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
from habitat.utils.visualizations.utils import images_to_video, observations_to_image
from habitat_baselines.config.default import get_config as get_habitat_config
from PIL import Image

from internnav.configs.evaluator import EvalCfg
from internnav.evaluator import DistributedEvaluator, Evaluator
from internnav.habitat_extensions.vln.utils import (
    get_axis_align_matrix,
    get_intrinsic_matrix,
    pixel_to_gps,
    preprocess_depth_image_v2,
    xyz_yaw_pitch_to_tf_matrix,
)
from internnav.model.utils.vln_utils import split_and_clean, traj_to_actions
from internnav.configs.agent import InitRequest, ResetRequest, StepRequest
from internnav.utils.comm_utils.client_utils import serialize_obs, solve_optimal_patch_ratio, \
    numpy_compression_by_patch
from internnav.utils.comm_utils.visual_encoder import VisionEncoder
from internnav.habitat_extensions.vln.system_log import InferenceLogger
from internnav.habitat_extensions.vln.s1_agent import System1
from internnav.habitat_extensions.vln.network_monitor import EdgeMonitor

# Import for Habitat registry side effects — do not remove
import internnav.habitat_extensions.vln.measures  # noqa: F401 # isort: skip


DEFAULT_IMAGE_TOKEN = "<image>"

MAX_STEPS = 8
MAX_LOCAL_STEPS = 4


class action_code(IntEnum):
    STOP = 0
    FORWARD = 1
    LEFT = 2
    RIGHT = 3
    LOOKUP = 4
    LOOKDOWN = 5


class system_perf(enumerate):
    TRANSMISSION = "transmission_time"
    S1 = "s1_time"
    S2 = "s2_time"
    BW = "bandwidth"
    STEP = "step_time"
    STEP_ID = "step_id"
    EPISODE_ID = "episode_id"
    COMP_RATIO = 'comp_ratio'


@Evaluator.register('lori_habitat_vln')
class LoriHabitatVLNEvaluator(DistributedEvaluator):
    def __init__(self, cfg: EvalCfg):
        args = argparse.Namespace(**cfg.eval_settings)
        self.save_video = args.save_video
        self.epoch = args.epoch
        self.max_steps_per_episode = args.max_steps_per_episode
        self.output_path = args.output_path

        self.inference_logger = InferenceLogger(os.path.join(self.output_path, 'system_perf.jsonl'))
        self.device = torch.device(f"cuda")

        # create habitat config
        self.config_path = cfg.env.env_settings['config_path']
        self.config = get_habitat_config(self.config_path)
        self.agent_config = get_agent_config(self.config.habitat.simulator)
        self.sim_sensors_config = self.config.habitat.simulator.agents.main_agent.sim_sensors

        with habitat.config.read_write(self.config):
            self.config.habitat.task.measurements.update(
                {
                    "top_down_map": TopDownMapMeasurementConfig(
                        map_padding=3,
                        map_resolution=1024,
                        draw_source=True,
                        draw_border=True,
                        draw_shortest_path=True,
                        draw_view_points=True,
                        draw_goal_positions=True,
                        draw_goal_aabbs=True,
                        fog_of_war=FogOfWarConfig(
                            draw=True,
                            visibility_dist=5.0,
                            fov=90,
                        ),
                    ),
                    "collisions": CollisionsMeasurementConfig(),
                }
            )
        cfg.env.env_settings['habitat_config'] = self.config
        cfg.env.env_settings['output_path'] = self.output_path

        # init agent and env
        super().__init__(cfg, init_agent=False)

        self.s1_agent = System1(cfg.agent)
        # self.s2_base_url = f"http://192.168.105.15:8023"
        self.s2_base_url = f"http://192.168.105.5:30814"
        self._init_agents(cfg.agent)
        self.ema_bandwidth = None
        with open('scripts/eval/configs/latency_profile.yaml', 'r', encoding='utf-8') as f:
            self.infer_profile_data: dict = yaml.safe_load(f)
        self.cloud_latency_threshold = 0.5 # seconds
        self.e2e_latency_threshold = 1.5

        # network monitor process
        self.network_monitor = EdgeMonitor(
            f"{self.s2_base_url}/agent/heartbeat",
            interval=0.5)
        self.network_monitor.start()

# ------------------------------------- model ------------------------------------------
        self.model_args = argparse.Namespace(**cfg.agent.model_settings)
        self.vis_debug = bool(getattr(self.model_args, "vis_debug", False))
        self.vis_debug_path = getattr(self.model_args, "vis_debug_path", os.path.join(self.output_path, "vis_debug"))
        self.vision_encoder = VisionEncoder(self.model_args.s1_type)
        self.set_adaptive_compression = bool(getattr(self.model_args, "adaptive_compression", False))

        self._camera_height = self.sim_sensors_config.rgb_sensor.position[1]
        self._min_depth = self.sim_sensors_config.depth_sensor.min_depth
        self._max_depth = self.sim_sensors_config.depth_sensor.max_depth

        camera_fov_rad = np.deg2rad(self.sim_sensors_config.depth_sensor.hfov)
        self._camera_fov = camera_fov_rad
        self._fx = self._fy = self.sim_sensors_config.depth_sensor.width / (2 * np.tan(camera_fov_rad / 2))

    def _init_agents(self, agent_cfg):
        self.s2_agent_name = self._init_s2_agent(agent_cfg)
    
    def _init_s2_agent(self, agent_cfg):
        agent_cfg.model_name = 'habitat_s2_agent'
        request_data = InitRequest(agent_config=agent_cfg)
        response = requests.post(
            url=f'{self.s2_base_url}/agent/init',
            json=request_data.model_dump(mode='json'),
            headers={'Content-Type': 'application/json'},
        )
        response.raise_for_status()
        return response.json()['agent_name']
    
    def s2_agent_step(self, rgb, text=None, step_id=None, look_down=None):
        obs = {'rgb': rgb, 'instruction': text, 'step_id': step_id, 'look_down': look_down}
        origin_size = None
        if self.set_adaptive_compression:
            # Estimate cloud latency
            origin_size = len(serialize_obs(obs))
            estimated_cloud_time = self.estimate_cloud_latency(origin_size)
            if_compressed = (estimated_cloud_time is not None) and \
                (estimated_cloud_time > self.cloud_latency_threshold)
            if if_compressed:
                compressed_img = self.compress_image(rgb)
                if compressed_img is not None:
                    obs['rgb'] = compressed_img
                    obs['compressed'] = 1

        upload_data = serialize_obs(obs)

        transmission_start_time = time.time()
        request_data = StepRequest(observation=upload_data).model_dump(mode='json')
        response = requests.post(
            url=f'{self.s2_base_url}/agent/{self.s2_agent_name}/step',
            json=request_data,
            headers={'Content-Type': 'application/json'},
        )
        response.raise_for_status()
        cloud_latency = time.time() - transmission_start_time

        response_data = response.json()['action']
        s2_inference_time = response_data.pop('processing_time')
        transmission_delay = cloud_latency - s2_inference_time

        self.inference_logger.record_by_key(system_perf.TRANSMISSION, transmission_delay)
        self.inference_logger.record_by_key(system_perf.S2, s2_inference_time)
        upload_data_size = len(upload_data)
        if origin_size is not None:
            self.inference_logger.record_by_key(system_perf.COMP_RATIO, \
                                                (origin_size - upload_data_size) / origin_size)

        self.update_bandwidth(upload_data_size, transmission_delay)

        return response_data
    
    def s1_agent_step(self, rgb, depth, habitat_time, traj_latents=None):
        s1_start_time = time.time()
        elasped_time = s1_start_time - self.step_start_time
        obs = {'rgb': rgb, 'depth': depth, 'traj_latents': traj_latents, 
               'latency_constraint': self.e2e_latency_threshold - elasped_time + habitat_time}
        actions = self.s1_agent.step(obs)
        self.inference_logger.record_by_key(system_perf.S1, time.time() - s1_start_time)
        return actions
    
    def semantic_hold_infer(self, look_down_image, look_down_depth, habitat_time):
        self.semantic_hold_steps += 1
        k = self.semantic_hold_steps

        # 1. 产生本地快模型（S1）离散候选动作
        try:
            s1_response = self.s1_agent_step(look_down_image, look_down_depth, habitat_time)
            s1_candidate_action = s1_response['local_actions'][0]
            print(f"[Semantic Hold] Local S1 inference during hold: candidate action {int(s1_candidate_action)}.")
        except Exception as e:
            print(f"[Semantic Hold] Local S1 inference failure during hold: {e}")
            return -1

        # 2. 计算风险项 A: 离散历史动作切换频率 (Volatility)
        history_len = 5
        recent_actions = self.action_history[-history_len:]
        
        action_volatility = 0.0
        momentum_action = s1_candidate_action
        if len(recent_actions) >= 2:
            transitions = sum(1 for i in range(1, len(recent_actions)) if recent_actions[i] != recent_actions[i-1])
            action_volatility = transitions / (len(recent_actions) - 1)  # 归一化到 [0, 1]
            
            # 提取动量行为：滑动历史窗口内的众数动作
            counts = Counter(recent_actions)
            momentum_action = counts.most_common(1)[0][0]
            print(f"[Semantic Hold] Momentum action: {int(momentum_action)}.")

        # 3. 计算风险项 B: 零算力开销视觉突变率 (Visual Shift)
        visual_shift = 0.0
        current_rgb = np.array(look_down_image)
        if self.semantic_hold_last_rgb is not None:
            diff = (current_rgb.astype(np.float32) - self.semantic_hold_last_rgb.astype(np.float32)) / 255.0
            visual_shift = float(np.mean(diff ** 2) * 100)
        self.semantic_hold_last_rgb = current_rgb

        # 4. 融合综合风险判定自适应外推长度
        w1, w2, w3 = 0.1, 1.0, 1.5  # 系统调优超参数
        R_k = (w1 * k) + (w2 * action_volatility) + (w3 * visual_shift)
        theta_safe = 1.0  # 安全动态硬截断阈值

        if R_k < theta_safe:
            # 处于安全包络线内：执行离散多数投票外推
            alpha_k = max(0.0, 1.0 - 0.2 * k)
            if alpha_k >= 0.5:
                action = s1_candidate_action
                log_msg = "S1 Candidate"
            else:
                action = momentum_action
                log_msg = "Historical Momentum"
            print(f"[Outage Extrapolation] R_k={R_k:.3f} < {theta_safe}. Mode: {log_msg}, Action: {int(action)}")
        else:
            # 超出安全包络线（断网过久或撞墙风险剧增）：自适应硬截断，强制刹车
            action = -1
            print(f"[Emergency Brake] Risk threshold exceeded! R_k={R_k:.3f} >= {theta_safe}. Action hard truncated to STOP.")
        
        return action
    
    def compress_image(self, image):
        vit_latency, patch_importance = self.vision_encoder.get_patch_importance(image)
        time_constraint = self.cloud_latency_threshold - vit_latency
        print(f"Latency constraint for s2 transmission is {time_constraint}s.")
        p_star = solve_optimal_patch_ratio(np.array(image),
                                        time_constraint,
                                        self.ema_bandwidth,
                                        compression_factor=4)
        if p_star < 1.0:
            print(f"Compression ratio is {p_star}.")
            compressed_img = numpy_compression_by_patch(np.array(image), 
                                                    patch_importance, 
                                                    keep_ratio=p_star, 
                                                    compression_factor=4)
            return compressed_img
        
        return None

    def estimate_cloud_latency(self, upload_size_bytes):
        estimate_transmission_time = self.cal_transmission_time(upload_size_bytes)
        if estimate_transmission_time is None:
            return
        
        return estimate_transmission_time

    def cal_transmission_time(self, upload_size_bytes):
        if self.ema_bandwidth is None:
            return 
        
        return upload_size_bytes / self.ema_bandwidth
    
    def update_bandwidth(self, data_size, transmission_latency, alpha=0.3):
        '''
        Update the bandwidth estimation.

        Args:
            data_size: The size of the data transmitted (in bytes).
            upload_latency: The time taken for the transmission (in seconds).
            alpha: The smoothing factor for EMA.
        '''
        bandwidth = data_size / transmission_latency # Bytes/s
        if self.ema_bandwidth is None:
            self.ema_bandwidth = bandwidth
        else:
            self.ema_bandwidth = alpha * bandwidth + (1 - alpha) * self.ema_bandwidth
        
        self.inference_logger.record_by_key(system_perf.BW, self.ema_bandwidth)

    def s1_agent_reset(self, reset_index=None):
        response = requests.post(
            url=f'{self.s1_base_url}/agent/{self.s1_agent_name}/reset',
            json=ResetRequest(reset_index=reset_index).model_dump(mode='json'),
            headers={'Content-Type': 'application/json'},
        )
        response.raise_for_status()

    def s2_agent_reset(self, reset_index=None):
        response = requests.post(
            url=f'{self.s2_base_url}/agent/{self.s2_agent_name}/reset',
            json=ResetRequest(reset_index=reset_index).model_dump(mode='json'),
            headers={'Content-Type': 'application/json'},
        )
        response.raise_for_status()

    def eval_action(self):
        """
        Run local episodes on this rank.

        Returns dict[str, Tensor] on GPU (1D tensors of same length).
        """
        # Old behavior was something like:
        # sucs, spls, oss, nes, ep_num = self.eval_action(self.rank)
        # Now just implement the actual eval here and return dict.

        if self.model_args.mode == 'dual_system':
            sucs, spls, oss, nes, ndtws = self._run_eval_dual_system()
        elif self.model_args.mode == 'system2':
            sucs, spls, oss, nes, ndtws = self._run_eval_system2()
        else:
            raise ValueError(f"Invalid mode: {self.model_args.mode}")

        result = {
            "sucs": sucs,  # shape [N_local]
            "spls": spls,  # shape [N_local]
            "oss": oss,  # shape [N_local]
            "nes": nes,  # shape [N_local]
        }

        if ndtws is not None:
            result["ndtws"] = ndtws  # shape [N_local]
        return result

    def calc_metrics(self, global_metrics: dict) -> dict:
        """
        global_metrics["sucs"] etc. are global 1-D CPU tensors with all episodes.
        """
        sucs_all = global_metrics["sucs"]
        spls_all = global_metrics["spls"]
        oss_all = global_metrics["oss"]
        nes_all = global_metrics["nes"]

        # avoid /0 if no episodes
        denom = max(len(sucs_all), 1)

        # clean NaN in spls, treat as 0.0
        torch.nan_to_num(spls_all, nan=0.0, posinf=0.0, neginf=0.0, out=spls_all)

        # clean inf in nes, only fiinite nes are counted
        nes_finite_mask = torch.isfinite(nes_all)
        nes_all = nes_all[nes_finite_mask]

        result_all = {
            "sucs_all": float(sucs_all.mean().item()) if denom > 0 else 0.0,
            "spls_all": float(spls_all.mean().item()) if denom > 0 else 0.0,
            "oss_all": float(oss_all.mean().item()) if denom > 0 else 0.0,
            "nes_all": float(nes_all.mean().item()) if denom > 0 else 0.0,
            # "length" will be filled by base class
        }

        if "ndtws" in global_metrics:
            ndtws_all = global_metrics["ndtws"]
            result_all["ndtws_all"] = float(ndtws_all.mean().item()) if denom > 0 else 0.0

        return result_all

    def parse_actions(self, output):
        action_patterns = '|'.join(re.escape(action) for action in self.actions2idx)
        # import ipdb; ipdb.set_trace()
        regex = re.compile(action_patterns)
        matches = regex.findall(output)
        actions = [self.actions2idx[match] for match in matches]
        actions = itertools.chain.from_iterable(actions)
        return list(actions)

    def resume_from_output_path(self) -> None:
        sucs, spls, oss, nes, ndtw = [], [], [], [], []
        if self.rank != 0:
            return sucs, spls, oss, nes, ndtw

        # resume from previous results
        if os.path.exists(os.path.join(self.output_path, 'progress.json')):
            with open(os.path.join(self.output_path, 'progress.json'), 'r') as f:
                for line in f.readlines():
                    res = json.loads(line)
                    sucs.append(res['success'])
                    spls.append(res['spl'])
                    oss.append(res['os'])
                    nes.append(res['ne'])
                    if 'ndtw' in res:
                        ndtw.append(res['ndtw'])
        return sucs, spls, oss, nes, ndtw

    def _run_eval_dual_system(self) -> tuple:  # noqa: C901
        # resume from previous results
        sucs, spls, oss, nes, ndtw = self.resume_from_output_path()

        # Episode loop is now driven by env.reset() + env.is_running
        process_bar = tqdm.tqdm(total=len(self.env.episodes), desc=f"Eval Epoch {self.epoch} Rank {self.rank}")

        while self.env.is_running:

            # ------------ 1. Start of episode ------------
            observations = self.env.reset()
            if not self.env.is_running or observations is None:
                break

            self.s1_agent.reset()
            self.s2_agent_reset()

            # Reset for network outage settings
            self.semantic_hold = False
            self.semantic_hold_steps = 0
            self.semantic_hold_last_rgb = None
            self.action_history = []  # 用于跟踪离散动作序列的历史窗口

            # ---- episode meta (scene_id, episode_id, instruction) ----
            # we get it from the underlying habitat env
            episode = self.env.get_current_episode()
            scene_id = episode.scene_id.split('/')[-2]
            episode_id = int(episode.episode_id)
            episode_instruction = episode.instruction.instruction_text
            print("episode start", episode_instruction)

            # save first frame per rank to validate sim quality
            os.makedirs(os.path.join(self.output_path, f'check_sim_{self.epoch}'), exist_ok=True)
            Image.fromarray(observations['rgb']).save(
                os.path.join(self.output_path, f'check_sim_{self.epoch}', f'rgb_{self.rank}.jpg')
            )

            vis_frames = []
            step_id = 0
            s2_step_num = 0
            vis_writer = None

            if self.save_video:
                os.makedirs(os.path.join(self.output_path, f'vis_{self.epoch}', f'{scene_id}'), exist_ok=True)
            if self.vis_debug:
                debug_dir = os.path.join(self.vis_debug_path, f'epoch_{self.epoch}')
                os.makedirs(debug_dir, exist_ok=True)
                vis_writer = imageio.get_writer(
                    os.path.join(debug_dir, f'{scene_id}_{episode_id:04d}.mp4'),
                    fps=5,
                )

            action_seq = []
           
            action = None
            local_actions = []

            done = False
            flag = False
            pixel_goal = None
            
            episode_start_time = time.time()
            # ---------- 2. Episode step loop -----------
            while (not done) and (step_id <= self.max_steps_per_episode):
                self.semantic_hold_steps = 0
                self.semantic_hold_last_rgb = None
                
                self.inference_logger.reset()
                self.inference_logger.record_by_key(system_perf.EPISODE_ID, episode_id)
                self.inference_logger.record_by_key(system_perf.STEP_ID, step_id)

                draw_pixel_goal = False
                # refactor agent get action
                rgb = observations["rgb"]
                depth = observations["depth"]
                x, y = observations["gps"]
                depth = filter_depth(depth.reshape(depth.shape[:2]), blur_type=None)
                depth = depth * (self._max_depth - self._min_depth) + self._min_depth
                depth = depth * 1000

                image = Image.fromarray(rgb).convert('RGB')
                save_raw_image = image.copy()

                self.step_start_time = time.time()
                habitat_time = 0

                if action == action_code.LOOKDOWN:
                    look_down_image = image
                    save_raw_image = look_down_image.copy()
                    look_down_depth, resize_shape = preprocess_depth_image_v2(
                        Image.fromarray(depth.astype(np.uint16), mode='I;16'),
                        do_depth_scale=True,
                        depth_scale=1000,
                        target_height=224,
                        target_width=224,
                    )
                    look_down_depth = torch.as_tensor(np.ascontiguousarray(look_down_depth)).float()
                    look_down_depth[look_down_depth > 5.0] = 5.0
                else:
                    # A. 仅在网络连接时推送边缘流，避免突发断网导致程序挂起
                    if self.network_monitor.connected or step_id == 0:
                        self.s2_agent_step(image)
                    else:
                        print("[Network Outage] Network disconnected. Skipping background S2 stream.")
                        s2_step_num -= 1

                    time_0 = time.time()
                    down_observations, _, _, _ = self.env.step(action_code.LOOKDOWN)
                    down_observations, _, _, _ = self.env.step(action_code.LOOKDOWN)
                    habitat_time += (time.time() - time_0)

                    look_down_image = Image.fromarray(down_observations["rgb"]).convert('RGB')
                    depth = down_observations["depth"]
                    depth = filter_depth(depth.reshape(depth.shape[:2]), blur_type=None)
                    depth = depth * (self._max_depth - self._min_depth) + self._min_depth
                    depth = depth * 1000
                    look_down_depth, resize_shape = preprocess_depth_image_v2(
                        Image.fromarray(depth.astype(np.uint16), mode='I;16'),
                        do_depth_scale=True,
                        depth_scale=1000,
                        target_height=224,
                        target_width=224,
                    )
                    look_down_depth = torch.as_tensor(np.ascontiguousarray(look_down_depth)).float()
                    look_down_depth[look_down_depth > 5.0] = 5.0

                    time_0 = time.time()
                    self.env.step(action_code.LOOKUP)
                    self.env.step(action_code.LOOKUP)
                    habitat_time += (time.time() - time_0)

                # B. 在线网络恢复检测 (在线重连机制)
                if self.network_monitor.connected and self.semantic_hold:
                    print("[Network Recovered] Reconnected to S2 server. Resetting hold state.")
                    self.semantic_hold = False
                    self.semantic_hold_steps = 0
                    pixel_goal = None  # 清空过时目标，强制触发全新 S2 全局规划
                    action_seq = []
                    local_actions = []

                if len(action_seq) == 0 and pixel_goal is None:
                    if self.network_monitor.connected:
                        # 正常在线模式：向边缘服务器请求S2慢模型推理子目标
                        look_down = (action == action_code.LOOKDOWN)
                        s2_response = self.s2_agent_step(look_down_image, episode_instruction, s2_step_num, look_down)

                        traj_latents = s2_response['traj_latents']
                        if traj_latents is not None:
                            time_0 = time.time()
                            self.env.step(action_code.LOOKUP)
                            self.env.step(action_code.LOOKUP)
                            habitat_time += (time.time() - time_0)

                            forward_action = 0
                            draw_pixel_goal = True
                            pixel_goal = s2_response['pixel_goal']

                            s1_response = self.s1_agent_step(look_down_image, look_down_depth, habitat_time, traj_latents)
                            local_actions = s1_response['local_actions']

                            action = local_actions[0]
                            if action == action_code.STOP:
                                self.inference_logger.record_by_key(system_perf.STEP, \
                                    time.time() - self.step_start_time - habitat_time)
                                self.inference_logger.flush()

                                pixel_goal = None
                                action = action_code.LEFT
                                observations, _, done, _ = self.env.step(action)
                                step_id += 1
                                s2_step_num += 1

                                # self.s1_agent.reset()
                                continue
                            print('predicted goal', pixel_goal, flush=True)
                        else:
                            action_seq = s2_response['action_seq']
                            print('actions', action_seq, flush=True)
                    else:
                        print("[Network Outage] S2 server unreachable. Triggering adaptive semantic hold.")
                        # 触发网络中断
                        self.semantic_hold = True

                # D. 多模态动作仲裁器 (含自适应离散动作外推模块)
                if self.semantic_hold:
                    action = self.semantic_hold_infer(look_down_image, look_down_depth, habitat_time)
                    print(f"[Semantic Hold] After semantic hold inference, selected action: {int(action)}.")
                    # 维持断网自治纯净状态，清空残留缓存
                    action_seq = []
                    local_actions = []
                elif len(action_seq) != 0:
                    action = action_seq[0]
                    action_seq.pop(0)
                elif pixel_goal is not None:
                    if len(local_actions) == 0:
                        # 还未到达子目标，需要s1生成新的local actions
                        s1_response = self.s1_agent_step(look_down_image, look_down_depth, habitat_time)
                        local_actions: list = s1_response['local_actions']
                        print("local_actions", local_actions)
                        action = local_actions.pop(0)
                    else:
                        action = local_actions.pop(0)

                    forward_action += 1
                    if forward_action > MAX_STEPS or action == action_code.STOP:
                        self.inference_logger.record_by_key(system_perf.STEP, \
                            time.time() - self.step_start_time - habitat_time)
                        self.inference_logger.flush()
                        # self.s1_agent.reset()

                        pixel_goal = None
                        step_id += 1
                        s2_step_num += 1
                        forward_action = 0
                        local_actions = []
                        continue
                else:
                    action = 0
                
                # 新增：记录每一时刻实际驱动仿真的真实动作到历史，确保风险计算闭环
                self.action_history.append(action)

                time_0 = time.time()
                info = self.env.get_metrics()
                habitat_time += (time.time() - time_0)

                if info['top_down_map'] is not None and self.save_video:
                    frame = observations_to_image({'rgb': np.asarray(save_raw_image)}, info)
                    if pixel_goal is not None and flag:
                        cv2.circle(frame, (pixel_goal[0], pixel_goal[1]), radius=8, color=(255, 0, 0), thickness=-1)
                    vis_frames.append(frame)

                print("step_id", step_id, "action", action)

                if vis_writer is not None:
                    vis = np.asarray(save_raw_image).copy()
                    vis = cv2.putText(
                        vis,
                        f"step {step_id} action {int(action)}",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2,
                    )
                    if pixel_goal is not None:
                        if draw_pixel_goal:
                            vis = cv2.putText(
                                vis,
                                f"{pixel_goal[0], pixel_goal[1]}",
                                (50, 100),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1,
                                (0, 255, 0),
                                2,
                            )
                            cv2.circle(vis, (pixel_goal[0], pixel_goal[1]), radius=8, color=(255, 0, 0), thickness=-1)
                    vis_writer.append_data(vis)

                self.inference_logger.record_by_key(system_perf.STEP, \
                    time.time() - self.step_start_time - habitat_time)
                self.inference_logger.flush()

                if action == action_code.LOOKDOWN:
                    self.env.step(action)
                    observations, _, done, _ = self.env.step(action)
                    flag = True
                else:
                    observations, _, done, _ = self.env.step(action)
                    step_id += 1
                    s2_step_num += 1
                    flag = False

            # ---------- 3. End of episode -----------
            # collect the metric result of this episode and write progress to the output_path/progress.json

            process_bar.update(1)

            # After the episode finishes, collect metrics:
            metrics = self.env.get_metrics()

            sucs.append(metrics['success'])
            spls.append(metrics['spl'])
            oss.append(metrics['oracle_success'])
            nes.append(metrics["distance_to_goal"])
            if 'ndtw' in metrics:
                ndtw.append(metrics["ndtw"])

            print(
                f"scene_episode {scene_id}_{episode_id:04d} success: {metrics['success']}, "
                f"spl: {metrics['spl']}, os: {metrics['oracle_success']}, "
                f"ne: {metrics['distance_to_goal']}"
            )

            # Write per-episode progress.json entry (still per-rank)
            result = {
                "scene_id": scene_id,
                "episode_id": episode_id,
                "success": metrics["success"],
                "spl": metrics["spl"],
                "os": metrics['oracle_success'],
                "ne": metrics["distance_to_goal"],
                "steps": step_id,
                "episode_instruction": episode_instruction,
                "episode_time": time.time() - episode_start_time,
            }
            if 'ndtw' in metrics:
                result['ndtw'] = metrics['ndtw']

            # save current progress
            os.makedirs(self.output_path, exist_ok=True)
            with open(os.path.join(self.output_path, 'progress.json'), 'a') as f:
                f.write(json.dumps(result) + "\n")

            # save video
            if self.save_video and metrics['success'] == 1.0:
                images_to_video(
                    vis_frames,
                    os.path.join(self.output_path, f'vis_{self.epoch}', f'{scene_id}'),
                    f'{episode_id:04d}',
                    fps=6,
                    quality=9,
                )
            vis_frames.clear()
            if vis_writer is not None:
                vis_writer.close()

        self.env.close()

        return (
            torch.tensor(sucs).to(self.device),
            torch.tensor(spls).to(self.device),
            torch.tensor(oss).to(self.device),
            torch.tensor(nes).to(self.device),
            torch.tensor(ndtw).to(self.device) if ndtw else None,
        )

    def _run_eval_system2(self) -> tuple:
        self.model.eval()

        # resume from previous results
        sucs, spls, oss, nes, ndtw = self.resume_from_output_path()

        # Episode loop is now driven by env.reset() + env.is_running
        process_bar = tqdm.tqdm(total=len(self.env.episodes), desc=f"Eval Epoch {self.epoch} Rank {self.rank}")

        while self.env.is_running:

            # ------------ 1. Start of episode ------------
            observations = self.env.reset()
            if not self.env.is_running or observations is None:
                break

            # ---- episode meta (scene_id, episode_id, instruction) ----
            # we get it from the underlying habitat env
            episode = self.env.get_current_episode()
            scene_id = episode.scene_id.split('/')[-2]
            episode_id = int(episode.episode_id)
            episode_instruction = episode.instruction.instruction_text
            print("episode start", episode_instruction)

            agent_state = self.env._env.sim.get_agent_state()
            rotation = agent_state.rotation
            translation = agent_state.position
            rotation_matrix = quaternion.as_rotation_matrix(rotation)
            transformation_matrix = np.eye(4)
            transformation_matrix[:3, :3] = rotation_matrix
            transformation_matrix[:3, 3] = translation

            agent = ShortestPathFollower(self.env._env.sim, 0.25, False)

            intrinsic_matrix = get_intrinsic_matrix(
                self.config.habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor
            )

            # save first frame per rank to validate sim quality
            os.makedirs(os.path.join(self.output_path, f'check_sim_{self.epoch}'), exist_ok=True)
            Image.fromarray(observations['rgb']).save(
                os.path.join(self.output_path, f'check_sim_{self.epoch}', f'rgb_{self.rank}.jpg')
            )

            vis_frames = []
            step_id = 0
            vis_writer = None

            if self.save_video:
                os.makedirs(os.path.join(self.output_path, f'vis_{self.epoch}', f'{scene_id}'), exist_ok=True)
            if self.vis_debug:
                debug_dir = os.path.join(self.vis_debug_path, f'epoch_{self.epoch}')
                os.makedirs(debug_dir, exist_ok=True)
                vis_writer = imageio.get_writer(
                    os.path.join(debug_dir, f'{scene_id}_{episode_id:04d}.mp4'),
                    fps=5,
                )
            initial_height = self.env._env.sim.get_agent_state().position[1]

            rgb_list = []
            action_seq = []
            input_images = []
            output_ids = None
            llm_outputs = ""
            goal = None
            action = None
            messages = []

            done = False
            flag = False

            # ---------- 2. Episode step loop -----------
            while (not done) and (step_id <= self.max_steps_per_episode):
                draw_pixel_goal = False
                # refactor agent get action
                rgb = observations["rgb"]
                depth = observations["depth"]
                x, y = observations["gps"]
                camera_yaw = observations["compass"][0]
                depth = filter_depth(depth.reshape(depth.shape[:2]), blur_type=None)
                depth = depth * (self._max_depth - self._min_depth) + self._min_depth
                depth = depth * 1000

                agent_state = self.env._env.sim.get_agent_state()
                height = agent_state.position[1] - initial_height  # Habitat GPS makes west negative, so flip y
                camera_position = np.array([x, -y, self._camera_height + height])
                tf_camera_to_episodic = (
                    xyz_yaw_pitch_to_tf_matrix(camera_position, camera_yaw, np.deg2rad(30)) @ get_axis_align_matrix()
                )

                image = Image.fromarray(rgb).convert('RGB')
                save_raw_image = image.copy()

                if action == action_code.LOOKDOWN:
                    look_down_image = image
                    save_raw_image = look_down_image.copy()
                else:
                    image = image.resize((self.model_args.resize_w, self.model_args.resize_h))
                    rgb_list.append(image)

                if len(action_seq) == 0 and goal is None:
                    if action == action_code.LOOKDOWN:
                        # last action is look down
                        sources = [{"from": "human", "value": ""}, {"from": "gpt", "value": ""}]
                        input_images += [look_down_image]
                        messages.append(
                            {'role': 'assistant', 'content': [{'type': 'text', 'text': llm_outputs}]}  # noqa: F405
                        )
                        input_img_id = -1
                    else:
                        sources = copy.deepcopy(self.conversation)
                        sources[0]["value"] = sources[0]["value"].replace(
                            '<instruction>.', episode.instruction.instruction_text[:-1]
                        )
                        cur_images = rgb_list[-1:]
                        if step_id == 0:
                            history_id = []
                        else:
                            history_id = np.unique(
                                np.linspace(0, step_id - 1, self.num_history, dtype=np.int32)
                            ).tolist()
                            placeholder = (DEFAULT_IMAGE_TOKEN + '\n') * len(history_id)
                            sources[0]["value"] += f' These are your historical observations: {placeholder}.'

                        history_id = sorted(history_id)
                        input_images = [rgb_list[i] for i in history_id] + cur_images
                        input_img_id = 0

                    prompt = random.choice(self.conjunctions) + DEFAULT_IMAGE_TOKEN
                    sources[0]["value"] += f" {prompt}."
                    prompt_instruction = copy.deepcopy(sources[0]["value"])
                    parts = split_and_clean(prompt_instruction)

                    content = []
                    for i in range(len(parts)):
                        if parts[i] == "<image>":
                            content.append({"type": "image", "image": input_images[input_img_id]})
                            input_img_id += 1
                        else:
                            content.append({"type": "text", "text": parts[i]})

                    messages.append({'role': 'user', 'content': content})

                    text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

                    inputs = self.processor(text=[text], images=input_images, return_tensors="pt").to(self.model.device)

                    with torch.no_grad():
                        output_ids = self.model.generate(
                            **inputs,
                            max_new_tokens=128,
                            do_sample=False,
                            use_cache=True,
                            past_key_values=None,
                            return_dict_in_generate=True,
                        ).sequences

                    llm_outputs = self.processor.tokenizer.decode(
                        output_ids[0][inputs.input_ids.shape[1] :], skip_special_tokens=True
                    )
                    print('step_id:', step_id, 'output text:', llm_outputs)

                    if bool(re.search(r'\d', llm_outputs)):  # output pixel goal
                        forward_action = 0
                        coord = [int(c) for c in re.findall(r'\d+', llm_outputs)]

                        pixel_goal = [int(coord[1]), int(coord[0])]
                        draw_pixel_goal = True

                        # look down --> horizontal
                        self.env.step(action_code.LOOKUP)
                        self.env.step(action_code.LOOKUP)

                        goal = pixel_to_gps(pixel_goal, depth / 1000, intrinsic_matrix, tf_camera_to_episodic)

                        goal = (transformation_matrix @ np.array([-goal[1], 0, -goal[0], 1]))[:3]

                        if not self.env._env.sim.pathfinder.is_navigable(np.array(goal)):
                            goal = np.array(self.env._env.sim.pathfinder.snap_point(np.array(goal)))

                        action = agent.get_next_action(goal)
                        if action == action_code.STOP:
                            goal = None
                            output_ids = None
                            action = action_code.LEFT  # random action to avoid deadlock
                            observations, _, done, _ = self.env.step(action)
                            step_id += 1
                            messages = []
                            continue
                        print('predicted goal', pixel_goal, goal, flush=True)

                    else:
                        action_seq = self.parse_actions(llm_outputs)
                        print('actions', action_seq, flush=True)

                if len(action_seq) != 0:
                    action = action_seq[0]
                    action_seq.pop(0)
                elif goal is not None:
                    action = agent.get_next_action(goal)
                    action = action.detach().cpu().numpy()[0] if isinstance(action, torch.Tensor) else action
                    action = action[0] if hasattr(action, "__len__") else action

                    forward_action += 1
                    if forward_action > MAX_STEPS:
                        goal = None
                        output_ids = None
                        messages = []
                        step_id += 1
                        forward_action = 0
                        continue
                    if action == action_code.STOP:
                        goal = None
                        output_ids = None
                        messages = []
                        step_id += 1
                        forward_action = 0
                        continue
                else:
                    action = 0

                info = self.env.get_metrics()

                if info['top_down_map'] is not None and self.save_video:
                    frame = observations_to_image({'rgb': np.asarray(save_raw_image)}, info)
                    if goal is not None and flag:
                        cv2.circle(frame, (pixel_goal[0], pixel_goal[1]), radius=8, color=(255, 0, 0), thickness=-1)
                    vis_frames.append(frame)

                print("step_id", step_id, "action", action)

                if vis_writer is not None:
                    vis = np.asarray(save_raw_image).copy()
                    vis = cv2.putText(
                        vis,
                        f"step {step_id} action {int(action)}",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2,
                    )
                    if draw_pixel_goal:
                        cv2.circle(vis, (pixel_goal[0], pixel_goal[1]), radius=8, color=(255, 0, 0), thickness=-1)
                    vis_writer.append_data(vis)

                if action == action_code.LOOKDOWN:
                    self.env.step(action)
                    observations, _, done, _ = self.env.step(action)
                    flag = True
                else:
                    observations, _, done, _ = self.env.step(action)
                    step_id += 1
                    messages = []
                    flag = False

            # ---------- 3. End of episode -----------
            # collect the metric result of this episode and write progress to the output_path/progress.json

            process_bar.update(1)

            # After the episode finishes, collect metrics:
            metrics = self.env.get_metrics()

            sucs.append(metrics['success'])
            spls.append(metrics['spl'])
            oss.append(metrics['oracle_success'])
            nes.append(metrics["distance_to_goal"])
            if 'ndtw' in metrics:
                ndtw.append(metrics["ndtw"])

            print(
                f"scene_episode {scene_id}_{episode_id:04d} success: {metrics['success']}, "
                f"spl: {metrics['spl']}, os: {metrics['oracle_success']}, "
                f"ne: {metrics['distance_to_goal']}"
            )

            # Write per-episode result.json entry (still per-rank)
            result = {
                "scene_id": scene_id,
                "episode_id": episode_id,
                "success": metrics["success"],
                "spl": metrics["spl"],
                "os": metrics['oracle_success'],
                "ne": metrics["distance_to_goal"],
                "steps": step_id,
                "episode_instruction": episode_instruction,
            }
            if 'ndtw' in metrics:
                result['ndtw'] = metrics['ndtw']

            os.makedirs(self.output_path, exist_ok=True)
            with open(os.path.join(self.output_path, 'progress.json'), 'a') as f:
                f.write(json.dumps(result) + "\n")
            if self.save_video and metrics['success'] == 1.0:
                images_to_video(
                    vis_frames,
                    os.path.join(self.output_path, f'vis_{self.epoch}', f'{scene_id}'),
                    f'{episode_id:04d}',
                    fps=6,
                    quality=9,
                )
            vis_frames.clear()
            if vis_writer is not None:
                vis_writer.close()

        self.env.close()

        return (
            torch.tensor(sucs).to(self.device),
            torch.tensor(spls).to(self.device),
            torch.tensor(oss).to(self.device),
            torch.tensor(nes).to(self.device),
            torch.tensor(ndtw).to(self.device) if ndtw else None,
        )
