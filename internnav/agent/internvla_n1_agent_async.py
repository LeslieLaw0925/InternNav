import copy
import itertools
import os
import re
import time
from datetime import datetime

import cv2
import imageio
import numpy as np
import torch

from collections import OrderedDict

from PIL import Image
from transformers import AutoProcessor
import ray

from internnav.agent.base import Agent
from internnav.configs.agent import AgentCfg
from internnav.configs.model.base_encoders import ModelCfg
from internnav.model.basemodel.internvla_n1.internvla_n1 import InternVLAN1ForCausalLM, build_navdp
from internnav.model.utils.vln_utils import split_and_clean, traj_to_actions
from internnav.model.utils.misc import set_random_seed


DEFAULT_IMAGE_TOKEN = "<image>"
S1_INFER = "execution"
S2_INFER = "planning"


class SharedObjKeys:
    PIXEL_GOAL = "goal"
    TRAJ_LATENT = "latent"
    PIXEL_GOAL_RGB = "rgb"
    PIXEL_GOAL_DEPTH = "depth"
    STATE = "state"


@ray.remote
class SharedMemory:
    def __init__(self):
        self.store = {} 

    def put(self, key, data):
        """写入共享内存（零拷贝）"""
        # ref = ray.put(data)    # Ray Object Store
        self.store[key] = data

    def get(self, key):
        """返回 ObjectRef(快模型可零拷贝获取)"""
        return self.store.get(key, None)
    
    def get_blocking(self, key):
        while key not in self.store or self.store[key] is None:
            time.sleep(0.001)
        return self.store[key]
    
    def clear(self):
        self.store = {}
    

@Agent.register('internvla_n1_arbiter')
class VLNArbiterAgent(Agent):
    def __init__(self, config: AgentCfg):
        super().__init__(config)
        vln_sensor_config = self.config.model_settings
        _model_settings = ModelCfg(**vln_sensor_config)

        self.shm = SharedMemory.remote()
        self.s1_agent = S1Agent.remote(self.shm, _model_settings)
        self.s2_agent = S2Agent.remote(self.shm, _model_settings)
        self.action_seq: list = []
        self.last_action: int = -1
        self.look_down: bool = False
        self.episode_idx: int = 0

        # vis debug
        self.vis_debug = vln_sensor_config['vis_debug']
        if self.vis_debug:
            self.debug_path = vln_sensor_config['vis_debug_path']
            os.makedirs(self.debug_path, exist_ok=True)
            self.fps_writer = imageio.get_writer(f"{self.debug_path}/fps_{self.episode_idx}_async.mp4", fps=5)
            self.fps_writer2 = imageio.get_writer(f"{self.debug_path}/fps_{self.episode_idx}_async_dp.mp4", fps=5)

    def __del__(self):
        ray.kill(self.s1_agent)
        ray.kill(self.s2_agent)
        ray.kill(self.shm)

    def step(self, obs):
        obs = obs[0]  # do not support batch_env currently?
        rgb = obs['rgb']
        depth = obs['depth']
        instruction = obs['instruction']
        pose = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        
        if self.last_action == 5:
            self.look_down = True
            # 按照internnav的逻辑，此时self.action_seq为None，输出latent_traj
            action_seq_ref = self.s2_agent.step.remote(rgb, depth, pose, instruction, \
                                      self.look_down)
            self.action_seq = ray.get(action_seq_ref)
            self.look_down = False

        if self.action_seq is not None:
            if len(self.action_seq) > 0:
                self.s2_agent.step_no_infer.remote(rgb, depth, pose)
            else:
                action_seq_ref = self.s2_agent.step.remote(rgb, depth, pose, instruction, self.look_down)
                self.action_seq = ray.get(action_seq_ref)
        else:
            action_seq_ref = self.s1_agent.step.remote(rgb, depth)
            self.action_seq = ray.get(action_seq_ref)
            if self.action_seq == []:
                self.action_seq = [-1]
            
        self.last_action = self.action_seq.pop(0)
        output = {'action': [self.last_action]}

        # Visualization
        if self.vis_debug:
            vis = rgb.copy()
            if 'action' in output:
                vis = cv2.putText(vis, str(output['action'][0]), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            self.fps_writer.append_data(vis)
        return [{'action': output['action'], 'ideal_flag': True}]

    def reset(self):
        self.s1_agent.reset.remote()
        self.s2_agent.reset.remote()
        self.shm.clear.remote()
        self.action_seq = []
        self.last_action = -1
        self.look_down = False
        self.episode_idx += 1

        self.s2_agent.reset.remote()
        self.s1_agent.reset.remote()

        if self.vis_debug:
            self.fps_writer.close()
            self.fps_writer2.close()
            self.fps_writer = imageio.get_writer(f"{self.debug_path}/fps_{self.episode_idx}_async.mp4", fps=5)
            self.fps_writer2 = imageio.get_writer(f"{self.debug_path}/fps_{self.episode_idx}_async_dp.mp4", fps=5)


@ray.remote(num_gpus=0.2)
class S1Agent:
    def __init__(self, shm, model_settings: ModelCfg):
        set_random_seed(0)
        self.shm = shm
        self.device = torch.device(model_settings.device)
        self.model = build_navdp(model_settings)
        self.model.eval()
        self.model.to(dtype=torch.bfloat16, device=self.device)
        self.depth_threshold = 5.0

    def step(self, rgb: np.ndarray, depth: np.ndarray):
        traj_latents = ray.get(self.shm.get.remote(SharedObjKeys.TRAJ_LATENT))
        pixel_goal_rgb = ray.get(self.shm.get.remote(SharedObjKeys.PIXEL_GOAL_RGB))
        pixel_goal_depth = ray.get(self.shm.get.remote(SharedObjKeys.PIXEL_GOAL_DEPTH))

        if not all([pixel_goal_rgb is not None, pixel_goal_depth is not None, traj_latents is not None]):
            return None
        
        processed_pixel_rgb = (np.array(Image.fromarray(pixel_goal_rgb).resize((224, 224))) / 255.0)
        processed_pixel_depth = (np.array(Image.fromarray(pixel_goal_depth[:, :, 0]).resize((224, 224))) * 10.0)
        processed_pixel_depth[processed_pixel_depth > self.depth_threshold] = self.depth_threshold
        
        processed_rgb = np.array(Image.fromarray(rgb).resize((224, 224))) / 255.0
        processed_depth = (np.array(Image.fromarray(depth[:, :, 0]).resize((224, 224))) * 10.0)  # should be 0-10m
        processed_depth[processed_depth > self.depth_threshold] = self.depth_threshold
        
        rgbs = (
            torch.stack([torch.from_numpy(processed_pixel_rgb), torch.from_numpy(processed_rgb)])
            .unsqueeze(0)
            .to(self.device)
        )  # [1, 2, 224, 224, 3]
        depths = (
            torch.stack([torch.from_numpy(processed_pixel_depth), torch.from_numpy(processed_depth)])
            .unsqueeze(0)
            .unsqueeze(-1)
            .to(self.device)
        )  # [1, 2, 224, 224, 1]

        dp_actions = self.step_s1(traj_latents, rgbs, depths, use_async=True)
        action_list = traj_to_actions(dp_actions)
        action_list = [x for x in action_list if x != 0]
        return action_list[:4]

    def step_s1(self, traj_latents, images_dp=None, depths_dp=None, use_async=False):
        if use_async:
            all_trajs = self.model.predict_pointgoal_action_async(
                traj_latents.to(self.device), images_dp, depths_dp, vlm_mask=None
            )
        else:
            all_trajs = self.model.predict_pointgoal_action(
                traj_latents.to(self.device), vlm_mask=None
            )
        return all_trajs
    
    def reset(self):
        pass


@ray.remote(num_gpus=0.8)
class S2Agent:
    def __init__(self, shm, model_settings: ModelCfg):
        set_random_seed(0)
        self.shm = shm
        self.device = torch.device(model_settings.device)
        print(f"args.model_path{model_settings.model_path}")
        self.model = InternVLAN1ForCausalLM.from_pretrained(
            model_settings.model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map={"": self.device},
        )
        self.model.eval()
        self.model.to(self.device)

        self.processor = AutoProcessor.from_pretrained(model_settings.model_path)
        self.processor.tokenizer.padding_side = 'left'

        self.resize_w = model_settings.resize_w
        self.resize_h = model_settings.resize_h
        self.num_history = model_settings.num_history
        # self.PLAN_STEP_GAP = model_settings.plan_step_gap
        self.intrinsic = self.get_intrinsic_matrix(
            model_settings.width, model_settings.height, model_settings.hfov
        )

        prompt = "You are an autonomous navigation assistant. Your task is to <instruction>. Where should you go next to stay on track? Please output the next waypoint's coordinates in the image. Please output STOP when you have successfully completed the task."
        answer = ""
        self.conversation = [{"from": "human", "value": prompt}, {"from": "gpt", "value": answer}]
        self.conjunctions = [
            'you can see ',
            'in front of you is ',
            'there is ',
            'you can spot ',
            'you are toward the ',
            'ahead of you is ',
            'in your sight is ',
        ]

        self.actions2idx = OrderedDict(
            {
                'STOP': [0],
                "↑": [1],
                "←": [2],
                "→": [3],
                "↓": [5],
            }
        )

        self.rgb_list = []
        self.depth_list = []
        self.pose_list = []
        self.episode_idx = 0
        self.conversation_history = []
        self.llm_output = ""
        self.past_key_values = None
        self.last_s2_idx = -100

        # output
        self.output_action = None
        self.output_latent = None
        self.output_pixel = None
        self.pixel_goal_rgb = None
        self.pixel_goal_depth = None

    def get_intrinsic_matrix(self, width, height, hfov) -> np.ndarray:
        width = width
        height = height
        fov = hfov
        fx = (width / 2.0) / np.tan(np.deg2rad(fov / 2.0))
        fy = fx  # Assuming square pixels (fx = fy)
        cx = (width - 1.0) / 2.0
        cy = (height - 1.0) / 2.0

        intrinsic_matrix = np.array(
            [[fx, 0.0, cx, 0.0], [0.0, fy, cy, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
        )
        return intrinsic_matrix

    def reset(self):
        self.rgb_list = []
        self.depth_list = []
        self.pose_list = []
        self.episode_idx = 0
        self.conversation_history = []
        self.llm_output = ""
        self.past_key_values = None

        self.output_action = None
        self.output_latent = None
        self.output_pixel = None
        self.pixel_goal_rgb = None
        self.pixel_goal_depth = None

        # self.save_dir = "test_data/" + datetime.now().strftime("%Y%m%d_%H%M%S")
        # os.makedirs(self.save_dir, exist_ok=True)

    def parse_actions(self, output):
        action_patterns = '|'.join(re.escape(action) for action in self.actions2idx)
        regex = re.compile(action_patterns)
        matches = regex.findall(output)
        actions = [self.actions2idx[match] for match in matches]
        actions = itertools.chain.from_iterable(actions)
        return list(actions)

    def step_no_infer(self, rgb, depth, pose):
        image = Image.fromarray(rgb).convert('RGB')
        image = image.resize((self.resize_w, self.resize_h))
        self.rgb_list.append(image)
        # image.save(f"{self.save_dir}/debug_raw_{self.episode_idx: 04d}.jpg")
        self.episode_idx += 1

    def trajectory_tovw(self, trajectory, kp=1.0):
        subgoal = trajectory[-1]
        linear_vel, angular_vel = kp * np.linalg.norm(subgoal[:2]), kp * subgoal[2]
        linear_vel = np.clip(linear_vel, 0, 0.5)
        angular_vel = np.clip(angular_vel, -0.5, 0.5)
        return linear_vel, angular_vel

    def step(self, rgb, depth, pose, instruction, look_down=False):
        self.output_action, self.output_latent, self.output_pixel = self.step_s2(
            rgb, depth, pose, instruction, self.intrinsic, look_down
        )
        if self.output_action is not None:
            return self.output_action
        elif self.output_latent is not None:
            self.shm.put.remote(SharedObjKeys.PIXEL_GOAL, copy.deepcopy(self.output_pixel))
            self.shm.put.remote(SharedObjKeys.PIXEL_GOAL_RGB, copy.deepcopy(rgb))
            self.shm.put.remote(SharedObjKeys.PIXEL_GOAL_DEPTH, copy.deepcopy(depth))
            self.output_latent = self.output_latent.detach().cpu()
            self.shm.put.remote(SharedObjKeys.TRAJ_LATENT, self.output_latent)
            print(f"S2 put latent traj to shm.")
            return None

    def step_s2(self, rgb, depth, pose, instruction, intrinsic, look_down=False):
        image = Image.fromarray(rgb).convert('RGB')
        if not look_down:
            image = image.resize((self.resize_w, self.resize_h))
            self.rgb_list.append(image)

            self.conversation_history = []
            self.past_key_values = None

            sources = copy.deepcopy(self.conversation)
            sources[0]["value"] = sources[0]["value"].replace('<instruction>.', instruction)
            cur_images = self.rgb_list[-1:]
            if self.episode_idx == 0:
                history_id = []
            else:
                history_id = np.unique(np.linspace(0, self.episode_idx - 1, self.num_history, dtype=np.int32)).tolist()
                placeholder = (DEFAULT_IMAGE_TOKEN + '\n') * len(history_id)
                sources[0]["value"] += f' These are your historical observations: {placeholder}.'

            history_id = sorted(history_id)
            self.input_images = [self.rgb_list[i] for i in history_id] + cur_images
            input_img_id = 0
            self.episode_idx += 1
        else:
            self.input_images.append(image)
            input_img_id = -1
            assert self.llm_output != "", "Last llm_output should not be empty when look down"
            sources = [{"from": "human", "value": ""}, {"from": "gpt", "value": ""}]
            self.conversation_history.append(
                {'role': 'assistant', 'content': [{'type': 'text', 'text': self.llm_output}]}
            )

        prompt = self.conjunctions[0] + DEFAULT_IMAGE_TOKEN
        sources[0]["value"] += f" {prompt}."
        prompt_instruction = copy.deepcopy(sources[0]["value"])
        parts = split_and_clean(prompt_instruction)

        content = []
        for i in range(len(parts)):
            if parts[i] == "<image>":
                content.append({"type": "image", "image": self.input_images[input_img_id]})
                input_img_id += 1
            else:
                content.append({"type": "text", "text": parts[i]})

        self.conversation_history.append({'role': 'user', 'content': content})

        text = self.processor.apply_chat_template(self.conversation_history, tokenize=False, add_generation_prompt=True)

        inputs = self.processor(text=[text], images=self.input_images, return_tensors="pt").to(self.device)
        t0 = time.time()
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=False,
                use_cache=True,
                past_key_values=self.past_key_values,
                return_dict_in_generate=True,
                raw_input_ids=copy.deepcopy(inputs.input_ids),
            )
        output_ids = outputs.sequences

        t1 = time.time()
        self.llm_output = self.processor.tokenizer.decode(
            output_ids[0][inputs.input_ids.shape[1] :], skip_special_tokens=True
        )
       
        self.last_output_ids = copy.deepcopy(output_ids[0])
        self.past_key_values = copy.deepcopy(outputs.past_key_values)
        print(f"output {self.episode_idx}  {self.llm_output} cost: {t1 - t0}s")
        if bool(re.search(r'\d', self.llm_output)):
            coord = [int(c) for c in re.findall(r'\d+', self.llm_output)]
            pixel_goal = [int(coord[1]), int(coord[0])]
            image_grid_thw = torch.cat([thw.unsqueeze(0) for thw in inputs.image_grid_thw], dim=0)
            pixel_values = inputs.pixel_values
            t0 = time.time()
            with torch.no_grad():
                traj_latents = self.model.generate_latents(output_ids, pixel_values, image_grid_thw)
                return None, traj_latents, pixel_goal

        else:
            action_seq = self.parse_actions(self.llm_output)
            return action_seq, None, None

