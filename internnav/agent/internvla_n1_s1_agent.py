import time

import cv2
import numpy as np
import torch
from PIL import Image
# import torch.nn.functional as F
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.utils.torch_utils import randn_tensor

from internnav.configs.agent import NewAgentCfg
from internnav.model.utils.vln_utils import traj_to_actions
from internnav.model.basemodel.internvla_n1.internvla_n1_arch import AsyncInternVLAN1MetaModel
from internnav.utils.common_log_util import common_logger as log
from internnav.utils.comm_utils.client_utils import find_optimal_config


class System1:
    def __init__(self, config: NewAgentCfg, 
                 latency_profile: dict, 
                 infer_logger=None,
                 device="cuda", 
                 dtype=torch.float16):
        from internnav.model.utils.misc import set_random_seed
        set_random_seed(0)

        vln_sensor_config = config.model_settings
        self.latency_profile = latency_profile
        self.set_adaptive_speedup = vln_sensor_config.get('adaptive_speedup', False)
        self.infer_logger = infer_logger
        self.device = device
        self.dtype = dtype
        self.config = dict(system1=vln_sensor_config.get('s1_type'),
                           navdp_pretrained=vln_sensor_config.get('navdp_pretrained'),
                           nextdit_pretrained=vln_sensor_config.get('nextdit_pretrained')
                           )
        self.model = AsyncInternVLAN1MetaModel(self.config)

        if 'nextdit' in self.config['system1']:
            self.model.load_state_dict(
                torch.load(vln_sensor_config.get('nextdit_pretrained'), map_location="cpu"))
            self.model.to(self.device, self.dtype)

            self.infer_step_range = [2, 4, 8, 10]
            self.traj_num_range = [8, 16, 24, 32]
            self.alpha = self.latency_profile.get('nextdit').get('alpha')
            self.beta = self.latency_profile.get('nextdit').get('beta')
            self.intercept = self.latency_profile.get('nextdit').get('c')
        elif 'navdp' in self.config['system1']:
            self.model.navdp.to(self.device, self.dtype)

            self.infer_step_range = [2, 8, 16, 20]
            self.traj_num_range = [8, 16, 24, 32]
            self.alpha = self.latency_profile.get('navdp').get('alpha')
            self.beta = self.latency_profile.get('navdp').get('beta')
            self.intercept = self.latency_profile.get('navdp').get('c')
        else:
            raise NotImplementedError
        
        self.depth_threshold = 5.0
        self.sys1_forward_step = 4

        self.action_list = []
        self.pixel_goal_rgb = None # record the corresponding pixel goal rgb when S2 finds the pixel goal
        self.pixel_goal_depth = None # record the corresponding pixel goal depth when S2 finds the pixel goal
        self.traj_latents = None # record the corresponding traj_latents when S2 finds the pixel goal
        self.ready_to_reach_goal = False

    def record_goal_obs(self, obs, traj_latents: torch.tensor):
        self.pixel_goal_rgb = obs.get('rgb')
        self.pixel_goal_depth = obs.get('depth', None)
        self.traj_latents = traj_latents.to(self.dtype)

    def step(self, obs: dict) -> dict[str, list]:
        if len(self.action_list) > 0:
            action = self.action_list.pop(0)
            return {'action': [{'action': [action], 'ideal_flag': True}]}
       
        if not all([self.pixel_goal_rgb is not None, 
                    self.traj_latents is not None]):
            raise ValueError("Missing required observation for System1 step.")
        
        start_time = time.time()
        
        if self.set_adaptive_speedup:
            latency_constraint = obs.pop('latency_constraint')
            log.info(f"[CONSTRAINT] Latency constraint for s1 step: {latency_constraint:.4f} seconds.")
            if latency_constraint <= 0:
                best_config = {'infer_step': self.infer_step_range[0], 
                            'traj_num': self.traj_num_range[0]}
            else:
                best_config = find_optimal_config(latency_constraint, 
                                                  self.alpha, 
                                                  self.beta, 
                                                  self.intercept, 
                                                  self.infer_step_range,
                                                  self.traj_num_range)
        else:
            s1_config = obs.get('infer_config', None)
            if s1_config is not None:
                best_config = {'infer_step': s1_config[0],
                               'traj_num': s1_config[1]}
            else:
                best_config = {'infer_step': self.infer_step_range[-1],
                            'traj_num': self.traj_num_range[-1]}
        log.info(f"Chosen config for System1 inference: {best_config}")

        return self.s1_infer(obs, best_config, start_time)

    def s1_infer(self, obs: dict, config: dict, start_time) -> dict[str, list]:
        rgb = obs.get('rgb', None)
        depth = obs.get('depth', None)
        
        processed_pixel_rgb = (np.array(Image.fromarray(self.pixel_goal_rgb).resize((224, 224))) / 255.0)
        processed_rgb = np.array(Image.fromarray(rgb).resize((224, 224))) / 255.0
        rgbs = (
            torch.stack([torch.from_numpy(processed_pixel_rgb), torch.from_numpy(processed_rgb)])
            .unsqueeze(0)
            .to(self.device, self.dtype)
        )  # [1, 2, 224, 224, 3]

        if depth is not None:
            processed_pixel_depth = (np.array(Image.fromarray(self.pixel_goal_depth[:, :, 0]).resize((224, 224))) * 10.0)
            processed_pixel_depth[processed_pixel_depth > self.depth_threshold] = self.depth_threshold

            processed_depth = (np.array(Image.fromarray(depth[:, :, 0]).resize((224, 224))) * 10.0)  # should be 0-10m
            processed_depth[processed_depth > self.depth_threshold] = self.depth_threshold

            depths = (
                torch.stack([torch.from_numpy(processed_pixel_depth), torch.from_numpy(processed_depth)])
                .unsqueeze(0)
                .unsqueeze(-1)
                .to(self.device, self.dtype)
            )  # [1, 2, 224, 224, 1]
        else:
            depths = None

        # # For visualization and debugging
        # self._draw_traj_img(rgb, rgbs, depths)
        # import pdb; pdb.set_trace()
        
        with torch.no_grad():
            dp_actions = self.step_s1(self.traj_latents, rgbs, depths_dp=depths, 
                                        num_inference_steps=config['infer_step'], 
                                        num_sample_trajs=config['traj_num'])
        log.info(f"[TIME] On-device system1 step time: {time.time() - start_time:.4f} seconds.")
                
        action_list = traj_to_actions(dp_actions)
        action_list = [x for x in action_list if x != 0]

        if action_list == []:
            action_list = [-1]
        else:
            action_list = action_list[:4]

        self.action_list = action_list
        if len(self.action_list) < self.sys1_forward_step:
            self.ready_to_reach_goal = True

        self.infer_logger.record_by_key("s1_time", time.time() - start_time)
        return {'action': [{'action': [self.action_list.pop(0)], 'ideal_flag': True}]}

    def step_s1(
        self,
        traj_latents,
        images_dp,
        depths_dp=None,
        predict_step_nums=32,
        guidance_scale: float = 1.0,
        num_inference_steps: int = 10, 
        num_sample_trajs: int = 32, 
    ):              
        if 'nextdit' in self.config['system1']:
            return self.nextdit_step(traj_latents,
                                     images_dp,
                                     predict_step_nums=predict_step_nums,
                                     guidance_scale=guidance_scale,
                                     num_inference_steps=num_inference_steps,
                                     num_sample_trajs=num_sample_trajs,
                                     )
        elif 'navdp' in self.config['system1']:
            if 'async' in self.config['system1']:
                all_trajs = self.model.navdp.predict_pointgoal_action_async(
                    traj_latents, images_dp, depths_dp, 
                    infer_step=num_inference_steps, 
                    sample_num=num_sample_trajs,
                )
            else:
                all_trajs = self.model.navdp.predict_pointgoal_action(traj_latents)
            return all_trajs
    
    def nextdit_step(self, traj_latents,
                     images_dp,
                     predict_step_nums=32,
                     guidance_scale: float = 1.0,
                     num_inference_steps: int = 10,
                     num_sample_trajs: int = 32,
                    ):
        scheduler = FlowMatchEulerDiscreteScheduler()
        device = traj_latents.device
        dtype = traj_latents.dtype

        traj_latents = self.model.cond_projector(traj_latents)
        if 'async' in self.config['system1']:
            with torch.no_grad():
                images_dp = images_dp.permute(0, 1, 4, 2, 3)
                images_dp_norm = (images_dp - self.model._resnet_mean) / self.model._resnet_std
                images_dp_feat = (
                    self.model.rgb_model.get_intermediate_layers(images_dp_norm.flatten(0, 1).to(dtype))[0]
                    .unflatten(dim=0, sizes=(1, -1))
                )
                memory_feat = self.model.memory_encoder(
                    images_dp_feat.flatten(1, 2)
                )  # [bs*select_size,512,384]
                memory_feat = torch.cat([images_dp_feat.flatten(1, 2), memory_feat], dim=-1)
                memory_tokens = self.model.rgb_resampler(memory_feat)
            hidden_states = torch.cat([memory_tokens, traj_latents], dim=1)
        else:
            hidden_states = traj_latents
        hidden_states_null = torch.zeros_like(hidden_states, device=device, dtype=dtype)
        hidden_states_input = torch.cat([hidden_states_null, hidden_states], 0)
        batch_size = traj_latents.shape[0]
        latent_size = predict_step_nums
        latent_channels = 3

        latents = randn_tensor(
            shape=(batch_size * num_sample_trajs, latent_size, latent_channels),
            generator=None,
            device=device,
            dtype=dtype,
        )

        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
        scheduler.set_timesteps(num_inference_steps, sigmas=sigmas)

        hidden_states_input = hidden_states_input.repeat_interleave(num_sample_trajs, dim=0)

        for t in scheduler.timesteps:
            latent_features = self.model.action_encoder(latents)
            pos_ids = (
                torch.arange(latent_features.shape[1])
                .reshape(1, -1)
                .repeat(batch_size, 1)
                .to(latent_features.device)
            )
            pos_embed = self.model.pos_encoding(pos_ids)
            latent_features += pos_embed  # [num_sample_trajs, t, 384]
            latent_model_input = latent_features.repeat(2, 1, 1)
            if hasattr(scheduler, "scale_model_input"):
                latent_model_input = scheduler.scale_model_input(latent_model_input, t)

            # predict noise model_output
            noise_pred = self.model.traj_dit(
                x=latent_model_input,
                timestep=t.unsqueeze(0)
                .expand(latent_model_input.shape[0])
                .to(latent_model_input.device, torch.long),
                z_latents=hidden_states_input,
            )

            noise_pred = self.model.action_decoder(noise_pred)

            # perform guidance
            noise_pred_uncond, noise_pred = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred - noise_pred_uncond)

            # compute previous: x_t -> x_t-1
            latents = scheduler.step(noise_pred, t, latents).prev_sample
        return latents.detach()

    def reset(self, reset_index=None):
        self.action_list = []
        self.pixel_goal_rgb = None
        self.pixel_goal_depth = None
        self.traj_latents = None
        self.ready_to_reach_goal = False

    def _draw_traj_img(self, rgb, rgbs, depths):
        with torch.no_grad():
            self.step_s1(self.traj_latents, rgbs, depths_dp=depths, 
                        num_inference_steps=10, 
                        num_sample_trajs=10)
                    
        trajectories_list, infer_configs, infer_latency = [], [], []
        for step_num in self.infer_step_range:
            for traj_num in self.traj_num_range:
                start_time = time.time()
                with torch.no_grad():
                    dp_actions = self.step_s1(self.traj_latents, rgbs, depths_dp=depths, 
                                                num_inference_steps=step_num, 
                                                num_sample_trajs=traj_num)
                infer_latency.append(time.time() - start_time)
                log.info(f'[TIME] Actual s1 step time: {time.time() - start_time:.2f} s')
                
                action_list = traj_to_actions(dp_actions, use_discrate_action=False)
                trajectories_list.append(action_list)
                infer_configs.append((step_num, traj_num))

        visualize_multiple_trajectories(rgb, trajectories_list, infer_configs, infer_latency)


def visualize_multiple_trajectories(image, trajectories_list, infer_configs=None, infer_latency=None, titles=None, suffix=''):
    import matplotlib.pyplot as plt

    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42

    """
    image: 原始图像 (numpy array)
    trajectories_list: 轨迹列表，例如 [traj1, traj2, ...]
    infer_configs: 推理配置列表
    infer_latency: 推理延迟列表
    titles: 每个轨迹图的标题列表，例如 ['Config A', 'Config B']
    """
    num_trajs = len(trajectories_list)
    # 总列数 = 1 (原图) + 轨迹数量
    num_cols = 1 + num_trajs
    
    traj_width_inch = 1.1
    # 原图宽度（可以单独设大一点）
    img_width_inch = 6
    # 总宽度
    fig_width = img_width_inch + traj_width_inch * num_trajs
    
    # 设置 figure
    fig, axes = plt.subplots(
        1, num_cols,
        figsize=(fig_width, 4),
        gridspec_kw={
            'width_ratios': [img_width_inch] + [traj_width_inch] * num_trajs
        }
    )

    # --- 1. 绘制原图 ---
    # OpenCV 默认是 BGR，matplotlib 需要 RGB
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    axes[0].imshow(img_rgb)
    axes[0].set_title("Subgoal Image", fontsize=15)
    axes[0].axis('off') # 隐藏坐标轴
    
    pad_inch = 0.015
    # --- 2. 循环绘制每个轨迹 ---
    for i, trajectory in enumerate(trajectories_list):
        ax = axes[i + 1]
        pos = ax.get_position()
        ax.set_position([pos.x0 + pad_inch, pos.y0, pos.width, pos.height])
        
        # 提取点坐标
        traj_points = np.array([[float(p[0]), float(p[1])] for p in trajectory if len(p) >= 2])
        
        if len(traj_points) > 0:
            x_coords = traj_points[:, 0]
            y_coords = traj_points[:, 1]
            
            # 绘制逻辑（保留你原有的样式）
            ax.set_facecolor('lightgray')
            l1, = ax.plot(y_coords, x_coords, 'b-', linewidth=2, label='Trajectory')
            l2, = ax.plot(y_coords[0], x_coords[0], 'go', markersize=8, label='Start')
            l3, = ax.plot(y_coords[-1], x_coords[-1], 'ro', markersize=8, label='End')
            l4, = ax.plot(0, 0, '+', color='black', markersize=10, markeredgewidth=2, label='Origin')

            if i == 0: # 只记录第一组句柄用于图例
                lines = [l1, l2, l3, l4]
                labels = ['Trajectory', 'Start Point', 'End Point', 'Origin Point']

            # 坐标轴精简化
            ax.invert_xaxis()
            ax.grid(True, alpha=0.3, linewidth=0.5)
            
            # 【关键】精简刻度以压缩空间
            ax.set_xlabel('Y (left +)', fontsize=10, fontweight='bold')
            if i == 0:
                ax.set_ylabel('X (up +)', fontsize=12, fontweight='bold')
            else:
                ax.set_yticklabels([]) # 隐藏中间轨迹图的纵坐标数字，节省空间
            ax.tick_params(labelsize=8)
            
            if titles:
                ax.set_title(titles[i], fontsize=12)
            else:
                ax.set_title(f"S = {infer_configs[i][0]}\nD = {infer_configs[i][1]}", fontsize=12)

    if infer_latency is not None:
        # 检查数据长度
        if len(infer_latency) != num_trajs:
            print(f"Error: Latency data length ({len(infer_latency)}) does not match number of trajectories ({num_trajs})")
            return

        # 获取轨迹子图区域的整体边界（从 axes[1] 到 axes[-1]）
        pos_first = axes[1].get_position()
        pos_last = axes[-1].get_position()
        
        # 计算轨迹子图区域的总宽度（Figure 相对单位 0-1）
        total_width = pos_last.x1 - pos_first.x0
        total_height = pos_first.y1 - pos_first.y0
        
        # 创建一个跨越所有轨迹子图的透明大坐标轴
        # [left, bottom, width, height]
        big_ax = fig.add_axes([pos_first.x0, pos_first.y0, total_width, total_height], frameon=False)
        
        # 彻底隐藏大坐标轴的X轴内容，避免干扰
        big_ax.set_xticks([])
        
        # 用于保存每个子图中心在大坐标轴中的相对X位置
        # 这里要包含所有 num_trajs 个点！
        x_centers_in_big_ax = []

        for i in range(1, num_cols):
            ax = axes[i]
            
            # 子图中心（display坐标）
            center_disp = ax.transAxes.transform((0.5, 0))  # (x=中点, y随便)
            
            # 转换到 big_ax 坐标系
            center_in_big = big_ax.transAxes.inverted().transform(center_disp)
            x_centers_in_big_ax.append(center_in_big[0])
        
        # 绘制时延折线 (橙色方块，醒目并加粗)
        l5, = big_ax.plot(x_centers_in_big_ax, infer_latency, color='#A03A13', 
                          marker='s', linewidth=2.5, markersize=7, 
                          markerfacecolor='white', markeredgewidth=2,
                          label='Latency', alpha=0.9, zorder=101)
        
        # 将大坐标轴设置在最上层，防止折线被子图背景遮挡
        big_ax.set_zorder(100)
        
        # 设置右侧坐标轴
        big_ax.yaxis.tick_right()
        big_ax.yaxis.set_label_position("right")
        big_ax.set_ylabel('Latency (s)', color='#A03A13', fontsize=12, fontweight='bold')
        big_ax.tick_params(axis='y', colors='#A03A13', labelsize=9)
        
        # 动态范围，防止折线紧贴边缘
        big_ax.set_ylim(0, max(infer_latency)*1.2)
        
        # 锁定 X 轴范围
        big_ax.set_xlim(0, 1)

        # 将时延加入图例
        lines.append(l5)
        labels.append('Latency')

    # 绘制全局图例，放在 Figure 正上方
    fig.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.5, 1.15),
               ncol=5, fontsize=18, frameon=True)
    
    # 保存结果
    save_path = f'logs/comparison_{int(time.time())}{suffix}.pdf'
    plt.savefig(save_path, bbox_inches='tight')
    print(f"Saved comparison to {save_path}")