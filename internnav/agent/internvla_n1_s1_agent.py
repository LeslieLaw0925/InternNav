import time

import cv2
import numpy as np
import torch
from PIL import Image
import torch.nn.functional as F
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.utils.torch_utils import randn_tensor

from internnav.configs.agent import NewAgentCfg
from internnav.model.utils.vln_utils import traj_to_actions
from internnav.model.basemodel.internvla_n1.internvla_n1_arch import AsyncInternVLAN1MetaModel
from internnav.utils.common_log_util import common_logger as log
from internnav.utils.comm_utils.client_utils import find_optimal_config


class System1:
    def __init__(self, config: NewAgentCfg, latency_profile: dict, device="cuda", dtype=torch.float16):
        from internnav.model.utils.misc import set_random_seed
        set_random_seed(0)

        vln_sensor_config = config.model_settings
        self.latency_profile = latency_profile
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
        rgb = obs.get('rgb', None)
        depth = obs.get('depth', None)
        self.latency_constraint = obs.get('latency_constraint', None)
        
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

        best_config = find_optimal_config(self.latency_constraint, self.alpha, 
                                          self.beta, self.intercept, 
                                          self.infer_step_range,
                                          self.traj_num_range)
        with torch.no_grad():
            dp_actions, _ = self.step_s1(self.traj_latents, rgbs, depths_dp=depths, 
                                         num_inference_steps=best_config['infer_step'], 
                                         num_sample_trajs=best_config['traj_num'])
        log.info(f'[TIME] On-device system1 step time: {time.time() - start_time:.2f} s')
        
        action_list, traj_var = traj_to_actions(dp_actions)
        action_list = [x for x in action_list if x != 0]

        log.info(f"Trajectory variance: {traj_var:.4f}.")

        if action_list == []:
            action_list = [-1]
        else:
            action_list = action_list[:4]

        self.action_list = action_list
        if len(self.action_list) < self.sys1_forward_step:
            self.ready_to_reach_goal = True

        return {'action': [{'action': [self.action_list.pop(0)], 'ideal_flag': True}]}

    def evaluate_latent(self, traj_latents: torch.Tensor, img_token: torch.Tensor):
        # 不确定性估计（非常关键）
        with torch.no_grad():
            traj_uncertainty = traj_latents.std(dim=1).mean(dim=-1).item()  # [B]
            img_token_uncertainty = img_token.std(dim=1).mean(dim=-1).item()  # [B]

        uncertainty = 0.7 * traj_uncertainty + 0.3 * img_token_uncertainty

        # normalize 到 0~1
        # u = torch.clamp(uncertainty / 0.5, 0, 1).item()

        # # 动态分配（你可以调范围）
        # min_steps, max_steps = 2, 10
        # min_trajs, max_trajs = 8, 32

        # num_inference_steps = int((min_steps + u.mean() * (max_steps - min_steps)).item())
        # num_sample_trajs = int((min_trajs + u.mean() * (max_trajs - min_trajs)).item())

        # # 防御
        # num_inference_steps = max(2, num_inference_steps)
        # num_sample_trajs = max(8, num_sample_trajs)

        return traj_uncertainty, img_token_uncertainty

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
            return all_trajs, None
    
    def nextdit_step(self, traj_latents,
                     images_dp,
                     predict_step_nums=32,
                     guidance_scale: float = 1.0,
                     num_inference_steps: int = 10, # 10,
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
                score = self.evaluate_latent(traj_latents, memory_tokens)
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
        return latents.detach(), score
    
    def reset(self, reset_index=None):
        self.action_list = []
        self.pixel_goal_rgb = None
        self.pixel_goal_depth = None
        self.traj_latents = None
        self.ready_to_reach_goal = False