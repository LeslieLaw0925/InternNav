import time

import numpy as np
import torch
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.utils.torch_utils import randn_tensor
import yaml

from internnav.agent.base import Agent
from internnav.configs.agent import NewAgentCfg
from internnav.model.utils.vln_utils import traj_to_actions
from internnav.model.basemodel.internvla_n1.internvla_n1_arch import AsyncInternVLAN1MetaModel
from internnav.utils.common_log_util import common_logger as log
from internnav.utils.comm_utils.client_utils import find_optimal_config
from internnav.model.utils.misc import set_random_seed


MAX_STEPS = 8
MAX_LOCAL_STEPS = 4


@Agent.register('habitat_s1_agent')
class System1(Agent):
    def __init__(self, config: NewAgentCfg):
        super().__init__(config)
        
        set_random_seed(0)
        vln_sensor_config = config.model_settings

        with open('scripts/eval/configs/latency_profile.yaml', 'r', encoding='utf-8') as f:
            profile: dict = yaml.safe_load(f)
        self.latency_profile = profile.get('s1_infer')

        self.set_adaptive_speedup = vln_sensor_config.get('adaptive_speedup', False)
        self.device = "cuda"
        self.dtype = torch.float16
        self.config = dict(system1=vln_sensor_config.get('s1_type'),
                           navdp_pretrained=vln_sensor_config.get('navdp_pretrained'),
                           nextdit_pretrained=vln_sensor_config.get('nextdit_pretrained')
                           )
        self.model = AsyncInternVLAN1MetaModel(self.config)

        if 'nextdit' in self.config['system1']:
            self.model.load_state_dict(
                torch.load(vln_sensor_config.get('nextdit_pretrained'), map_location="cpu"))
            self.model.to(self.device, self.dtype)

            self.infer_step_range = [2, 4, 6, 8, 10]
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
        
        self.pixel_goal_rgb = None # record the corresponding pixel goal rgb when S2 finds the pixel goal
        self.pixel_goal_depth = None # record the corresponding pixel goal depth when S2 finds the pixel goal
        self.traj_latents = None # record the corresponding traj_latents when S2 finds the pixel goal

    def record_goal_obs(self, rgb, depth, traj_latents):
        self.pixel_goal_rgb = torch.tensor(np.array(rgb.resize((224, 224)))).to(self.dtype) / 255
        if depth is not None:
            self.pixel_goal_depth = depth.unsqueeze(-1).to(self.dtype)
        self.traj_latents = torch.from_numpy(np.array(traj_latents)).to(self.device, self.dtype)

    def step(self, obs: dict):
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
            best_config = {'infer_step': self.infer_step_range[-1],
                           'traj_num': self.traj_num_range[-1]}
        log.info(f"Chosen config for System1 inference: {best_config}")

        return self.s1_infer(obs, best_config, start_time)

    def s1_infer(self, obs: dict, config: dict, start_time) -> dict[str, list]:
        look_down_image = obs.get('rgb')
        look_down_depth = obs.get('depth', None)
        traj_latents = obs.get('traj_latents', None)
        if traj_latents is not None:
            self.record_goal_obs(look_down_image, look_down_depth, traj_latents)

        if not all([self.pixel_goal_rgb is not None, 
                    self.traj_latents is not None]):
            raise ValueError("Missing required observation for System1 step.")
        
        image_dp = torch.tensor(np.array(look_down_image.resize((224, 224)))).to(self.dtype) / 255
        images_dp = torch.stack([self.pixel_goal_rgb, image_dp]).unsqueeze(0).to(self.device)

        if look_down_depth is not None:
            depth_dp = look_down_depth.unsqueeze(-1).to(self.dtype)
            depths_dp = torch.stack([self.pixel_goal_depth, depth_dp]).unsqueeze(0).to(self.device)
        else:
            depths_dp = None

        with torch.no_grad():
            dp_actions = self.step_s1(self.traj_latents, images_dp, depths_dp,
                                      num_inference_steps=config.get('infer_step'),
                                      num_sample_trajs=config.get('traj_num'))

        action_list, _ = traj_to_actions(dp_actions)
        if len(action_list) < MAX_STEPS:
            action_list += [0] * (MAX_STEPS - len(action_list))

        local_actions = action_list
        if len(local_actions) >= MAX_LOCAL_STEPS:
            local_actions = local_actions[:MAX_LOCAL_STEPS]
        
        log.info(f"[TIME] On-device S1 step time: {time.time() - start_time:.4f}s.")
        return {'local_actions': local_actions}

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
        self.pixel_goal_rgb = None
        self.pixel_goal_depth = None
        self.traj_latents = None
