import numpy as np
import torch
from PIL import Image
import time

from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.utils.torch_utils import randn_tensor

from internnav.configs.agent import AgentCfg
from internnav.configs.model.base_encoders import ModelCfg
from internnav.model.utils.vln_utils import traj_to_actions
from internnav.model.utils.misc import set_random_seed
from internnav.model.basemodel.internvla_n1.internvla_n1_arch import AsyncInternVLAN1MetaModel
from internnav.utils.common_log_util import common_logger as log


class System1:
    def __init__(self, config: AgentCfg):
        set_random_seed(0)
        vln_sensor_config = config.model_settings
        model_settings = ModelCfg(**vln_sensor_config)

        self.device = torch.device(model_settings.device)
        self.dtype = torch.bfloat16
        self.config = dict(system1=model_settings.s1_type,
                           navdp_pretrained=model_settings.navdp_pretrained,
                           nextdit_pretrained=model_settings.nextdit_pretrained
                           )
        self.model = AsyncInternVLAN1MetaModel(self.config)

        if 'nextdit' in self.config['system1']:
            self.model.load_state_dict(
                torch.load(model_settings.nextdit_pretrained, map_location="cpu"))
            self.model.to(self.device, self.dtype)
        elif 'navdp' in self.config['system1']:
            self.model.navdp.to(self.device, self.dtype)
        else:
            raise NotImplementedError
        
        self.depth_threshold = 5.0
        self.action_list = None

    def step(self, obs: dict):
        if self.action_list is not None and len(self.action_list) > 0:
            action = self.action_list.pop(0)
            return [{'action': [action], 'ideal_flag': True}]

        rgb = obs.get('rgb', None)
        depth = obs.get('depth', None)
        pixel_goal_rgb = obs.get('pixel_goal_rgb', None)
        pixel_goal_depth = obs.get('pixel_goal_depth', None)
        traj_latents = obs.get('traj_latents', None)

        if not all([rgb is not None, 
                    pixel_goal_rgb is not None, 
                    traj_latents is not None]):
            raise ValueError("Missing required observation for System1 step.")
        
        processed_pixel_rgb = (np.array(Image.fromarray(pixel_goal_rgb).resize((224, 224))) / 255.0)
        processed_rgb = np.array(Image.fromarray(rgb).resize((224, 224))) / 255.0
        rgbs = (
            torch.stack([torch.from_numpy(processed_pixel_rgb), torch.from_numpy(processed_rgb)])
            .unsqueeze(0)
            .to(self.device)
        )  # [1, 2, 224, 224, 3]

        if depth is not None:
            processed_pixel_depth = (np.array(Image.fromarray(pixel_goal_depth[:, :, 0]).resize((224, 224))) * 10.0)
            processed_pixel_depth[processed_pixel_depth > self.depth_threshold] = self.depth_threshold

            processed_depth = (np.array(Image.fromarray(depth[:, :, 0]).resize((224, 224))) * 10.0)  # should be 0-10m
            processed_depth[processed_depth > self.depth_threshold] = self.depth_threshold

            depths = (
                torch.stack([torch.from_numpy(processed_pixel_depth), torch.from_numpy(processed_depth)])
                .unsqueeze(0)
                .unsqueeze(-1)
                .to(self.device)
            )  # [1, 2, 224, 224, 1]
        else:
            depths = None

        traj_latents = torch.from_numpy(np.array(traj_latents)).\
            to(self.device, self.dtype)
        start_time = time.time()
        with torch.no_grad():
            dp_actions = self.step_s1(traj_latents, rgbs, depths_dp=depths)
        log.info(f'[TIME] On-device system1 step time: {time.time() - start_time:.2f} s')
            
        action_list = traj_to_actions(dp_actions)
        action_list = [x for x in action_list if x != 0]

        if action_list == []:
            action_list = [-1]
        else:
            action_list = action_list[:4]

        self.action_list = action_list
        return [{'action': [self.action_list.pop(0)], 'ideal_flag': True}]

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

        elif 'navdp' in self.config['system1']:
            if 'async' in self.config['system1']:
                all_trajs = self.model.navdp.predict_pointgoal_action_async(
                    traj_latents, images_dp, depths_dp
                )
            else:
                all_trajs = self.model.navdp.predict_pointgoal_action(traj_latents)
            return all_trajs
        
    def reset(self, reset_index=None):
        self.action_list = None