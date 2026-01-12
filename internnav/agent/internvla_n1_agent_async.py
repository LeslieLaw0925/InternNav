import copy
import itertools
import os
import re
import time
from datetime import datetime
import sys
sys.path.append('.')
sys.path.append('./src/diffusion-policy')

import cv2
import imageio
import numpy as np
import torch

from collections import OrderedDict

from PIL import Image
from transformers import AutoProcessor, AutoTokenizer
import ray
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.utils.torch_utils import randn_tensor
import torch.nn as nn

from internnav.agent.base import Agent
from internnav.configs.agent import AgentCfg
from internnav.configs.model.base_encoders import ModelCfg
from internnav.model.basemodel.internvla_n1.internvla_n1 import InternVLAN1ForCausalLM
from internnav.model.utils.vln_utils import split_and_clean, traj_to_actions
from internnav.model.utils.misc import set_random_seed
from internnav.model.basemodel.internvla_n1.internvla_n1_arch import build_traj_dit, \
    build_depthanythingv2, LatentEmbSize, SinusoidalPositionalEncoding, MemoryEncoder, \
    QFormer


DEFAULT_IMAGE_TOKEN = "<image>"
_RESNET_MEAN = [0.485, 0.456, 0.406]
_RESNET_STD = [0.229, 0.224, 0.225]
TROCH_DTYPE = torch.bfloat16

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
        self.store[key] = data

    def get(self, key):
        return self.store.get(key, None)
    
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
        self.PLAN_STEP_GAP = getattr(_model_settings, 'plan_step_gap', 8)

        self.action_seq: list = []
        self.last_action: int = -1
        self.look_down: bool = False
        self.episode_idx: int = 0
        self.forward_step_num: int = 0
        self.output_pixel = None
        self.which_is_inferring: str = "s2"

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

    def reset(self, reset_index=None):
        self.s1_agent.reset.remote(reset_index)
        self.s2_agent.reset.remote(reset_index)
        self.shm.clear.remote()

        '''reset_index: [0]'''
        if reset_index is not None:
            self.episode_idx += 1
            if self.vis_debug:
                self.fps_writer.close()
                self.fps_writer2.close()
        else:
            self.episode_idx = -1

        self.action_seq = []
        self.last_action = -1
        self.look_down = False
        self.forward_step_num = 0
        self.output_pixel = None
        self.which_is_inferring = "s2"

        if self.vis_debug:
            self.fps_writer = imageio.get_writer(f"{self.debug_path}/fps_{self.episode_idx}_async.mp4", fps=5)
            self.fps_writer2 = imageio.get_writer(f"{self.debug_path}/fps_{self.episode_idx}_async_dp.mp4", fps=5)

    def s2_step(self, rgb, depth, pose, instruction, look_down=False):
        action_seq_ref = self.s2_agent.step.remote(rgb, depth, pose, instruction, \
                                        look_down)
        action_seq = ray.get(action_seq_ref)
        self.forward_step_num = 0
        self.which_is_inferring = "s2"
        return action_seq
    
    def s1_step(self, rgb, depth=None):
        action_seq_ref = self.s1_agent.step.remote(rgb, depth)
        action_seq = ray.get(action_seq_ref)
        self.which_is_inferring = "s1"
        self.forward_step_num += len(action_seq)
        return action_seq

    def step(self, obs):
        obs = obs[0]  # do not support batch_env currently?
        rgb = obs['rgb']
        depth = obs['depth']
        instruction = obs['instruction']
        pose = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        
        if self.last_action == 5:
            # 此时代表S2找到了pixel goal，获取pixel goal的rgb，depth，以及traj_latent
            self.s2_step(rgb, depth, pose, instruction, look_down=True)
            self.output_pixel = ray.get(self.shm.get.remote(SharedObjKeys.PIXEL_GOAL))
            # 启动S1进行轨迹输出
            self.action_seq = self.s1_step(rgb, depth)

        if self.action_seq == []:
            if self.which_is_inferring == "s1":
                if self.forward_step_num > self.PLAN_STEP_GAP:
                    self.action_seq = self.s2_step(rgb, depth, pose, instruction, look_down=False)
                else:
                    self.action_seq = self.s1_step(rgb, depth)
            else:
                self.action_seq = self.s2_step(rgb, depth, pose, instruction, look_down=False)
        else:
            self.s2_agent.step_no_infer.remote(rgb, depth, pose)
        
        self.last_action = self.action_seq.pop(0)
        output = {'action': [self.last_action]}

        # Visualization
        if self.vis_debug:
            vis = rgb.copy()
            if 'action' in output:
                vis = cv2.putText(vis, str(output['action'][0]), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            if self.output_pixel is not None:
                pixel = self.output_pixel
                vis = cv2.putText(
                    vis,
                    f"{pixel[1]}, {pixel[0]}",
                    (50, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                )

                cv2.circle(vis, (pixel[1], pixel[0]), 5, (0, 255, 0), -1)
                self.output_pixel = None

            self.fps_writer.append_data(vis)

        return [{'action': output['action'], 'ideal_flag': True}]


@ray.remote(num_gpus=0.2)
class S1Agent:
    def __init__(self, shm, model_settings: ModelCfg):
        set_random_seed(0)
        self.shm = shm
        self.device = torch.device(model_settings.device)
        model_path = model_settings.model_path.lower()

        self.latent_queries = nn.Parameter(torch.randn(1, 4, 3584))

        if 'navdp' in model_path:
            # Use navdp as s1 model
            self.type = 'navdp'
            self.build_navdp(model_settings.navdp_pretrained)
        elif 'dualvln' in model_path:
            # Use nextdit as s1 model
            self.type = 'nextdit'
            self.build_nextdit()
        else:
            assert False, "Unknown S1 model type."

        self.depth_threshold = 5.0

    def build_navdp(self, navdp_pretrained=None):
        from internnav.model.basemodel.internvla_n1.navdp import NavDP_Policy_DPT_CriticSum_DAT
        navdp = NavDP_Policy_DPT_CriticSum_DAT(navdp_pretrained=navdp_pretrained,
                                            memory_size=2, 
                                            navdp_version=0.1)
        navdp.load_model()    
        self.model = navdp
        self.model.to(dtype=torch.bfloat16, device=self.device)
        self.model.eval()
    
    def build_nextdit(self):
        self.traj_dit, self.noise_scheduler = build_traj_dit(None)
        self.traj_dit = self.traj_dit.to(self.device, dtype=TROCH_DTYPE)
        self.action_encoder = nn.Linear(3, 384, bias=True).to(self.device, dtype=TROCH_DTYPE)
        self.pos_encoding = SinusoidalPositionalEncoding(384).to(self.device, dtype=TROCH_DTYPE)
        self.action_decoder = nn.Linear(384, 3, bias=True).to(self.device, dtype=TROCH_DTYPE)
        self.cond_projector = nn.Sequential(
            nn.Linear(3584, LatentEmbSize), nn.GELU(approximate="tanh"), nn.Linear(LatentEmbSize, LatentEmbSize)
        ).to(self.device, dtype=TROCH_DTYPE)

        self.rgb_model = build_depthanythingv2(None).to(self.device, dtype=TROCH_DTYPE)
        self.memory_encoder = MemoryEncoder().to(self.device, dtype=TROCH_DTYPE)
        self.rgb_resampler = QFormer().to(self.device, dtype=TROCH_DTYPE)

        for name, value in (("_resnet_mean", _RESNET_MEAN), ("_resnet_std", _RESNET_STD)):
            self.rgb_model.register_buffer(name, 
                                           torch.BFloat16Tensor(value).view(1, 1, 3, 1, 1).to(self.device), 
                                           persistent=False)

    def step(self, rgb: np.ndarray, depth: np.ndarray = None):
        traj_latents = ray.get(self.shm.get.remote(SharedObjKeys.TRAJ_LATENT))
        pixel_goal_rgb = ray.get(self.shm.get.remote(SharedObjKeys.PIXEL_GOAL_RGB))
        if not all([pixel_goal_rgb is not None, traj_latents is not None]):
            return None
        
        processed_pixel_rgb = (np.array(Image.fromarray(pixel_goal_rgb).resize((224, 224))) / 255.0)
        processed_rgb = np.array(Image.fromarray(rgb).resize((224, 224))) / 255.0
        rgbs = (
            torch.stack([torch.from_numpy(processed_pixel_rgb), torch.from_numpy(processed_rgb)])
            .unsqueeze(0)
            .to(self.device)
        )  # [1, 2, 224, 224, 3]

        if depth is not None:
            pixel_goal_depth = ray.get(self.shm.get.remote(SharedObjKeys.PIXEL_GOAL_DEPTH))
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

        if self.type == 'nextdit':
            dp_actions = self.step_s1_nextdit(traj_latents.to(self.device), 
                                              images_dp=rgbs, 
                                              use_async=True)
        elif self.type == 'navdp':
            dp_actions = self.step_s1_navdp(traj_latents.to(self.device), 
                                            images_dp=rgbs, 
                                            depths_dp=depths, 
                                            use_async=True)
        else:
            assert False, "Unknown S1 model type."
        
        action_list = traj_to_actions(dp_actions)
        action_list = [x for x in action_list if x != 0]

        if action_list == []:
            return [-1]
        else:
            return action_list[:4]

    def step_s1_nextdit(self, traj_latents, 
                        images_dp=None, 
                        use_async=False,
                        predict_step_nums=32,
                        guidance_scale: float = 1.0,
                        num_inference_steps: int = 10,
                        num_sample_trajs: int = 32,
                        ):
        scheduler = FlowMatchEulerDiscreteScheduler()
        device = traj_latents.device
        dtype = traj_latents.dtype

        traj_latents = self.cond_projector(traj_latents)
        if use_async:
            with torch.no_grad():
                images_dp = images_dp.permute(0, 1, 4, 2, 3)
                images_dp_norm = (images_dp - self.rgb_model._resnet_mean) / self.rgb_model._resnet_std
                images_dp_feat = (
                    self.rgb_model.get_intermediate_layers(images_dp_norm.flatten(0, 1).to(dtype))[0]
                    .unflatten(dim=0, sizes=(1, -1))
                )
                memory_feat = self.memory_encoder(
                    images_dp_feat.flatten(1, 2)
                )  # [bs*select_size,512,384]
                memory_feat = torch.cat([images_dp_feat.flatten(1, 2), memory_feat], dim=-1)
                memory_tokens = self.rgb_resampler(memory_feat)
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
            latent_features = self.action_encoder(latents)
            pos_ids = (
                torch.arange(latent_features.shape[1])
                .reshape(1, -1)
                .repeat(batch_size, 1)
                .to(latent_features.device)
            )
            pos_embed = self.pos_encoding(pos_ids)
            latent_features += pos_embed  # [num_sample_trajs, t, 384]
            latent_model_input = latent_features.repeat(2, 1, 1)
            if hasattr(scheduler, "scale_model_input"):
                latent_model_input = scheduler.scale_model_input(latent_model_input, t)

            # predict noise model_output
            noise_pred = self.traj_dit(
                x=latent_model_input,
                timestep=t.unsqueeze(0)
                .expand(latent_model_input.shape[0])
                .to(latent_model_input.device, torch.long),
                z_latents=hidden_states_input,
            )

            noise_pred = self.action_decoder(noise_pred)

            # perform guidance
            noise_pred_uncond, noise_pred = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred - noise_pred_uncond)

            # compute previous: x_t -> x_t-1
            latents = scheduler.step(noise_pred, t, latents).prev_sample
        return latents.detach()

    def step_s1_navdp(self, traj_latents, images_dp=None, depths_dp=None, use_async=False):
        if use_async:
            all_trajs = self.model.predict_pointgoal_action_async(
                traj_latents, images_dp, depths_dp
            )
        else:
            all_trajs = self.model.predict_pointgoal_action(traj_latents)
        return all_trajs
    
    def reset(self, reset_index=None):
        pass


@ray.remote(num_gpus=0.8)
class S2Agent:
    def __init__(self, shm, model_settings: ModelCfg):
        set_random_seed(0)
        self.shm = shm
        self.device = torch.device(model_settings.device)
        print(f"args.model_path: {model_settings.model_path}")
        self.model = InternVLAN1ForCausalLM.from_pretrained(
            model_settings.model_path,
            torch_dtype=TROCH_DTYPE,
            attn_implementation="flash_attention_2",
            device_map={"": self.device},
        )
        self.model.eval()
        self.model.to(self.device)

        self.tokenizer = AutoTokenizer.from_pretrained(model_settings.model_path, use_fast=True)
        self.processor = AutoProcessor.from_pretrained(model_settings.model_path)
        self.processor.tokenizer = self.tokenizer
        self.processor.tokenizer.padding_side = 'left'

        self.resize_w = model_settings.resize_w
        self.resize_h = model_settings.resize_h
        self.num_history = model_settings.num_history

        self.init_prompts()

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
       
        self.rgb_list = []
        self.depth_list = []
        self.pose_list = []
        self.episode_idx = 0  # S2's episode idx is different from the system's idx
        self.conversation_history = []  # Multi-turn conversation exists when looking down
        self.llm_output = ""

    def init_prompts(self):
        self.DEFAULT_IMAGE_TOKEN = "<image>"
        # For absolute pixel goal
        prompt = "You are an autonomous navigation assistant. Your task is to <instruction>. Where should you go next to stay on track? Please output the next waypoint\'s coordinates in the image. Please output STOP when you have successfully completed the task."
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

    def reset(self, reset_index=None):
        self.rgb_list = []
        self.depth_list = []
        self.pose_list = []
        self.episode_idx = 0
        self.conversation_history = []
        self.llm_output = ""

        self.output_action = None
        self.output_latent = None
        self.output_pixel = None

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
        self.episode_idx += 1

    def step(self, rgb, depth, pose, instruction, look_down=False):
        self.output_action, self.output_latent, self.output_pixel = self.step_s2(
            rgb, depth, pose, instruction, look_down
        )

        if self.output_latent is not None:
            self.output_latent = self.output_latent.detach().cpu()
            self.shm.put.remote(SharedObjKeys.TRAJ_LATENT, self.output_latent)
            self.shm.put.remote(SharedObjKeys.PIXEL_GOAL, copy.deepcopy(self.output_pixel))
            self.shm.put.remote(SharedObjKeys.PIXEL_GOAL_RGB, copy.deepcopy(rgb))
            self.shm.put.remote(SharedObjKeys.PIXEL_GOAL_DEPTH, copy.deepcopy(depth))
            print(f"S2 put latent traj to shm.")

        return self.output_action    

    def step_s2(self, rgb, depth, pose, instruction, look_down=False):
        # Need to be careful: look_down images are not added to rgb_list and won't be selected as history
        # 1. Preprocess input
        image = Image.fromarray(rgb).convert('RGB')
        if not look_down:  # Don't add look_down images to rgb_list
            image = image.resize((self.resize_w, self.resize_h))
            self.rgb_list.append(image)

            # 2. Prepare input for the model
            # Clear conversation history when not looking down, provide normal image history and instruction
            self.conversation_history = []
            # 2.1 instruction
            sources = copy.deepcopy(self.conversation)
            sources[0]["value"] = sources[0]["value"].replace('<instruction>.', instruction)
            # 2.2 images
            cur_images = self.rgb_list[-1:]
            if self.episode_idx == 0:
                history_id = []
            else:
                history_id = np.unique(np.linspace(0, self.episode_idx - 1, self.num_history, dtype=np.int32)).tolist()
                placeholder = (self.DEFAULT_IMAGE_TOKEN + '\n') * len(history_id)
                sources[0]["value"] += f' These are your historical observations: {placeholder}.'

            history_id = sorted(history_id)
            self.input_images = [self.rgb_list[i] for i in history_id] + cur_images
            input_img_id = 0
            self.episode_idx += 1  # Only increment when not looking down to maintain correspondence with rgb_list idx
        else:
            # Continue conversation based on previous when looking down
            self.input_images.append(image)  # This image should be the look_down image
            input_img_id = -1
            assert self.llm_output != "", "Last llm_output should not be empty when look down"
            sources = [{"from": "human", "value": ""}, {"from": "gpt", "value": ""}]
            self.conversation_history.append(
                {'role': 'assistant', 'content': [{'type': 'text', 'text': self.llm_output}]}
            )

        prompt = self.conjunctions[0] + self.DEFAULT_IMAGE_TOKEN
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

        # 3. Model inference
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=False,
                use_cache=True,
                past_key_values=None,
                return_dict_in_generate=True,
            ).sequences
        self.llm_output = self.processor.tokenizer.decode(
            output_ids[0][inputs.input_ids.shape[1] :], skip_special_tokens=True
        )
        print(f"============ output {self.episode_idx}  {self.llm_output}")

        # 4. Post-process results
        if bool(re.search(r'\d', self.llm_output)):  # Output pixel goal
            coord = [int(c) for c in re.findall(r'\d+', self.llm_output)]
            pixel_goal = [int(coord[1]), int(coord[0])]
            output_pixel = np.array(pixel_goal)

            image_grid_thw = torch.cat([thw.unsqueeze(0) for thw in inputs.image_grid_thw], dim=0)
            with torch.no_grad():
                traj_latents = self.model.generate_latents(output_ids, inputs.pixel_values, image_grid_thw)
            return None, traj_latents, output_pixel

        else:  # Output action
            action_seq = self.parse_actions(self.llm_output)
            return action_seq, None, None
