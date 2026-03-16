import copy
import itertools
import os
import re
from collections import OrderedDict
import time

import cv2
import imageio
import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, AutoTokenizer

from internnav.agent.base import Agent
from internnav.configs.agent import AgentCfg
from internnav.configs.model.base_encoders import ModelCfg
from internnav.model.utils.vln_utils import split_and_clean
from internnav.model.utils.misc import set_random_seed


DEFAULT_IMAGE_TOKEN = "<image>"
TROCH_DTYPE = torch.bfloat16


def init_swir_model(device):
    from internnav.agent.swin_ir.network_swinir import SwinIR as net

    model = net(upscale=1, in_chans=3, img_size=128, window_size=8,
                img_range=1., depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6],
                mlp_ratio=2, upsampler='', resi_connection='1conv')
    param_key_g = 'params'
    
    model_path = 'checkpoints/005_colorDN_DFWB_s128w8_SwinIR-M_noise50.pth'
    pretrained_model = torch.load(model_path)
    model.load_state_dict(pretrained_model[param_key_g] if param_key_g in pretrained_model.keys() else pretrained_model, strict=True)
    model.eval()
    model = model.to(device)
    return model


def init_realgan_model():
    from realesrgan import RealESRGANer
    from basicsr.archs.rrdbnet_arch import RRDBNet

    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4)
    netscale = 4

    # restorer
    upsampler = RealESRGANer(
        scale=netscale,
        model_path='checkpoints/RealESRGAN_x4plus_anime_6B.pth',
        model=model,
        tile=400,
        tile_pad=10,
        pre_pad=0,
        half=True,
        )
    
    return upsampler

@Agent.register('internvla_n1_cloud')
class CloudAgent(Agent):
    def __init__(self, config: AgentCfg):
        super().__init__(config)
        vln_sensor_config = self.config.model_settings
        _model_settings = ModelCfg(**vln_sensor_config)
        self.s2_agent = System2(_model_settings)
        self.device = torch.device(_model_settings.device)
        # self.img_enchanced_model = init_realgan_model()

        self.action_seq: list = []
        self.last_action: int = -1
        self.look_down: bool = False
        self.episode_idx: int = 0
        self.output_pixel = None
        self.traj_latents = None

        # vis debug
        self.vis_debug = vln_sensor_config['vis_debug']
        if self.vis_debug:
            self.debug_path = vln_sensor_config['vis_debug_path']
            os.makedirs(self.debug_path, exist_ok=True)
            self.fps_writer = imageio.get_writer(f"{self.debug_path}/fps_{self.episode_idx}_async.mp4", fps=5)
            self.fps_writer2 = imageio.get_writer(f"{self.debug_path}/fps_{self.episode_idx}_async_dp.mp4", fps=5)

    def reset(self, reset_index=None):
        self.action_seq = []
        self.last_action = -1
        self.look_down = False
        self.output_pixel = None
        self.s2_agent.reset(reset_index)
        
        '''reset_index: [0]'''
        if reset_index is not None:
            self.episode_idx += 1
            if self.vis_debug:
                self.fps_writer.close()
                self.fps_writer2.close()
        else:
            self.episode_idx = -1

        if self.vis_debug:
            self.fps_writer = imageio.get_writer(f"{self.debug_path}/fps_{self.episode_idx}_async.mp4", fps=5)
            self.fps_writer2 = imageio.get_writer(f"{self.debug_path}/fps_{self.episode_idx}_async_dp.mp4", fps=5)

    def step(self, obs):
        infer_start_time = time.time()

        obs = obs[0]  # do not support batch_env currently?
        rgb = cv2.imdecode(obs['rgb'], cv2.IMREAD_COLOR)
        # if obs['compressed'] == 1:
        #     start_time = time.time()
        #     rgb = self.img_restore(rgb)
        #     rgb = self.img_enhance(rgb)
        #     rgb = cv2.detailEnhance(rgb, sigma_s=10, sigma_r=0.15)
        #     print(f"Image process time: {time.time() - start_time:.2f}s")
        
        depth = obs.get('depth', None)
        instruction = obs['instruction']
        current_stage = obs['stage']
        pose = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        traj_latents = None

        if current_stage == "s1":
            self.s2_agent.step_no_infer(rgb, depth, pose)
            output = {'action': [-1]}
        else:
            if self.last_action == 5:
                # 此时S2找到了pixel goal，获取pixel goal的rgb，depth，以及traj_latent
                _, traj_latents, self.output_pixel = \
                    self.s2_agent.step(rgb, depth, pose, instruction, look_down=True)
                traj_latents = traj_latents.detach().cpu().to(dtype=torch.float32).numpy().tolist()
                self.action_seq = []
                self.last_action = -1
            else:
                if self.action_seq == []:
                    self.action_seq, _, _ = \
                        self.s2_agent.step(rgb, depth, pose, instruction, look_down=False)
                else:
                    self.s2_agent.step_no_infer(rgb, depth, pose)
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

        return [{'action': output['action'],
                 'ideal_flag': True, 
                 'traj_latents': traj_latents,
                 'infer_start_time': infer_start_time,
                 'infer_end_time': time.time()}]

    def img_restore(self, img_lq: np.ndarray, window_size: int = 8):
        img_lq = img_lq.astype(np.float32) / 255.
        img_lq = np.transpose(img_lq if img_lq.shape[2] == 1 else img_lq[:, :, [2, 1, 0]], (2, 0, 1))  # HCW-BGR to CHW-RGB
        img_lq = torch.from_numpy(img_lq).float().unsqueeze(0).to(self.device)  # CHW-RGB to NCHW-RGB

        # inference
        with torch.no_grad():
            # pad input image to be a multiple of window_size
            _, _, h_old, w_old = img_lq.size()
            h_pad = (h_old // window_size + 1) * window_size - h_old
            w_pad = (w_old // window_size + 1) * window_size - w_old
            img_lq = torch.cat([img_lq, torch.flip(img_lq, [2])], 2)[:, :, :h_old + h_pad, :]
            img_lq = torch.cat([img_lq, torch.flip(img_lq, [3])], 3)[:, :, :, :w_old + w_pad]
            output = self.swin_ir_model(img_lq)
            output = output[..., :h_old, :w_old]

        # save image
        output = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
        if output.ndim == 3:
            output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))  # CHW-RGB to HCW-BGR
        output = (output * 255.0).round().astype(np.uint8)  # float32 to uint8
        
        return output

    def img_enhance(self, img: np.ndarray):
        try:
            output, _ = self.img_enchanced_model.enhance(img, outscale=1)
            return output
        except Exception as e:
            print(f"Img enhance failed: {e}")
            return img
        
class System2:
    def __init__(self, model_settings: ModelCfg):
        from internnav.model.basemodel.internvla_n1.internvla_n1 import InternVLAN1ForCausalLM

        set_random_seed(0)
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