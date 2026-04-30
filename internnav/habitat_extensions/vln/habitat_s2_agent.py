import copy
import itertools
import os
import re
from collections import OrderedDict
import random
import argparse

import numpy as np
import torch
from transformers import AutoProcessor
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from internnav.agent.base import Agent
from internnav.configs.agent import AgentCfg
from internnav.model.utils.vln_utils import split_and_clean
from internnav.model.utils.misc import set_random_seed
from internnav.model.basemodel.internvla_n1.internvla_n1 import InternVLAN1ForCausalLM


DEFAULT_IMAGE_TOKEN = "<image>"
TROCH_DTYPE = torch.bfloat16


@Agent.register('habitat_s2_agent')
class System2(Agent):
    def __init__(self, config: AgentCfg):
        super().__init__(config)
        set_random_seed(0)

        self.output_path = "logs"

        self.model_args = argparse.Namespace(**config.model_settings)
        self.vis_debug = bool(getattr(self.model_args, "vis_debug", False))
        self.vis_debug_path = getattr(self.model_args, "vis_debug_path", os.path.join(self.output_path, "vis_debug"))

        processor = AutoProcessor.from_pretrained(self.model_args.model_path)
        processor.tokenizer.padding_side = 'left'

        device = torch.device(f"cuda:0")
        if self.model_args.mode == 'dual_system':
            model = InternVLAN1ForCausalLM.from_pretrained(
                self.model_args.model_path,
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
                device_map={"": device},
            )
        elif self.model_args.mode == 'system2':
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model_args.model_path,
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
                device_map={"": device},
            )
        else:
            raise ValueError(f"Invalid mode: {self.model_args.mode}")

        model.eval()
        self.device = device

        self.model = model
        self.processor = processor
        self.num_history = self.model_args.num_history

        self.init_prompts()  

        self.messages = []
        self.rgb_list = []
        self.input_images = []
        self.llm_outputs = ""

    def init_prompts(self):
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
        self.messages = []
        self.input_images = []
        self.rgb_list = []

        self.depth_list = []
        self.pose_list = []
        self.llm_output = ""

    def parse_actions(self, output):
        action_patterns = '|'.join(re.escape(action) for action in self.actions2idx)
        regex = re.compile(action_patterns)
        matches = regex.findall(output)
        actions = [self.actions2idx[match] for match in matches]
        actions = itertools.chain.from_iterable(actions)
        return list(actions)

    def step_no_infer(self, rgb):
        image = rgb.resize((self.model_args.resize_w, self.model_args.resize_h))
        self.rgb_list.append(image)

    def step(self, obs: dict):
        instruction = obs.get('instruction')
        if instruction is None:
            self.step_no_infer(obs.get('rgb'))
            return {}
        
        look_down_image = obs.get('rgb')
        look_down = obs.get('look_down', False)
        step_id = obs.get('step_id', 0)

        if look_down:
            # last action is look down
            sources = [{"from": "human", "value": ""}, {"from": "gpt", "value": ""}]
            self.input_images += [look_down_image]
            self.messages.append(
                {'role': 'assistant', 'content': [{'type': 'text', 'text': self.llm_outputs}]}  # noqa: F405
            )
            input_img_id = -1
        else:
            self.messages = []
            sources = copy.deepcopy(self.conversation)
            sources[0]["value"] = sources[0]["value"].replace(
                '<instruction>.', instruction[:-1]
            )
            cur_images = self.rgb_list[-1:]
            if step_id == 0:
                history_id = []
            else:
                history_id = np.unique(
                    np.linspace(0, step_id - 1, self.num_history, dtype=np.int32)
                ).tolist()
                placeholder = (DEFAULT_IMAGE_TOKEN + '\n') * len(history_id)
                sources[0]["value"] += f' These are your historical observations: {placeholder}.'

            history_id = sorted(history_id)
            # import pdb; pdb.set_trace()
            self.input_images = [self.rgb_list[i] for i in history_id] + cur_images
            input_img_id = 0

        prompt = random.choice(self.conjunctions) + DEFAULT_IMAGE_TOKEN
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

        self.messages.append({'role': 'user', 'content': content})

        text = self.processor.apply_chat_template(self.messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=self.input_images, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=False,
                use_cache=True,
                past_key_values=None,
                return_dict_in_generate=True,
            ).sequences

        self.llm_outputs = self.processor.tokenizer.decode(
            output_ids[0][inputs.input_ids.shape[1] :], skip_special_tokens=True
        )
        print('step_id:', step_id, 'output text:', self.llm_outputs)

        if bool(re.search(r'\d', self.llm_outputs)):  # output pixel goal
            coord = [int(c) for c in re.findall(r'\d+', self.llm_outputs)]
            pixel_goal = [int(coord[1]), int(coord[0])]

            pixel_values = inputs.pixel_values
            image_grid_thw = torch.cat([thw.unsqueeze(0) for thw in inputs.image_grid_thw], dim=0)

            with torch.no_grad():
                traj_latents = self.model.generate_latents(output_ids, pixel_values, image_grid_thw)
            traj_latents = traj_latents.detach().cpu().to(dtype=torch.float32).numpy().tolist()
            # import pdb; pdb.set_trace()
            return {'action_seq': [],
                    'traj_latents': traj_latents,
                    'pixel_goal': pixel_goal,}
        else:
            action_seq = self.parse_actions(self.llm_outputs)
            print('actions', action_seq, flush=True)        
            return {'action_seq': action_seq,
                    'traj_latents': None,
                    'pixel_goal': None,}