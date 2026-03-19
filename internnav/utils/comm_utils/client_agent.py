import base64
import pickle
from typing import Any, Dict, List, Optional
from time import time
import cv2

import requests
from PIL import Image
import PIL
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms import v2

from internnav.configs.agent import AgentCfg, NewAgentCfg, InitRequest, ResetRequest, StepRequest
from internnav.agent.internvla_n1_s1_agent import System1
from internnav.utils.common_log_util import common_logger as log
from .client_utils import init_visual_encoder, image_preprocess
from .visual_encoder import VisionEncoder, numpy_compression_v2, draw_heatmap_on_image, numpy_compression


def serialize_obs(obs):
    serialized = pickle.dumps(obs)
    encoded = base64.b64encode(serialized).decode('utf-8')
    return encoded


def remove_from_obs(obs: dict) -> dict:
    keys = ['globalgps', 'globalrotation', 'topdown_rgb', 'topdown_depth', 'instruction_tokens', 'depth']
    for key in keys:
        obs.pop(key, None)
    return obs


class AgentClient:
    """
    Client class for Agent service.
    """

    def __init__(self, config: NewAgentCfg):
        self.base_url = f'http://{config.cloud_server_host}:{config.cloud_server_port}'
        self.s1_agent = System1(config)
        self.device = self.s1_agent.device
        # self.visual_encoder, self.visual_processor, self.img_feature_extractor = \
        #     init_visual_encoder(config.visual_encoder, self.device)
        self.vision_encoder = VisionEncoder()
        self.agent_name = self._initialize_cloud_agent(config)

        self.ema_bandwidth = None

        self.current_stage = 's2'
        self.forward_step_num = 0
        self.PLAN_STEP_GAP = 8
        self.compressed_ratios = np.arange(0.1, 1.0, 0.1)
        self.transmission_delay_threshold = 0.3  # Set a threshold for transmission delay (in seconds)

    def _initialize_cloud_agent(self, config: NewAgentCfg) -> str:
        request_data = InitRequest(agent_config=config).model_dump(mode='json')

        response = requests.post(
            url=f'{self.base_url}/agent/init',
            json=request_data,
            headers={'Content-Type': 'application/json'},
        )
        response.raise_for_status()

        return response.json()['agent_name']

    def _preprocess_obs(self, obs: list[dict]):
        serialized_obs = serialize_obs(obs)
        upload_data_size = len(serialized_obs) # in bytes

        rgb = obs[0]['rgb']
        input_dict = {'rgb': rgb}
        fix_upload_size = upload_data_size - len(serialize_obs(input_dict)) # in bytes
        rgb = Image.fromarray(rgb).convert('RGB')
        width, height = rgb.size

        optimal_ratio = self.solve_optimal_resolution(rgb, fix_upload_size)
        rgb = rgb.resize((int(width * optimal_ratio), int(height * optimal_ratio)))
        return np.array(rgb)

    def step(self, obs: List[Dict[str, Any]]) -> List[List[int]]:
        obs[0] = remove_from_obs(obs[0])
        obs[0]['stage'] = self.current_stage  # Add current stage information to the observation
        orgin_rgb = obs[0]['rgb']

        serialized_obs = serialize_obs(obs)
        upload_data_size = len(serialized_obs)  # in bytes
        log.info(f"Original observation size: {upload_data_size / 1024:.2f} KB")

        # estimated_transmission_delay = self.estimate_transmission_time(upload_data_size)
        # if estimated_transmission_delay is not None and estimated_transmission_delay > self.transmission_delay_threshold:
        # log.info(f"[TIME] Estimated transmission time: {estimated_transmission_delay:.4f}s")
        preprocess_start_time = time()
        # compressed_rgb = self.compress_rgb(orgin_rgb)
        obs[0]['rgb'] = numpy_compression_v2(orgin_rgb)
        obs[0]['compressed'] = 1  # Indicate that the RGB has been compressed
        serialized_obs = serialize_obs(obs)
        compressed_size = len(serialized_obs)  # in bytes
        log.info(f"Compressed observation size: {compressed_size / 1024:.2f} KB")
        log.info(f"Transmission size reduction ratio: {(upload_data_size - compressed_size) / upload_data_size * 100:.4f}%")
        preprocess_end_time = time()
        log.info(f"[TIME] Image compression time: {preprocess_end_time - preprocess_start_time:.4f}s")

        transmission_start_time = time()
        request_data = StepRequest(observation=serialized_obs).model_dump(mode='json')

        response = requests.post(
            url=f'{self.base_url}/agent/{self.agent_name}/step',
            json=request_data,
            headers={'Content-Type': 'application/json'},
        )
        response.raise_for_status()

        response_data = response.json()
        transmission_end_time = time()

        cloud_data: dict = response_data['action'][0]
        cloud_inference_latency = cloud_data['processing_time']
        cloud_response_latency = transmission_end_time - transmission_start_time
        transmission_latency = cloud_response_latency - cloud_inference_latency

        log.info(f"[TIME] Cloud inference time: {cloud_inference_latency:.4f}s")
        log.info(f"[TIME] Actual transmission time: {transmission_latency:.4f}s")

        self.update_bandwidth(compressed_size, transmission_latency)

        if self.current_stage == 's2':
            traj_latents = cloud_data.get('traj_latents', None)  # obtain traj_latents for System1
            if traj_latents is not None:
                obs[0]['rgb'] = orgin_rgb  # Use original RGB for System1 processing

                draw_heatmap_on_image(orgin_rgb, 
                                      self.vision_encoder.get_patch_importance(orgin_rgb))

                self.s1_agent.record_goal_obs(obs[0], traj_latents)
                s1_response_data = self.s1_agent.step(obs[0])

                self.forward_step_num += 1
                self.current_stage = 's1'
                return s1_response_data['action']
            else:
                response_data['action'][0].pop('traj_latents', None)  # Remove traj_latents if not present
                return response_data['action']
        else:
            obs[0]['rgb'] = orgin_rgb
            s1_response_data = self.s1_agent.step(obs[0])
            self.forward_step_num += 1

            # if len(self.s1_agent.action_list) == 0 and self.s1_agent.ready_to_reach_goal:
            if self.forward_step_num > self.PLAN_STEP_GAP and len(self.s1_agent.action_list) == 0:
                self.current_stage = 's2'
                self.forward_step_num = 0
                self.s1_agent.ready_to_reach_goal = False
            return s1_response_data['action']

    def reset(self, reset_index: Optional[List] = None) -> None:
        response = requests.post(
            url=f'{self.base_url}/agent/{self.agent_name}/reset',
            json=ResetRequest(reset_index=reset_index).model_dump(mode='json'),
            headers={'Content-Type': 'application/json'},
        )
        response.raise_for_status()

        self.s1_agent.reset(reset_index)

        self.current_stage = 's2'  # Reset to initial stage after reset
        self.forward_step_num = 0

    def compress_rgb(self, rgb: np.ndarray):
        patch_importance = self.vision_encoder.get_patch_importance(rgb)
        compressed_img = numpy_compression(rgb, patch_importance)
        return compressed_img
    
    def estimate_transmission_time(self, upload_size_bytes):
        if self.ema_bandwidth is None:
            return  # Bandwidth not yet estimated, cannot provide a reliable estimate
        
        transmission_delay = (upload_size_bytes * 8) / self.ema_bandwidth
        return transmission_delay

    def update_bandwidth(self, data_size, transmission_latency, alpha=0.2):
        '''
        Update the bandwidth estimation.

        Args:
            data_size: The size of the data transmitted (in bytes).
            upload_latency: The time taken for the transmission (in seconds).
            alpha: The smoothing factor for EMA.
        '''
        bandwidth = (data_size * 8) / transmission_latency # bps
        if self.ema_bandwidth is None:
            self.ema_bandwidth = bandwidth
        else:
            self.ema_bandwidth = alpha * bandwidth + (1 - alpha) * self.ema_bandwidth

        log.info(f"[BANDWIDTH] Current bandwidth estimation: {self.ema_bandwidth / 1024:.4f} Kbps")
    
    def solve_optimal_resolution(self, rgb: PIL.Image, fix_upload_size: int):
        if self.ema_bandwidth is None:
            return 1.0  # If bandwidth is not yet estimated, use original resolution

        width, height = rgb.size

        images = [rgb]
        upload_sizes = []
        for ratio in self.ratios:
            resized_image = rgb.resize((int(width * ratio), int(height * ratio)))
            resize_len = len(serialize_obs({'rgb': resized_image})) + fix_upload_size

            upload_sizes.append(resize_len)
            images.append(resized_image)
        
        upload_latencies = [self.estimate_upload_time(size * 8) for size in upload_sizes]

        token_loss = self.token_consistency_loss(images)
        perceptual_loss = self.perceptual_loss(images)
        total_loss = token_loss + perceptual_loss

        # 最小化loss，并满足上传延迟在阈值内
        optimal_ratio = 0.1
        # min_loss = float('inf')
        for ratio, latency, loss in zip(self.ratios, upload_latencies, total_loss):
            if latency <= self.transmission_delay_threshold:
                if loss <= total_loss[self.ratios == optimal_ratio][0]:
                    optimal_ratio = ratio

        return optimal_ratio
        
    def token_consistency_loss(self, images: list[PIL.Image]):
        inputs = self.visual_processor(images=images, return_tensors="pt", padding=True)
        pixel_values = inputs['pixel_values'].to(self.device)
        with torch.no_grad():
            tokens = self.visual_encoder(pixel_values).last_hidden_state

        # remove CLS token
        tokens = tokens[:, 1:, :]
        # Global average pooling
        average_global_tokens = tokens.mean(dim=1)
        average_global_tokens = F.normalize(average_global_tokens, dim=-1)

        origin_token = average_global_tokens[:1]  # feature of the original image
        comp_tokens = average_global_tokens[1:]  # features of the compressed images

        mse_losses = F.mse_loss(comp_tokens, origin_token, reduction='none')
        mse_losses = mse_losses.mean(dim=1)
        return mse_losses

    def perceptual_loss(self, images: list[PIL.Image]):
        list_of_tensors = [v2.functional.to_image(img.resize((224, 224))) for img in images]
        batch_tensor = torch.stack(list_of_tensors).to(self.device)
        preprocessed_batch = image_preprocess(batch_tensor)
        
        extracted_features = self.img_feature_extractor(preprocessed_batch)
        extracted_features = extracted_features.view(extracted_features.size(0), -1)  # Flatten the features
        extracted_features = F.normalize(extracted_features, dim=-1)

        origin_feat = extracted_features[:1]  # Feature of the original image
        comp_feat = extracted_features[1:]  # Features of the compressed images

        mse_losses = F.mse_loss(comp_feat, origin_feat, reduction='none')
        mse_losses = mse_losses.mean(dim=1)
        return mse_losses