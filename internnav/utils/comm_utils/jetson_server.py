#!/usr/bin/env python
import base64
import multiprocessing
import pickle
from typing import Any, Dict, List
from time import time
import os

import uvicorn
from fastapi import APIRouter, FastAPI, HTTPException, status
import requests
import numpy as np

from internnav.agent.base import Agent
from internnav.configs.agent import InitRequest, ResetRequest, StepRequest
from internnav.configs.agent import NewAgentCfg
from internnav.agent.internvla_n1_s1_agent import System1
from internnav.utils.common_log_util import common_logger as log
from .visual_encoder import VisionEncoder, numpy_compression_v2, draw_heatmap_on_image
from .client_utils import serialize_obs, remove_from_obs


class AgentServer:
    """
    Server class for Agent service.
    """

    def __init__(self, host: str, port: int, config: NewAgentCfg):
        self.host = host
        self.port = port
        self.app = FastAPI(title='Jetson Service')
        self.agent_instances: Dict[str, Agent] = {}
        self._router = APIRouter(prefix='/agent')
        self._register_routes()
        self.app.include_router(self._router)

        self.base_url = f'http://{config.cloud_server_host}:{config.cloud_server_port}'
        self.s1_agent = System1(config)
        self.device = "cuda"
        self.vision_encoder = VisionEncoder()

        self.ema_bandwidth = None
        self.current_stage = 's2'
        self.forward_step_num = 0
        self.PLAN_STEP_GAP = 8
        self.compressed_ratios = np.arange(0.1, 1.0, 0.1)
        self.transmission_delay_threshold = 0.3  # Set a threshold for transmission delay (in seconds)
        self.if_compressed = False
        os.makedirs("logs/test_data", exist_ok=True)

    def _register_routes(self):
        route_config = [
            ('/init', self.init_agent, ['POST'], status.HTTP_201_CREATED),
            ('/{agent_name}/step', self.step_agent, ['POST'], None),
            ('/{agent_name}/reset', self.reset_agent, ['POST'], None),
            # TODO: Add stop server route
        ]

        for path, handler, methods, status_code in route_config:
            self._router.add_api_route(
                path=path,
                endpoint=handler,
                methods=methods,
                status_code=status_code,
            )

    async def init_agent(self, request: InitRequest):
        self.agent_name = request.agent_config.model_name

        response = requests.post(
            url=f'{self.base_url}/agent/init',
            json=request.model_dump(mode='json'),
            headers={'Content-Type': 'application/json'},
        )
        response.raise_for_status()
        return response.json()

    async def step_agent(self, agent_name: str, request: StepRequest):
        self._validate_agent_exists(agent_name)

        def transfer(obs):
            obs = base64.b64decode(obs)
            obs = pickle.loads(obs)
            return obs

        obs = transfer(request.observation)
        return self.preprocess_obs(obs)

    def preprocess_obs(self, obs: List[Dict[str, Any]]):
        obs[0] = remove_from_obs(obs[0])
        obs[0]['stage'] = self.current_stage  # Add current stage information to the observation
        orgin_rgb = obs[0]['rgb']

        # vit_start_time = time()
        # important_map = self.vision_encoder.get_patch_importance(orgin_rgb)
        # log.info(f"[TIME] On-device ViT infer time: {time() - vit_start_time:.2f}s.")
        # draw_heatmap_on_image(orgin_rgb, important_map)

        serialized_obs = serialize_obs(obs)
        upload_data_size = len(serialized_obs)  # in bytes
        log.info(f"Original observation size: {upload_data_size / 1024:.2f} KB")

        # estimated_transmission_delay = self.estimate_transmission_time(upload_data_size)
        # if estimated_transmission_delay is not None and estimated_transmission_delay > self.transmission_delay_threshold:
        # log.info(f"[TIME] Estimated transmission time: {estimated_transmission_delay:.4f}s")
        if self.if_compressed:
            preprocess_start_time = time()
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
        cloud_inference_latency = response_data['action'][0].pop('processing_time')
        cloud_response_latency = transmission_end_time - transmission_start_time
        transmission_latency = cloud_response_latency - cloud_inference_latency

        log.info(f"[TIME] Cloud inference time: {cloud_inference_latency:.4f}s")
        log.info(f"[TIME] Actual transmission time: {transmission_latency:.4f}s")

        # self.update_bandwidth(compressed_size, transmission_latency)

        if self.current_stage == 's2':
            traj_latents = cloud_data.get('traj_latents', None)  # obtain traj_latents for System1
            if traj_latents is not None:
                obs[0]['rgb'] = orgin_rgb  # Use original RGB for System1 processing
                self.s1_agent.record_goal_obs(obs[0], traj_latents)
                s1_response_data = self.s1_agent.step(obs[0])

                self.forward_step_num += 1
                self.current_stage = 's1'
                return s1_response_data
            else:
                response_data['action'][0].pop('traj_latents', None)  # Remove traj_latents if not present
                return response_data
        else:
            obs[0]['rgb'] = orgin_rgb
            s1_response_data = self.s1_agent.step(obs[0])
            self.forward_step_num += 1

            # if len(self.s1_agent.action_list) == 0 and self.s1_agent.ready_to_reach_goal:
            if self.forward_step_num > self.PLAN_STEP_GAP and len(self.s1_agent.action_list) == 0:
                self.current_stage = 's2'
                self.forward_step_num = 0
                self.s1_agent.ready_to_reach_goal = False
            return s1_response_data

    async def reset_agent(self, agent_name: str, request: ResetRequest):
        self._validate_agent_exists(agent_name)

        response = requests.post(
            url=f'{self.base_url}/agent/{self.agent_name}/reset',
            json=request.model_dump(mode='json'),
            headers={'Content-Type': 'application/json'},
        )
        response.raise_for_status()

        reset_index = getattr(request, 'reset_index', None)
        self.s1_agent.reset(reset_index)

        self.current_stage = 's2'  # Reset to initial stage after reset
        self.forward_step_num = 0

        return response.json()

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

    def _validate_agent_exists(self, agent_name: str):
        if agent_name != self.agent_name:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail='Agent not found')

    def run(self, reload=False):
        uvicorn.run(
            self.app,
            host=self.host,
            port=self.port,
            reload=reload,
            reload_dirs=['./internnav/agent/', './internnav/model/'],
        )


def start_server(host='localhost', port=8087, dist=False):
    """
    start a server in the backgrouond process

    Args:
        host
        port

    Returns:
        The rank of the process group
        -1, if not part of the group

    """
    ctx = multiprocessing.get_context("spawn")
    p = ctx.Process(target=_run_server if not dist else _run_server_dist, args=(host, port))
    p.daemon = True
    p.start()
    print(f"Server started on {host}:{port} (pid={p.pid})")
    return p


def _run_server_dist(host='localhost', port=8087):
    import torch

    from internnav.utils.dist import get_rank

    device_idx = get_rank()
    torch.cuda.set_device(device_idx)
    print(f"Server using GPU {device_idx}")
    server = AgentServer(host, port)
    server.run()


def _run_server(host='localhost', port=8087):
    server = AgentServer(host, port)
    server.run()
