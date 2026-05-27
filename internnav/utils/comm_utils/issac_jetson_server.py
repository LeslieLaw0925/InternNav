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
import torch
import yaml

from internnav.agent.base import Agent
from internnav.configs.agent import InitRequest, ResetRequest, StepRequest
from internnav.configs.agent import NewAgentCfg
from internnav.agent.internvla_n1_s1_agent import System1
from internnav.utils.common_log_util import common_logger as log
from internnav.utils.comm_utils.visual_encoder import VisionEncoder, numpy_compression_v2, draw_heatmap_on_image, \
    draw_origin_image, random_compression
from internnav.utils.comm_utils.client_utils import *
from internnav.utils.comm_utils.system_log import InferenceLogger


class system_perf(enumerate):
    TRANSMISSION = "transmission_time"
    S1 = "s1_time"
    S2 = "s2_time"
    BW = "bandwidth"
    STEP = "step_time"
    STEP_ID = "step_id"
    COMP_RATIO = "comp_ratio"
    # EPISODE_ID = "episode_id"


class IssacAgentServer:
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
        self.device = "cuda"
        self.dtype = torch.float16
        self.inference_logger = InferenceLogger()

        with open('scripts/eval/configs/latency_profile.yaml', 'r', encoding='utf-8') as f:
            self.infer_profile_data: dict = yaml.safe_load(f)

        self.s1_agent = System1(config, 
                                self.infer_profile_data.get('s1_infer'),
                                infer_logger=self.inference_logger,
                                device=self.device, dtype=self.dtype)
        vln_sensor_config = config.model_settings
        self.s1_type = vln_sensor_config.get('s1_type')
        self.vision_encoder = VisionEncoder(self.s1_type, device=self.device)

        self.image_compression_fachtor = 4
        self.e2e_latency_threshold = 1.5 # seconds
        self.cloud_latency_threshold = 0.5 # TODO: 需要合理设置这个值, seconds

        self.ema_bandwidth = None
        self.current_stage = 's2'
        self.forward_step_num = 0
        self.PLAN_STEP_GAP = 8

        self.set_adaptive_compression = vln_sensor_config.get('adaptive_compression', False)
        self.if_compressed = False
        # import pdb; pdb.set_trace()
        self.episode = 0
        os.makedirs(f"logs/test_data/episode_{self.episode}", exist_ok=True)
        
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

    def switch_stage(self):
        force_s2 = (self.forward_step_num > self.PLAN_STEP_GAP) and len(self.s1_agent.action_list) == 0

        stage = 's2' if force_s2 else 's1'
        return stage
    
    def _request_s2(self, obs: List[Dict[str, Any]], start_time: float):
        origin_rgb, depth = obs[0]['rgb'], obs[0].pop('depth', None)

        if self.set_adaptive_compression:
            # Estimate cloud latency
            estimated_cloud_time = self.estimate_cloud_latency(len(serialize_obs(obs)))
            preprocess_time = time() - start_time
            self.if_compressed = (estimated_cloud_time is not None) and \
                ((estimated_cloud_time + preprocess_time) > self.cloud_latency_threshold)
            log.info(f"Image compression needed: {self.if_compressed}")

        response_data = self._transmit_obs(obs, start_time) # transmit the continuous observations to the server and get the response for system2
        cloud_data: dict = response_data['action'][0]
        traj_latents = cloud_data.get('traj_latents', None)

        if traj_latents is not None:
            traj_latents = torch.from_numpy(np.array(traj_latents)).to(self.device, self.dtype)
            
            obs[0]['rgb'] = origin_rgb
            obs[0]['depth'] = depth
            self.s1_agent.record_goal_obs(obs[0], traj_latents)
            
            s2_elasped_time = time() - start_time
            obs[0]['latency_constraint'] = self.e2e_latency_threshold - s2_elasped_time
            s1_response_data = self.s1_agent.step(obs[0])

            self.forward_step_num += 1
            self.current_stage = 's1'
            return s1_response_data
        
        return response_data
    
    def _transmit_obs(self, obs: List[Dict[str, Any]], start_time: float):
        origin_upload_size = None
        if self.if_compressed:
            image = obs[0]['rgb']
            vit_latency, patch_importance = self.vision_encoder.get_patch_importance(image)
            log.info(f"[TIME] ViT inference latency: {vit_latency:.4f}s")
            preprocess_time = time() - start_time
            time_constraint = self.cloud_latency_threshold - vit_latency - preprocess_time
            log.info(f"[CONSTRAINT] Remaining time constraint for image transmission: {time_constraint:.4f} seconds.")
            p_star = solve_optimal_patch_ratio(image,
                                               time_constraint,
                                               self.ema_bandwidth,
                                               compression_factor=self.image_compression_fachtor)
            log.info(f"Calculated patch keep ratio (p_star): {p_star:.4f}")
            if p_star < 1.0:
                origin_upload_size = len(serialize_obs(obs))
                obs[0]['rgb'] = numpy_compression_by_patch(image, 
                                                           patch_importance, 
                                                           keep_ratio=p_star, 
                                                           compression_factor=self.image_compression_fachtor)
                obs[0]['compressed'] = 1  # Indicate that the RGB has been compressed
            self.if_compressed = False

        serialized_obs = serialize_obs(obs)
        upload_data_size = len(serialized_obs)  # in bytes
        log.info(f"Upload observation size: {upload_data_size / 1024:.2f} KB")

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
        cloud_inference_latency = response_data['action'][0].pop('processing_time')
        cloud_response_latency = transmission_end_time - transmission_start_time
        transmission_latency = cloud_response_latency - cloud_inference_latency

        log.info(f"[TIME] Actual cloud inference time: {cloud_inference_latency:.4f}s")
        log.info(f"[TIME] Actual transmission time: {transmission_latency:.4f}s")

        self.inference_logger.record_by_key(system_perf.S2, cloud_inference_latency)
        self.inference_logger.record_by_key(system_perf.TRANSMISSION, transmission_latency)

        self.update_bandwidth(upload_data_size, transmission_latency)
        self.inference_logger.record_by_key(system_perf.BW, self.ema_bandwidth)
        if origin_upload_size is not None:
            self.inference_logger.record_by_key(system_perf.COMP_RATIO, \
                                                (origin_upload_size - upload_data_size) / origin_upload_size)
        return response_data
    
    def preprocess_obs(self, obs: List[Dict[str, Any]]):
        start_time = time()
        obs[0] = remove_from_obs(obs[0])
        
        if self.current_stage == 's1':
            self.current_stage = self.switch_stage()

        obs[0]['stage'] = self.current_stage  # Add current stage information to the observation
        if self.current_stage == 's2':
            self.forward_step_num = 0            
            response_data = self._request_s2(obs, start_time)
        else:
            origin_rgb, depth = obs[0]['rgb'], obs[0].pop('depth', None)

            if self.set_adaptive_compression:
                # Estimate transmission latency
                estimated_transmission_time = self.cal_transmission_time(len(serialize_obs(obs)))
                preprocess_time = time() - start_time
                self.if_compressed = (estimated_transmission_time is not None) and \
                    ((estimated_transmission_time + preprocess_time) > self.cloud_latency_threshold)
                log.info(f"Image compression needed: {self.if_compressed}")

            self._transmit_obs(obs, start_time) # transmit the continuous observations to the server

            obs[0]['rgb'] = origin_rgb # restore 
            obs[0]['depth'] = depth # restore 
            preprocess_time = time() - start_time
            obs[0]['latency_constraint'] = self.e2e_latency_threshold - preprocess_time
            response_data = self.s1_agent.step(obs[0])
            
            self.forward_step_num += 1

        self.inference_logger.record_by_key(system_perf.STEP, time() - start_time)
        self.inference_logger.flush()

        return response_data

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
        self.episode += 1
        os.makedirs(f"logs/test_data/episode_{self.episode}", exist_ok=True)

        self.inference_logger.reset()
        return response.json()

    def estimate_cloud_latency(self, upload_size_bytes):
        estimate_transmission_time = self.cal_transmission_time(upload_size_bytes)
        if estimate_transmission_time is None:
            return
        
        # max_s2_inference_time = self.infer_profile_data['s2_infer']['max_inference_time']
        return estimate_transmission_time

    def cal_transmission_time(self, upload_size_bytes):
        if self.ema_bandwidth is None:
            return 
        
        return upload_size_bytes / self.ema_bandwidth
    
    def update_bandwidth(self, data_size, transmission_latency, alpha=0.3):
        '''
        Update the bandwidth estimation.

        Args:
            data_size: The size of the data transmitted (in bytes).
            upload_latency: The time taken for the transmission (in seconds).
            alpha: The smoothing factor for EMA.
        '''
        bandwidth = data_size / transmission_latency # Bytes/s
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
