#!/usr/bin/env python
import base64
import multiprocessing
import pickle
from typing import Dict

import uvicorn
from fastapi import APIRouter, FastAPI, HTTPException, status

from internnav.agent.base import Agent
from internnav.configs.agent import AgentCfg, ResetRequest, StepRequest
from internnav.agent.internvla_n1_s1_agent import System1


class S1Server:
    """
    Server class for S1 fast service.
    """

    def __init__(self, config: AgentCfg):
        self.host = 'localhost'
        self.port = config.s1_port
        self.app = FastAPI(title='S1 Fast Service')
        self.s1 = System1(config)

        self.agent_instances: Dict[str, Agent] = {}
        self._router = APIRouter(prefix='/s1_agent')
        self._register_routes()
        self.app.include_router(self._router)

    def _register_routes(self):
        route_config = [
            ('/step', self.step_agent, ['POST'], None),
            ('/reset', self.reset_agent, ['POST'], None),
            # TODO: Add stop server route
        ]

        for path, handler, methods, status_code in route_config:
            self._router.add_api_route(
                path=path,
                endpoint=handler,
                methods=methods,
                status_code=status_code,
            )

    async def step_agent(self, agent_name: str, request: StepRequest):
        def transfer(obs):
            obs = base64.b64decode(obs)
            obs = pickle.loads(obs)
            return obs

        obs = transfer(request.observation)
        obs = obs[0]  # do not support batch_env currently?
        rgb = obs['rgb']
        depth = obs['depth']

        action = self.s1.step(rgb, depth)
        return {'action': action}

    async def reset_agent(self, agent_name: str, request: ResetRequest):
        self.s1.reset(getattr(request, 'reset_index', None))
        return {'status': 'success'}

    def _validate_agent_exists(self, agent_name: str):
        if agent_name not in self.agent_instances:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail='Agent not found')

    def run(self, reload=False):
        uvicorn.run(
            self.app,
            host=self.host,
            port=self.port,
            reload=reload,
            reload_dirs=['./internnav/agent/', './internnav/model/'],
        )


def start_system1(config: AgentCfg, dist=False):
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
    p = ctx.Process(target=_run_server if not dist else _run_server_dist, args=(config,))
    p.daemon = True
    p.start()
    print(f"S1 Server starts (pid={p.pid})")
    return p


def _run_server_dist(config: AgentCfg):
    import torch

    from internnav.utils.dist import get_rank

    device_idx = get_rank()
    torch.cuda.set_device(device_idx)
    print(f"Server using GPU {device_idx}")
    server = S1Server(config)
    server.run()


def _run_server(config: AgentCfg):
    server = S1Server(config)
    server.run()
