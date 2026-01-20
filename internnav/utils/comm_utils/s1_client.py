import base64
import pickle
from typing import Any, Dict, List, Optional

import requests

from internnav.configs.agent import AgentCfg, InitRequest, ResetRequest, StepRequest
from internnav.agent.internvla_n1_s1_agent import System1


def serialize_obs(obs):
    serialized = pickle.dumps(obs)
    encoded = base64.b64encode(serialized).decode('utf-8')
    return encoded


class S1AgentClient:
    """
    Client class for Agent service with local S1.
    """

    def __init__(self, config: AgentCfg):        
        self.server_url = f'http://{config.server_host}:{config.server_port}'

        self.agent_name = self._initialize_agent(config)
        self.s1_server = System1(config)

        self.latest_traj_latents = None
        self.pixel_goal_rgb = None
        self.pixel_goal_depth = None
        self.last_action: list = None
        self.step_flag = 1

    def _initialize_agent(self, config: AgentCfg) -> str:
        request_data = InitRequest(agent_config=config).model_dump(mode='json')

        response = requests.post(
            url=f'{self.server_url}/agent/init',
            json=request_data,
            headers={'Content-Type': 'application/json'},
        )
        response.raise_for_status()

        return response.json()['agent_name']

    def step(self, obs: List[Dict[str, Any]]) -> List[List[int]]:
        if self.step_flag:
            return self.cloud_step(obs)
        else:
            return self.local_s1_step(obs)
    
    def local_s1_step(self, obs: List[Dict[str, Any]]) -> List[List[int]]:
        obs[0]['traj_latent'] = self.latest_traj_latents
        obs[0]['pixel_goal_rgb'] = self.pixel_goal_rgb
        obs[0]['pixel_goal_depth'] = self.pixel_goal_depth

        return self.s1_server.step(obs[0])
    
    def cloud_step(self, obs: List[Dict[str, Any]]) -> List[List[int]]:
        if self.last_action == [5]:
            self.pixel_goal_rgb = obs[0].get('rgb')
            self.pixel_goal_depth = obs[0].get('depth')

        request_data = StepRequest(observation=serialize_obs(obs)).model_dump(mode='json')

        response = requests.post(
            url=f'{self.server_url}/agent/{self.agent_name}/step',
            json=request_data,
            headers={'Content-Type': 'application/json'},
        )
        response.raise_for_status()

        response_data = response.json()
        action: list = response_data.get('action')
        self.latest_traj_latents = action[0].pop('traj_latent')
        self.last_action = action[0].get('action')

        return action

    def reset(self, reset_index: Optional[List] = None) -> None:
        response = requests.post(
            url=f'{self.server_url}/agent/{self.agent_name}/reset',
            json=ResetRequest(reset_index=reset_index).model_dump(mode='json'),
            headers={'Content-Type': 'application/json'},
        )
        response.raise_for_status()

        self.latest_traj_latents = None
        self.pixel_goal_rgb = None
        self.pixel_goal_depth = None
        self.last_action: list = None
        self.step_flag = 1