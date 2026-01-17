import subprocess
import base64
import pickle
from typing import Any, Dict, List, Optional

import requests

from internnav.configs.agent import AgentCfg, InitRequest, ResetRequest, StepRequest
from internnav.utils.comm_utils.s1_server import start_system1


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
        self.s1_server_process = start_system1(config)
        self.latest_traj_latents = None

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
        request_data = StepRequest(observation=serialize_obs(obs)).model_dump(mode='json')

        response = requests.post(
            url=f'{self.server_url}/agent/{self.agent_name}/step',
            json=request_data,
            headers={'Content-Type': 'application/json'},
        )
        response.raise_for_status()

        response_data = response.json()
        action: list = response_data.get('action')
        self.latest_traj_latents = action[0].pop('traj_latent', None)
        
        return action

    def reset(self, reset_index: Optional[List] = None) -> None:
        response = requests.post(
            url=f'{self.server_url}/agent/{self.agent_name}/reset',
            json=ResetRequest(reset_index=reset_index).model_dump(mode='json'),
            headers={'Content-Type': 'application/json'},
        )
        response.raise_for_status()

        

