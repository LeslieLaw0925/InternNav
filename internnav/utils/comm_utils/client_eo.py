import base64
import pickle
from typing import Any, Dict, List, Optional
from time import time

import numpy as np
import requests

from internnav.configs.agent import NewAgentCfg, InitRequest, ResetRequest, StepRequest
from internnav.utils.comm_utils.system_log import InferenceLogger


class system_perf(enumerate):
    TRANSMISSION = "transmission_time"
    S1 = "s1_time"
    S2 = "s2_time"
    BW = "bandwidth"
    STEP = "step_time"
    STEP_ID = "step_id"


def remove_from_obs(obs: dict) -> dict:
    keys = ['globalgps', 'globalrotation', 'topdown_rgb', 
            'topdown_depth', 'instruction_tokens', 'depth'] # depth
    for key in keys:
        obs.pop(key, None)
    return obs


def serialize_obs(obs):
    serialized = pickle.dumps(obs)
    encoded = base64.b64encode(serialized).decode('utf-8')
    return encoded


class AgentClient:
    """
    Client class for Agent service.
    """

    def __init__(self, config: NewAgentCfg):
        self.base_url = f'http://{config.cloud_server_host}:{config.cloud_server_port}'
        self.agent_name = self._initialize_agent(config)
        self.inference_logger = InferenceLogger()

    def _initialize_agent(self, config: NewAgentCfg) -> str:
        request_data = InitRequest(agent_config=config).model_dump(mode='json')

        response = requests.post(
            url=f'{self.base_url}/agent/init',
            json=request_data,
            headers={'Content-Type': 'application/json'},
        )
        response.raise_for_status()

        return response.json()['agent_name']

    def step(self, obs: List[Dict[str, Any]]) -> List[List[int]]:
        obs[0] = remove_from_obs(obs[0])
        import pdb; pdb.set_trace()
        # obs[0]['depth'] = obs[0]['depth'].astype(np.uint16)

        start_time = time()
        request_data = StepRequest(observation=serialize_obs(obs)).model_dump(mode='json')

        response = requests.post(
            url=f'{self.base_url}/agent/{self.agent_name}/step',
            json=request_data,
            headers={'Content-Type': 'application/json'},
        )
        response.raise_for_status()
        step_time = time() - start_time
        
        response_data = response.json()['action']
        cloud_processing_time = response_data[0].pop('step_time')
        transmission_time = step_time - cloud_processing_time
        s1_time = response_data[0].pop('s1_time', None)
        s2_time = response_data[0].pop('s2_time', None)

        if s1_time is not None:
            self.inference_logger.record_by_key(system_perf.S1, s1_time)
        if s2_time is not None:
            self.inference_logger.record_by_key(system_perf.S2, s2_time)

        self.inference_logger.record_by_key(system_perf.TRANSMISSION, transmission_time)
        self.inference_logger.record_by_key(system_perf.STEP, step_time)
        self.inference_logger.flush()
        return response_data

    def reset(self, reset_index: Optional[List] = None) -> None:
        response = requests.post(
            url=f'{self.base_url}/agent/{self.agent_name}/reset',
            json=ResetRequest(reset_index=reset_index).model_dump(mode='json'),
            headers={'Content-Type': 'application/json'},
        )
        response.raise_for_status()
