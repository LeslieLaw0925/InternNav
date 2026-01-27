from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class AgentCfg(BaseModel):
    cloud_server_host: str = 'localhost'
    cloud_server_port: int = 8087
    local_server_port: int = 8022
    model_name: str
    ckpt_path: str = None
    model_settings: Dict[str, Any]


class InitRequest(BaseModel, extra='allow'):
    agent_config: AgentCfg


class StepRequest(BaseModel, extra='allow'):
    observation: Any


class ResetRequest(BaseModel):
    reset_index: Optional[List]
    partial_reset: bool = False


__all__ = ['AgentCfg', 'InitRequest', 'StepRequest', 'ResetRequest']
