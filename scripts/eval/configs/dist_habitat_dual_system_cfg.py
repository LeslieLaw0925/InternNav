from internnav.configs.agent import AgentCfg, NewAgentCfg
from internnav.configs.evaluator import EnvCfg, EvalCfg

eval_cfg = EvalCfg(
    agent=NewAgentCfg(
        model_name='habitat_s2_agent',
        ckpt_path='',
        local_server_port=8023,
        model_settings={},
    ),
    env=EnvCfg(
        env_type='habitat',
        env_settings={
            # habitat sim specifications - agent, sensors, tasks, measures etc. are defined in the habitat config file
            'config_path': 'scripts/eval/configs/vln_r2r.yaml',
        },
    ),
    eval_type='habitat_vln',
    eval_settings={
        # all current parse args
        "output_path": "./logs/habitat/test_dual_system",  # output directory for logs/results
        "save_video": False,  # whether to save videos
        "epoch": 0,  # epoch number for logging
        "max_steps_per_episode": 500,  # maximum steps per episode
        # distributed settings
        "port": "2333",  # communication port
        "dist_url": "env://",  # url for distributed setup
    },
)
