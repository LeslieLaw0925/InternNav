from internnav.configs.agent import NewAgentCfg
from internnav.configs.evaluator import EnvCfg, EvalCfg

eval_cfg = EvalCfg(
    agent=NewAgentCfg(
        model_name='internvla_n1',
        ckpt_path='',
        model_settings={
            "mode": "dual_system",  # inference mode: dual_system or system2
            # "model_path": "checkpoints/InternVLA-N1-w-NavDP",
            "model_path": "checkpoints/InternVLA-N1-DualVLN", 
            's1_type': 'nextdit_async', # 'nextdit_async' or 'navdp_async'
            'nextdit_pretrained': "checkpoints/nextdit_from_dual_vln.ckpt",
            'navdp_pretrained': "checkpoints/navdp_from_w_navdp.ckpt",
            'adaptive_compression': False,
            'adaptive_speedup': False,
            "num_history": 8,
            "resize_w": 384,  # image resize width
            "resize_h": 384,  # image resize height
            'width': 640,
            'height': 480,
            "max_new_tokens": 1024,  # maximum number of tokens for generation
            "vis_debug": False,  # If vis_debug=True, save debug videos per episode
            "vis_debug_path": "./logs/habitat/vis_debug",
        },
    ),
    env=EnvCfg(
        env_type='habitat',
        env_settings={
            # habitat sim specifications - agent, sensors, tasks, measures etc. are defined in the habitat config file
            'config_path': 'scripts/eval/configs/vln_rxr.yaml',
        },
    ),
    eval_type='cloud_habitat_vln',
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
