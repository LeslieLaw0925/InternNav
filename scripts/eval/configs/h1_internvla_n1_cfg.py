# from scripts.eval.configs.agent import *
from internnav.configs.agent import NewAgentCfg
from internnav.configs.evaluator import (
    EnvCfg,
    EvalCfg,
    EvalDatasetCfg,
    SceneCfg,
    TaskCfg,
)

eval_cfg = EvalCfg(
    agent=NewAgentCfg(
        cloud_server_host='192.168.105.11',
        cloud_server_port=30091,
        local_server_port=8023,
        model_name='internvla_n1_cloud',
        ckpt_path='',
        model_settings={},
    ),
    env=EnvCfg(
        env_type='internutopia',
        env_settings={
            'use_fabric': False,  # Please set use_fabric=False due to the render delay;
            'headless': True,
        },
    ),
    task=TaskCfg(
        task_name='test_n1',
        task_settings={
            'env_num': 1,
            'use_distributed': False,  # If the others setting in task_settings, please set use_distributed = False.
            'proc_num': 1,
            'max_step': 1000,  # If use flash mode，default 1000; descrete mode, set 50000
        },
        scene=SceneCfg(
            scene_type='mp3d',
            scene_data_dir='data/scene_data/mp3d_pe',
        ),
        robot_name='h1',
        robot_flash=True,  # If robot_flash is True, the mode is flash (set world_pose directly); else you choose physical mode.
        flash_collision=False,  # If flash_collision is True, the robot will stop when collision detected.
        robot_usd_path='data/Embodiments/vln-pe/h1/h1_internvla.usd',
        camera_resolution=[640, 480],  # (W,H)
        camera_prim_path='torso_link/h1_1_25_down_30',
        one_step_stand_still=True,  # For dual-system, please keep this param True.
    ),
    dataset=EvalDatasetCfg(
        dataset_type="mp3d",
        dataset_settings={
            'base_data_dir': 'data/vln_pe/raw_data/r2r',
            'split_data_types': ['val_unseen'],  # 'val_seen'
            'filter_stairs': True,  # For iros challenge, this is False; For results in the paper, this is True.
            # 'selected_scans': ['zsNo4HB9uLZ'],
            # 'selected_scans': ['8194nk5LbLH', 'pLe4wQe7qrG'],
        },
    ),
    eval_type='vln_distributed',
    eval_settings={
        'save_to_json': True,
        'vis_output': True,
        'use_agent_server': True,  # If use_agent_server=True, please start the agent server first.
    },
)