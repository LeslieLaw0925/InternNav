rm -r logs data/sample_episodes

MESA_GL_VERSION_OVERRIDE=4.6 python scripts/eval/eval.py --config scripts/eval/configs/h1_internvla_n1_cfg.py 