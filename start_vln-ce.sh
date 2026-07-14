export TRITON_PTXAS_PATH=$(which ptxas)

rm -r logs

python scripts/eval/eval.py --config scripts/eval/configs/dist_habitat_dual_system_cfg.py