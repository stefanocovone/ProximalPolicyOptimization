@echo off

REM Run the Python script 5 times with different parameters and experiment tags

echo RUNNING EXPERIMENT 1
python test_PPO_shepherding1M.py --k_R 0.1 --k_p 0 --k_all 0.0 --k_chi 0.0 --exp_tag "all_targets1"

echo RUNNING EXPERIMENT 2
python test_PPO_shepherding1M.py --k_R 0.01 --k_p 0 --k_all 0.0 --k_chi 0.2 --exp_tag "all_targets2"

echo RUNNING EXPERIMENT 3
python test_PPO_shepherding1M.py --k_R 0.1 --k_p 0 --k_all 0.5 --k_chi 0.0 --exp_tag "all_targets3"

REM Done
echo All experiments have been launched.
pause
