@echo off

REM Run the Python script 5 times with different parameters and experiment tags

echo RUNNING EXPERIMENT 2
python test_PPO_shepherding1M.py --k_R 0.01 --k_p 0 --k_all 0.0 --k_chi 0.2 --exp_tag "test_5" --l_r 0.0003

echo RUNNING EXPERIMENT 3
python test_PPO_shepherding1M.py --k_R 0.01 --k_p 0 --k_all 0.0 --k_chi 0.2 --exp_tag "test_5" --l_r 0.0002

echo RUNNING EXPERIMENT 4
python test_PPO_shepherding1M.py --k_R 0.01 --k_p 0 --k_all 0.0 --k_chi 0.2 --exp_tag "test_5" --l_r 0.0001

echo RUNNING EXPERIMENT 1
python test_PPO_shepherding1M.py --k_R 0.01 --k_p 1 --k_all 0 --k_chi 0.0 --exp_tag "test3_1" --l_r 0.0003

REM Done
echo All experiments have been launched.
pause
