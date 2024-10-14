@echo off

REM Run the Python script 5 times with different parameters and experiment tags

echo RUNNING EXPERIMENT 1
python test_PPO_shepherding1M.py --k_R 0.01 --k_p 0 --k_all 0.0 --k_chi 0.1 --exp_tag "test2_1" --num_targets 3

echo RUNNING EXPERIMENT 1
python test_PPO_shepherding1M.py --k_R 0.01 --k_p 0 --k_all 0.0 --k_chi 0.1 --exp_tag "test2_1" --num_targets 5

echo RUNNING EXPERIMENT 1
python test_PPO_shepherding1M.py --k_R 0.01 --k_p 0 --k_all 0.0 --k_chi 0.1 --exp_tag "test2_1" --num_targets 7

echo RUNNING EXPERIMENT 2
python test_PPO_shepherding1M.py --k_R 0.01 --k_p 0 --k_all 0.0 --k_chi 0.1 --exp_tag "test2_2" --num_targets 3

echo RUNNING EXPERIMENT 2
python test_PPO_shepherding1M.py --k_R 0.01 --k_p 0 --k_all 0.0 --k_chi 0.1 --exp_tag "test2_2" --num_targets 5

echo RUNNING EXPERIMENT 2
python test_PPO_shepherding1M.py --k_R 0.01 --k_p 0 --k_all 0.0 --k_chi 0.1 --exp_tag "test2_2" --num_targets 7

echo RUNNING EXPERIMENT 3
python test_PPO_shepherding1M.py --k_R 0.01 --k_p 0 --k_all 0.0 --k_chi 0.1 --exp_tag "test2_3" --num_targets 3

echo RUNNING EXPERIMENT 3
python test_PPO_shepherding1M.py --k_R 0.01 --k_p 0 --k_all 0.0 --k_chi 0.1 --exp_tag "test2_3" --num_targets 5

echo RUNNING EXPERIMENT 3
python test_PPO_shepherding1M.py --k_R 0.01 --k_p 0 --k_all 0.0 --k_chi 0.1 --exp_tag "test2_3" --num_targets 7

echo RUNNING EXPERIMENT 4
python test_PPO_shepherding1M.py --k_R 0.01 --k_p 0 --k_all 0.0 --k_chi 0.1 --exp_tag "test2_4" --num_targets 3

echo RUNNING EXPERIMENT 4
python test_PPO_shepherding1M.py --k_R 0.01 --k_p 0 --k_all 0.0 --k_chi 0.1 --exp_tag "test2_4" --num_targets 5

echo RUNNING EXPERIMENT 4
python test_PPO_shepherding1M.py --k_R 0.01 --k_p 0 --k_all 0.0 --k_chi 0.1 --exp_tag "test2_4" --num_targets 7


REM Done
echo All experiments have been launched.
pause
