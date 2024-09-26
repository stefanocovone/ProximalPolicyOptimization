@echo off
REM Initialize Conda and activate a specific environment

REM Initialize Conda (replace with your actual Conda path)
CALL C:\Users\YourUserName\Anaconda3\Scripts\activate.bat

REM Activate your Conda environment (replace 'myenv' with your environment name)
conda activate stefano_env

REM Run the Python script 5 times with different parameters and experiment tags

python test_PPO_shepherding1M.py --k_R 0.01 --k_p 5 --k_all 0.5 --k_chi 0.0 --exp_tag "test_2"
python test_PPO_shepherding1M.py --k_R 0.01 --k_p 0 --k_all 0.1 --k_chi 0.0 --exp_tag "test_3"
python test_PPO_shepherding1M.py --k_R 0.01 --k_p 0 --k_all 1.0 --k_chi 0.0 --exp_tag "test_4"
python test_PPO_shepherding1M.py --k_R 0.01 --k_p 5 --k_all 0.1 --k_chi 0.0 --exp_tag "test_1"
python test_PPO_shepherding1M.py --k_R 0.01 --k_p 0 --k_all 0.0 --k_chi 0.1 --exp_tag "test_5"

REM Done
echo All experiments have been launched.
pause
