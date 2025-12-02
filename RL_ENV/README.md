

## Setup

To install, clone the repository and run the following:

```bash 
git submodule update --init --recursive
pip install -r requirements.txt
```

The code was tested on Python 3.8 and 3.9.
If you don't have MuJoCo installed, follow the instructions here: https://github.com/openai/mujoco-py#install-mujoco.

## Running Instructions

Example:

```bash
# DMC environments.
python3 synther/online/online_exp.py --env quadruped-walk-v0 --results_folder online_logs/ --exp_name SynthER --gin_config_files 'config/online/sac_synther_dmc.gin' --gin_params 'redq_sac.utd_ratio = 20' 'redq_sac.num_samples = 1000000'

# OpenAI environments (different gin config).
python3 synther/online/online_exp.py --env HalfCheetah-v2 --results_folder online_logs/ --exp_name SynthER --gin_config_files 'config/online/sac_synther_openai.gin' --gin_params 'redq_sac.utd_ratio = 20' 'redq_sac.num_samples = 1000000'
```
