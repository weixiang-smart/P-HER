# P-HER
The code of paper "Trajectory Progress-based Prioritizing and Intrinsic Reward Mechanism for Robust Training of Robotic Manipulations" submitted to T-ASE.\
Our code is developed based on [OpenAI Baselines](https://github.com/openai/baselines)


## Requirement(important)
- Python==3.6.13
- tensorflow==1.15.0
- numpy==1.19.5
- mujoco==2.0.0
- mujoco_py==2.0.13
- mpi4py==3.1.4
- gym==0.15.7
- panda-gym==2.0.0 forked from qgallouedec/panda-gym (https://github.com/weixiang-smart/panda-gym)

## Installation
 `pip install -e .`

## Usage
1. Open the terminal in ./basedlines/her/experiment

2. Train the model with P-HER in PandaPickAndPlaceJoints-v2  by running the command
```
python train.py --env_name PandaPickAndPlaceJoints-v2  --prioritization motivation --ratio_o 0.75 --ratio 0.25 --seed 2 --n_epochs 100 --num_cpu 8 --logdir logs/PandaPickAndPlaceJoints-v2/test/7525/finaltest/r2 --logging True
```

## Reference
- [OpenAI Baselines](https://github.com/openai/baselines)
- [Energy-Based Hindsight Experience Prioritization](https://github.com/ruizhaogit/EnergyBasedPrioritization)
- [Diversity-based Trajectory and Goal Selection with Hindsight Experience Replay](https://github.com/TianhongDai/div-hindsight)
