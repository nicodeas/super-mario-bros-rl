# super-mario-bros-rl



https://github.com/nicodeas/super-mario-bros-rl/assets/77263595/9715b3ef-d875-48eb-bc42-b96b493602bf



https://github.com/nicodeas/super-mario-bros-rl/assets/77263595/b49a21f3-14ae-47d2-9529-89183f85448c



## Introduction

Super Mario Bros RL was initiated as an exploration into the fascinating world of reinforcement learning (RL). The goal of this project was to delve deeper into the principles and applications of RL, gaining experience in developing and implementing RL algorithms. Reinforcement learning algorithms such as Q-learning, Deep Q-learning (DQN), Double Deep Q-learning(DDQN) and Advantage Actor Critic(A2C) were explored before finally deciding to use Proximal Policy Optimisation(PPO). The reinforcment learning algorithms were explored in the exact order above. Other algorithms explored include a rule based agent and genetic algorithms. PPO code and a report on experiments carried out are provided in this repository.

## Challenges

As the first approach involved Q-learning, dealing using individual images as model state space resulted in a very large Q table. To simplify our statespace, we use template matching following this [documentation](https://docs.opencv.org/3.4/d4/dc6/tutorial_py_template_matching.html). This improved results and was then intergrated into Deep reinforcment learning algorithms later on through creating custom gym wrappers.

After further training, it was found that simple template matching was not sufficient which lead to experimentation with entire images provided by the environment with deep network iterations listed above.

## Quick Start

### Install Dependencies

1. Ensure [Anaconda](https://www.anaconda.com/download) has been installed
1. Create environment by running `conda env create -f environment.yml`

## Training Models

To train models, look in the `stable_baselines/train_image.py` file.

Configure:

1. Environment ID ( level you would like to train your model on)
1. Number of processes to run in parallel (8 CPUs were used in training)
1. Device, this is where you can specify whether to utilise cuda or mps
1. How frequent and where you would like models to be saved

To start training, run `python3 -m stable_baselines.train_image`

Training logs can be viewed with [Tensorboard](https://pytorch.org/tutorials/recipes/recipes/tensorboard_with_pytorch.html#run-tensorboard) by running `tensorboard --logdir=<training_log_directroy>`

## Evaluating

Set:

1. Environment ID ( level you would like to run your model on)
1. Model save location

To evaluate model, run `python3 -m stable_baselines.run_image`
