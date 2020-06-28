# About this project

This projects contains my solution of the third project in the **[Udacity Deep Reinforcement Learning Nanodegree](https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893)** of [Udacity](https://www.udacity.com/).
The goal of this project is to train two agents to play tennis.




# The Environment

This project uses an environmens similar but not ideantical to the [Tennis](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#tennis) Unity ML-Agents environment.

In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1. If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01. Thus, the goal of each agent is to keep the ball in play.

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation. Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping. 

![Trained Agent](./img/tennis.png)

The task is episodic, and the environment is considered solved, when the average of those scores is at least +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). Specifically,

- After each episode, the rewards that each agent received (without discounting) are added up, to get a score for each agent. This yields 2 (potentially different) scores. Then the maximum of these 2 scores are taken. 
- This yields a single score for each episode.


To run the code in this project, the specified environment of Udacity is needed. To set it up, follow the instructions below.

## Step 1 - Getting started
Install PyTorch, the ML-Agents toolkit, and a few more Python packages according to the [instructions in the DRLND GitHub repository](https://github.com/udacity/deep-reinforcement-learning#dependencies).

Furthermore, install the [dataclasses](https://docs.python.org/3/library/dataclasses.html) python module.

## Step 2 - Download the Unity Environment
For this project, you **don't** need to install Unity. Instead, choose the pre-built environmen provided by Udacity matching your operating system:

* Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
* Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
* Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)
* Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)


# Instructions

To explore the environment and train the agent, start a jupyter notebook, open Tennis_ddpg.ipynb and execute the steps. For more information, and an exmaple on how to use the agent, please check instructions inside the notebook.

## Project structure

* `Tennis_ddpg.ipynb`: the jupyter notebook for executing the training of the DDPG agent
* `Tennis_maddpg.ipynb`: the jupyter notebook for executing the training of the MADDPG agent
* `src\ddpg.py` : the implementation of the DDPG agent
* `src\maddpg.py` : the implementation of the DDPG agent
* `src\model.py` : the PyTorch models of the neural networks used by the Agent
* `src\replay_buffer.py` : The replay buffer implementation for memory
* `src\config.py` : the default configuration/hyperparameters of the models
* `src\noise.py`  : the implementation of the Ornstein-Uhlenbeck process
* `results\`  : contains the trained agents weights

# Results

The trained agent solved the environment in 969 episodes.
For a detailed explanation, please read the [project report](./Report.md)


# Notes
The project uses the code and task description provided in the **[Udacity Deep Reinforcement Learning Nanodegree](https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893)**  class as a basis.
