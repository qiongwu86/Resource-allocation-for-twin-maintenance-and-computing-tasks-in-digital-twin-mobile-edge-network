from marl_train import Runner, Agent
from setting import arg
from my_env import Environment
from marl_ddpg import MADDPG
from marl_train import Agent
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
import torch
import numpy as np
import matplotlib.pyplot as plt
import algorithm_SAC.RL_SAC as RL_SAC

if __name__ == '__main__':
    # get the params
    args = arg()
    env = Environment(args.lane_num, args.n_agents, args.width)
    runner = Runner(args, env)
    if args.evaluate:
        pass
    else:
        runner.run()