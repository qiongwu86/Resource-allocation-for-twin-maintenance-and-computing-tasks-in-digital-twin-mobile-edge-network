from marl_train import Runner, Agent
from setting import arg
import my_env
import numpy as np
import torch 
#np.random.seed(1234)
#torch.manual_seed(1234)

if __name__ == '__main__':
    # get the params
    args = arg()
    env = my_env.Environment(args.lane_num, args.n_agents)
    runner = Runner(args, env)
    if args.evaluate:
        pass
    else:
        runner.run()

        