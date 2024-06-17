import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
sys.path.append(os.path.dirname(__file__))

from marl_train import Runner, Agent
from setting import arg
from my_env import Environment
from marl_ddpg import MADDPG as MADDPG5


def maddpg_test():
        agents = []
        for i in range(args.n_agents):
            Agent.policy = MADDPG5(args, i).load_model()
            agent = Agent(i, args)
            agents.append(agent)
        returns = []
        t_dt = []
        t_tk = []
        alloc_tk = []
        alloc_dt = []
        for episode in range(args.test_episodes):
            # reset the environment
            s = env.make_new_game(args.n_agents)
            rewards = 0
            time_dt = 0
            time_tk = 0
            allocation_tk = 0
            allocation_dt = 0
            for time_step in range(args.test_episode_len):
                #self.env.render()
                actions = []
                with torch.no_grad():
                    for agent_id, agent in enumerate(agents):
                        action = agent.select_action(s[agent_id], 0, 0)
                        actions.append(action)
                for i in range(len(actions)):    
                    actions[i] = actions[i]/2*100
                s_next, r = env.act_for_testing(actions)
                rewards += r
                time_dt += np.sum(env.t_DT)/(args.n_agents)
                time_tk += np.sum(env.t_TK)/args.n_agents
                allocation_tk +=np.sum(actions,axis=0)[0]
                allocation_dt +=np.sum(actions,axis=0)[1]
                s = s_next
            t_dt.append(time_dt/args.test_episode_len)
            t_tk.append(time_tk/args.test_episode_len)
            alloc_tk.append(allocation_tk/args.test_episode_len)
            alloc_dt.append(allocation_dt/args.test_episode_len)
            rewards = rewards/args.test_episode_len
            returns.append(rewards)
        #plt.plot(range(len(returns)),returns)
        #plt.show()
        print('\nReturns is', sum(returns) / args.test_episodes)#rewards)
        print('\nTime consumption for maintaining digital twins:', sum(t_dt)/args.test_episodes)
        print('\nTime consumption for computing tasks:', sum(t_tk)/args.test_episodes)
        print('\nResources for maintaining digital twins:', sum(alloc_dt)/args.test_episodes)
        print('\nResources for computation tasks:', sum(alloc_tk)/args.test_episodes)
        #return sum(returns) / args.evaluate_episodes

if __name__ == '__main__':
    # get the params
    args = arg()
    env = Environment(args.lane_num, args.n_agents, args.width)
    maddpg_test()