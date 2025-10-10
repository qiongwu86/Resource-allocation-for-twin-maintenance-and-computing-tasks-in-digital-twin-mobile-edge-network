import argparse
from MADDPG_MATD3_main import Runner
from matd3 import MATD3
from my_env import Environment
import copy
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Hyperparameters Setting for MADDPG and MATD3 in MPE environment")
    parser.add_argument("--max_train_steps", type=int, default=int(5e4), help=" Maximum number of training steps")
    parser.add_argument("--episode_limit", type=int, default=25, help="Maximum number of steps per episode")
    parser.add_argument("--evaluate_freq", type=float, default=100, help="Evaluate the policy every 'evaluate_freq' steps")
    parser.add_argument("--evaluate_times", type=float, default=3, help="Evaluate times")
    parser.add_argument("--max_action", type=float, default=0.3, help="Max action")

    parser.add_argument("--algorithm", type=str, default="MATD3", help="MADDPG or MATD3")
    parser.add_argument("--buffer_size", type=int, default=int(2e5), help="The capacity of the replay buffer")
    parser.add_argument("--batch_size", type=int, default=1024, help="Batch size")
    parser.add_argument("--hidden_dim", type=int, default=64, help="The number of neurons in hidden layers of the neural network")
    parser.add_argument("--noise_std_init", type=float, default=0.5, help="The std of Gaussian noise for exploration")
    parser.add_argument("--noise_std_min", type=float, default=0.05, help="The std of Gaussian noise for exploration")
    parser.add_argument("--noise_decay_steps", type=float, default=3e4, help="How many steps before the noise_std decays to the minimum")
    parser.add_argument("--use_noise_decay", type=bool, default=True, help="Whether to decay the noise_std")
    parser.add_argument("--lr_a", type=float, default=2e-4, help="Learning rate of actor")
    parser.add_argument("--lr_c", type=float, default=1e-3, help="Learning rate of critic")
    parser.add_argument("--gamma", type=float, default=0.95, help="Discount factor")
    parser.add_argument("--tau", type=float, default=0.01, help="Softly update the target network")
    parser.add_argument("--use_orthogonal_init", type=bool, default=True, help="Orthogonal initialization")
    parser.add_argument("--use_grad_clip", type=bool, default=True, help="Gradient clip")
    # --------------------------------------MATD3--------------------------------------------------------------------
    parser.add_argument("--policy_noise", type=float, default=0.08, help="Target policy smoothing")
    parser.add_argument("--noise_clip", type=float, default=0.1, help="Clip noise")
    parser.add_argument("--policy_update_freq", type=int, default=2, help="The frequency of policy updates")

    args = parser.parse_args()
    args.noise_std_decay = (args.noise_std_init - args.noise_std_min) / args.noise_decay_steps

runner = Runner(args, number=1, seed=0)
runner.agent_n = [MATD3(args, agent_id) for agent_id in range(args.N)]
for i in range(args.N):
    runner.agent_n[i].load_model(args.algorithm, runner.number, i) 
evaluate_reward = 0
T_dt = 0
T_tk = 0
for _ in range(args.evaluate_times):
    obs_n = runner.env_evaluate.make_new_game(runner.args.N)
    episode_reward = 0
    t_dt, t_tk = 0, 0
    for _ in range(runner.args.episode_limit):
        a_n = [agent.choose_action(obs, noise_std=0) for agent, obs in zip(runner.agent_n, obs_n)]  # We do not add noise when evaluating
        obs_next_n, _, r_n = runner.env_evaluate.step(copy.deepcopy(a_n))
        episode_reward += r_n#[0]
        obs_n = obs_next_n
        t_dt += np.mean(runner.env_evaluate.t_DT)
        t_tk += np.mean(runner.env_evaluate.t_TK)

    evaluate_reward += episode_reward/args.episode_limit
    T_dt += t_dt/args.episode_limit
    T_tk += t_tk/args.episode_limit

evaluate_reward = evaluate_reward / args.evaluate_times
T_dt = T_dt/args.evaluate_times
T_tk = T_tk/args.evaluate_times
print('mean return:', evaluate_reward, '\nDT maintaining:', T_dt, '\nTK processing:', T_tk)
