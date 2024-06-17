class arg(object):
    def __init__(self) -> None:
        self.lane_num = 3
        self.n_agents = 5
        self.width = 120
        self.n_hidden_1 = 300
        self.n_hidden_2 = 100
        self.actor_hidden = 2
        self.critic_hidden = 1
        self.n_features = 9
        self.n_output = 2
        self.action_bound = 0.22
        self.gamma = 0.95
        self.tau = 0.01
        self.noise_rate = 0.1
        self.lr_actor = 2e-4
        self.lr_critic = 1e-3
        self.epsilon = 0.1
        self.max_episode_len = 100
        self.time_steps = 50000
        self.buffer_size = int(2e5)
        self.batch_size = 64
        self.evaluate_episodes = 10
        self.evaluate_episode_len = 100
        self.evaluate_rate = 100
        self.evaluate = False
        self.test_episodes = 10
        self.test_episode_len = 100
        self.save_rate = 2000
        self.save_dir = 'model'
        self.scenario_name = 'marl_model'
        self.state_dim = self.n_features*self.n_agents
        self.action_dim = 2*self.n_agents

