env = gym.make(ENV)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set seeds
env.seed(SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
max_action = float(env.action_space.high[0])

policy = TD3(state_dim, action_dim, max_action, env)

replay_buffer = ReplayBuffer()

runner = Runner(env, policy, replay_buffer)

total_timesteps = 0
timesteps_since_eval = 0
episode_num = 0
done = True

# Populate replay buffer
observe(env, replay_buffer, OBSERVATION)


# Train agent
train(policy, env)


policy.load()

for i in range(100):
    evaluate_policy(policy, env, render=True)

env.close()