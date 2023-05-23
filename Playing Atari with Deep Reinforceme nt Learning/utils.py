import gymnasium as gym


def generate_env(env_id) -> gym.Env:
    env = gym.make(env_id, render_mode='human')
    env.metadata['render_fps'] = 60
    env = gym.wrappers.ResizeObservation(env, (84, 84))
    env = gym.wrappers.GrayScaleObservation(env)
    env = gym.wrappers.FrameStack(env, 4)
    return env


def make_env(env_id, seed):
    def thunk():
        env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        # if capture_video:
        #     if idx == 0:
        #         env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        env = gym.wrappers.ResizeObservation(env, (84, 84))
        env = gym.wrappers.GrayScaleObservation(env)
        env = gym.wrappers.FrameStack(env, 4)
        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk


def random_play(env_id, total_episodes=10):
    env = gym.make(env_id, render_mode='human')
    env.metadata['render_fps'] = 60
    for episode in range(total_episodes):
        observation = env.reset()
        is_done = False
        print(episode)
        while not is_done:
            action = env.action_space.sample()
            observation, reward, is_terminal_state, is_done, info = env.step(action)
            env.render()
            if is_terminal_state:
                break

