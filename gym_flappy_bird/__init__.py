from gym.envs.registration import register

register(
    id='FlappyBird-v0',
    entry_point='gym_flappy_bird.envs:FlappyBirdEnv',
)

register(
    id='FlappyBirdFeature-v0',
    entry_point='gym_flappy_bird.envs:FlappyBirdFeatureEnv',
)