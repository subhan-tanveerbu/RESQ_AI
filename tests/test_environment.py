from config import RescueConfig
from environment.rescue_env import RescueEnv

def test_environment_reset_and_step():
    env = RescueEnv(RescueConfig(seed=1))
    obs, info = env.reset(seed=2)
    assert obs.shape[0] == env.observation_size
    assert info["health"] > 0
    for _ in range(5):
        obs, reward, done, info = env.step(4)
        assert len(obs) == env.observation_size
        assert isinstance(reward, float)
        if done: break

def test_observation_is_finite():
    env = RescueEnv()
    obs, _ = env.reset()
    assert obs.dtype.name == "float32"
    assert all(map(lambda x: x == x, obs))
