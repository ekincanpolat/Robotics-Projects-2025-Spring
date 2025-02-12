import time
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from kuka_env import KukaEnv

model = PPO.load("ppo_kuka_iiwa")

env = DummyVecEnv([lambda: KukaEnv()])

# test the model for 20 steps
obs = env.reset()

for _ in range(40):
    action, _states = model.predict(obs)  # PPO agent produces actions according to the existing observation
    obs, rewards, dones, infos = env.step(action)
    env.render()  # visualize
    time.sleep(0.09)

    if dones.any():
        obs = env.reset()

env.close()
print("Test completed!")
