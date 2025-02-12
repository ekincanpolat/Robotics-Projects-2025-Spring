from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from kuka_env import KukaEnv

env = DummyVecEnv([lambda: KukaEnv()])
# PPO model setup
#Policy of Neural N:"MLP" (Multi-Layer Perceptron)
#Verbose-> prints out the state
model = PPO("MlpPolicy", env, verbose=1)

model.learn(total_timesteps=500000)

model.save("ppo_kuka_iiwa")

print("Training finished! Model saved.")
