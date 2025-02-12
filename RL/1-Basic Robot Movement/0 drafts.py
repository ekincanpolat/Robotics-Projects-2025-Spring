import gym
from stable_baselines3 import PPO
from kuka_env import KukaEnv  # Kendi oluşturduğumuz environment'ı import ettik

# Ortamı başlat
env = KukaEnv()

# PPO ajanını oluştur
model = PPO("MlpPolicy", env, verbose=1)

# Ajanı eğit
model.learn(total_timesteps=100000)

# Modeli kaydet
model.save("ppo_kuka_iiwa")

print("Eğitim tamamlandı ve model kaydedildi!")
