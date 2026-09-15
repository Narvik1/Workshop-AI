# Example 3.23 OpenAI Gym CartPole (Diperbarui untuk Gymnasium)
import gymnasium as gym

# Menggunakan versi v1 dan menambahkan render_mode agar animasi terlihat
env = gym.make('CartPole-v1', render_mode="human")

for i_episode in range(20):
    # reset() sekarang mengembalikan observation dan info
    observation, info = env.reset()
    
    for t in range(100):
        # env.render() otomatis tertangani oleh render_mode="human", 
        # tapi kita tetap bisa memanggilnya
        env.render()
        
        # Mengambil aksi secara acak
        action = env.action_space.sample()
        
        # step() sekarang mengembalikan 5 nilai
        observation, reward, terminated, truncated, info = env.step(action)
        
        # Kondisi 'done' terjadi jika episode selesai (terminated) atau batas waktu habis (truncated)
        done = terminated or truncated
        
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break # Keluar dari loop t dan lanjut ke episode berikutnya

env.close()