import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

# 변수
n_particles = 100000 #입자 수
t_final = 10.0 # 계산할 시간
dt_sim = 0.1 # 시뮬레이션 dt
d_coeff = 0.1 # 확산 계수
grid_size = 101 # 격자 크기
spatial_step = 0.1 # 격자 간격
domain_half = (grid_size - 1) * spatial_step / 2 # 반절 크기

dt_diff = (0.2 * spatial_step**2) / d_coeff  #수치해석 dt. 안정적인게 알아서 계산하게 해둠

# 실제 분포
def run_simulation(n_p, t_f, dt, d):
    n_steps = int(t_f / dt)
    sigma = np.sqrt(2 * d * dt) 
    
    deltas = np.random.normal(loc=0, scale=sigma, size=(n_p, n_steps, 2))
    trajectories = np.cumsum(deltas, axis=1)
    
    final_positions = trajectories[:, -1, :]

    squared_distances = np.sum(trajectories**2, axis=2)
    msd = np.mean(squared_distances, axis=0)
    
    return final_positions, msd

# 이론상 분포
def run_numerical_analysis(t_f, d, size, step, dt):
    u = np.zeros((size, size))
    center = size // 2

    x = np.arange(size)
    y = np.arange(size)
    X, Y = np.meshgrid(x, y)
    
    # 정규분포로 안하면 laplacian 계산이 잘 안됨...
    init_sigma = 1.0 
    u = np.exp(-((X - center)**2 + (Y - center)**2) / (2 * init_sigma**2))
    u /= np.sum(u) * (step**2) 
    
    n_steps_num = int(t_f / dt)
    
    # 확산방정식 du/dt = D * laplacian(u) 를 유한차분법으로 풀음
    # runge-kuta 4차 쓰려고 했는데 계산이 너무 오래걸려서 오일러 방법으로 함
    # 오일러방법으로 풀면 a_(n+1) = a_n + dt * D * laplacian(a_n) 이렇게 됨

    for _ in range(n_steps_num):
        u_new = u.copy()
        
        laplacian = (u[2:, 1:-1] + u[:-2, 1:-1] + u[1:-1, 2:] + u[1:-1, :-2] - 4*u[1:-1, 1:-1]) / (step**2)
        
        u_new[1:-1, 1:-1] = u[1:-1, 1:-1] + d * dt * laplacian
        
        u = u_new
    
    return u

actual_positions, sim_msd = run_simulation(n_particles, t_final, dt_sim, d_coeff)

theoretical_u = run_numerical_analysis(t_final, d_coeff, grid_size, spatial_step, dt_diff)

fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharex=True, sharey=True)

hist, xedges, yedges = np.histogram2d(actual_positions[:, 1], actual_positions[:, 0],  bins=(grid_size, grid_size),  range=[[-domain_half, domain_half], [-domain_half, domain_half]],  density=True)

vmax_val = max(np.max(hist), np.max(theoretical_u)) * 1.1
norm = Normalize(vmin=0, vmax=vmax_val)

im0 = axes[0].imshow(hist, origin='lower', cmap='plasma', norm=norm, extent=[-domain_half, domain_half, -domain_half, domain_half])
axes[0].set_title(f"Actual Distribution (Simulation)\nN={n_particles}, t={t_final}, D={d_coeff}")
axes[0].set_xlabel('X')
axes[0].set_ylabel('Y')
fig.colorbar(im0, ax=axes[0], label='Probability Density')

im1 = axes[1].imshow(theoretical_u, origin='lower', cmap='plasma', norm=norm, extent=[-domain_half, domain_half, -domain_half, domain_half])
axes[1].set_title(f"Theoretical Distribution (Numerical FDM)\nGrid={grid_size}x{grid_size}, t={t_final}, D={d_coeff}")
axes[1].set_xlabel('X')
fig.colorbar(im1, ax=axes[1], label='Probability Density')

plt.tight_layout()
plt.show()

print(f"Simulation MSD at t={t_final}: {sim_msd[-1]:.4f} (Theoretical: 4Dt = {4 * d_coeff * t_final:.4f})")