import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# --- 물리적 매개변수 설정 ---
radius = 5.0        # 원운동 반지름
omega = 1.0         # 각속도
speed_light = 10.0  # 빛의 속도
num_pulses = 60     # 누적할 과거 파동의 수

# --- 격자 설정 ---
grid_size = 20.0
grid_points = 150   # 해상도
x = np.linspace(-grid_size, grid_size, grid_points)
y = np.linspace(-grid_size, grid_size, grid_points)
X, Y = np.meshgrid(x, y)

# 3차원 벡터 연산을 위한 관찰자 좌표 (Vectorization)
R_obs = np.stack((X, Y, np.zeros_like(X)), axis=-1)

# --- 시각화 초기화 ---
fig, ax = plt.subplots(figsize=(10, 8))
ax.set_aspect('equal')
ax.set_facecolor('black') 
fig.patch.set_facecolor('#2b2b2b') # 창 배경색

ax.set_title("Radiation Field of a Charge in Circular Motion", color='white', fontsize=14)
ax.set_xlabel("x", color='white')
ax.set_ylabel("y", color='white')
ax.tick_params(colors='white')

# 궤적 표시
trajectory_theta = np.linspace(0, 2*np.pi, 100)
ax.plot(radius * np.cos(trajectory_theta), radius * np.sin(trajectory_theta), 
        'w--', alpha=0.3)

# 전하 표시
charge_plot = ax.scatter([], [], color='#00CCFF', s=100, zorder=5)

# --- [핵심] imshow를 이용한 빠르고 부드러운 렌더링 ---
extent = [-grid_size, grid_size, -grid_size, grid_size]
# vmin, vmax를 고정하여 프레임이 넘어가도 색상 척도가 변하지 않게(깜빡이지 않게) 설정합니다.
im = ax.imshow(np.zeros((grid_points, grid_points)), origin='lower', extent=extent, 
               cmap='magma', vmin=0, vmax=0.4, animated=True, zorder=1)

fig.colorbar(im, label='Field Intensity')

# --- 수치 계산 함수 ---
def calculate_field_intensity(t):
    Z = np.zeros((grid_points, grid_points))
    
    for i in range(num_pulses):
        # 과거 지연 시간
        t_past = t - i * (2 * np.pi / omega / num_pulses) * 1.0 
        
        # 전하의 상태 벡터 (위치, 속도, 가속도)
        r_q = np.array([radius * np.cos(omega * t_past), radius * np.sin(omega * t_past), 0.0]) 
        v_q = np.array([-radius * omega * np.sin(omega * t_past), radius * omega * np.cos(omega * t_past), 0.0]) 
        a_q = np.array([-radius * omega**2 * np.cos(omega * t_past), -radius * omega**2 * np.sin(omega * t_past), 0.0]) 
        
        beta_q = v_q / speed_light        
        
        # 격자 전체 동시 계산
        R_vec = R_obs - r_q
        R_mag = np.linalg.norm(R_vec, axis=-1)
        
        # 전하 위치(특이점) 근처의 무한대 발산 방지
        mask = R_mag > 0.1
        R_mag_safe = np.where(mask, R_mag, 1.0) 
        
        R_hat = R_vec / R_mag_safe[..., np.newaxis]
        
        # 리에나르-비헤르트 포텐셜 복사항(가속도 의존성) 수치 계산
        R_hat_dot_beta = np.sum(R_hat * beta_q, axis=-1)
        denom = (1 - R_hat_dot_beta)**3
        
        inner_cross = np.cross(R_hat - beta_q, a_q)
        outer_cross = np.cross(R_hat, inner_cross)
        num_mag = np.linalg.norm(outer_cross, axis=-1)
        
        intensity = np.zeros_like(Z)
        intensity[mask] = (num_mag[mask] / (speed_light**2 * R_mag_safe[mask] * denom[mask])) * np.exp(-i / 10)
        
        Z += np.abs(intensity)

    # 이전 코드에 있던 Z / np.max(Z) (정규화) 코드를 제거하여 깜빡임을 방지합니다.
    return Z

# --- 애니메이션 업데이트 함수 ---
def update(frame):
    t = frame * 0.05
    
    # 전하 위치 갱신
    x_q = radius * np.cos(omega * t)
    y_q = radius * np.sin(omega * t)
    charge_plot.set_offsets(np.array([[x_q, y_q]]))
    
    # 필드 계산
    Z = calculate_field_intensity(t)
    
    # 그림 전체를 새로 그리지 않고 데이터(행렬) 값만 덮어씌워 렌더링 최적화
    im.set_array(Z)
    
    return charge_plot, im

# blit=True를 사용하여 변경된 픽셀만 다시 그리도록 하여 속도를 극대화합니다.
ani = FuncAnimation(fig, update, frames=200, interval=40, blit=True)

plt.tight_layout()
plt.show()