import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import threading
import time
from collections import deque

# --- 물리 설정 ---
k = 3.0  
charge_q = 1.0
c = 5.0  
damping = 0.95
acc_smooth_factor = 0.15  # 가속도가 변하는 부드러움 정도 (0~1)

grid_range = 10
grid_density = 0.8
X, Y = np.meshgrid(np.arange(-grid_range, grid_range + grid_density, grid_density),
                np.arange(-grid_range, grid_range + grid_density, grid_density))
Z_plane = np.zeros_like(X)

# 상태 변수
charge_pos = np.array([0.0, 0.0, 0.0])
charge_vel = np.array([0.0, 0.0, 0.0])
target_acc = np.array([0.0, 0.0, 0.0])   # 키보드 입력 값
smooth_acc = np.array([0.0, 0.0, 0.0])   # 필터링된 실제 가속도

history = deque(maxlen=400) 
dt = 0.01
running = True

def physics_engine():
    global charge_pos, charge_vel, smooth_acc, running
    while running:
        # 가속도 평활화 (가속도 자체에 관성을 부여)
        smooth_acc += (target_acc - smooth_acc) * acc_smooth_factor
        
        charge_vel += smooth_acc * dt
        charge_vel *= damping
        charge_pos += charge_vel * dt
        
        history.append((time.time(), charge_pos.copy(), smooth_acc.copy()))
        time.sleep(dt)

def get_retarded_data(obs_x, obs_y, obs_z, current_time):
    # 각 점으로부터의 거리를 행렬로 한꺼번에 계산
    dx = obs_x - charge_pos[0]
    dy = obs_y - charge_pos[1]
    dz = obs_z - charge_pos[2]
    dist = np.sqrt(dx**2 + dy**2 + dz**2)
    
    # 지연 시간 행렬 (dt_matrix)
    dt_matrix = dist / c
    ret_times = current_time - dt_matrix
    
    # 히스토리에서 가장 가까운 시점 찾기 (벡터화하기 까다로우므로 근사 혹은 최근값 사용)
    # 여기서는 각 점의 지연 상태를 효율적으로 뽑기 위해 간단한 조회 사용
    if not history:
        return np.zeros_like(obs_x), np.zeros_like(obs_x), np.zeros_like(obs_x), \
            np.zeros_like(obs_x), np.zeros_like(obs_x), np.zeros_like(obs_x)

    # 행렬 크기에 맞는 빈 결과 생성
    Ex, Ey, Ez = np.zeros_like(obs_x), np.zeros_like(obs_x), np.zeros_like(obs_x)
    
    # 히스토리 배열화 (조회 성능 향상)
    h_times = np.array([h[0] for h in history])
    
    # 각 그리드 점에 대해 (최적화를 위해 이 부분은 루프를 돌되 내부 연산은 벡터화)
    for i in range(obs_x.shape[0]):
        for j in range(obs_x.shape[1]):
            # 최적의 과거 인덱스 찾기
            idx = np.abs(h_times - ret_times[i, j]).argmin()
            r_pos, r_acc = history[idx][1], history[idx][2]
            
            rx, ry, rz = obs_x[i,j]-r_pos[0], obs_y[i,j]-r_pos[1], obs_z[i,j]-r_pos[2]
            R = np.sqrt(rx**2 + ry**2 + rz**2)
            R = max(R, 0.5)
            nx, ny, nz = rx/R, ry/R, rz/R
            
            # 속도 항 + 복사 항 계산
            Ev = k * charge_q / (R**2)
            dot_an = r_acc[0]*nx + r_acc[1]*ny + r_acc[2]*nz
            Er_c = -k * charge_q / (c**2 * R)
            
            Ex[i,j] = Ev*nx + Er_c*(r_acc[0] - dot_an*nx)
            Ey[i,j] = Ev*ny + Er_c*(r_acc[1] - dot_an*ny)
            Ez[i,j] = Ev*nz + Er_c*(r_acc[2] - dot_an*nz)
            
    return Ex, Ey, Ez

# --- 시각화 ---
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
charge_marker, = ax.plot([0], [0], [0], 'ro', markersize=10)
quiver_arrows = None

def update(frame):
    global quiver_arrows
    charge_marker.set_data_3d([charge_pos[0]], [charge_pos[1]], [charge_pos[2]])
    Ex, Ey, Ez = get_retarded_data(X, Y, Z_plane, time.time())
    
    if quiver_arrows: quiver_arrows.remove()
    quiver_arrows = ax.quiver(X, Y, Z_plane, Ex, Ey, Ez, 
                            length=0.7, normalize=True, color='blue', alpha=0.5)
    return charge_marker, quiver_arrows

def on_press(event):
    val = 20.0
    if event.key == 'up':    target_acc[1] = val
    elif event.key == 'down':  target_acc[1] = -val
    elif event.key == 'left':  target_acc[0] = -val
    elif event.key == 'right': target_acc[0] = val
    elif event.key == 'a':     target_acc[2] = val
    elif event.key == 'z':     target_acc[2] = -val

def on_release(event):
    if event.key in ['up', 'down', 'left', 'right', 'q', 'a']:
        target_acc[:] = 0

fig.canvas.mpl_connect('key_press_event', on_press)
fig.canvas.mpl_connect('key_release_event', on_release)
threading.Thread(target=physics_engine, daemon=True).start()
ani = FuncAnimation(fig, update, interval=30, cache_frame_data=False)
ax.set_xlim(-grid_range, grid_range); ax.set_ylim(-grid_range, grid_range); ax.set_zlim(-grid_range, grid_range)
plt.show()
running = False