import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.optimize import fsolve

SCENARIO = 2

G=1.0
dt=0.01

if SCENARIO == 1:
    # [스윙바이] 거대 항성, 공전하는 행성, 접근하는 탐사선
    masses = np.array([1000.0, 10.0, 1e-6]) # 항성, 행성, 탐사선
    
    r_planet = 10.0
    v_planet = np.sqrt(G * masses[0] / r_planet)
    
    positions = np.array([
        [0.0, 0.0],                # 항성 위치
        [r_planet, 0.0],           # 행성 위치
        [r_planet - 1.0, -10.0]    # 탐사선 위치 (행성의 공전 궤도 뒤쪽에서 접근)
    ])
    
    velocities = np.array([
        [0.0, 0.0],                
        [0.0, v_planet],           # 행성의 공전 속도
        [1.0, v_planet * 1.5]      # 탐사선의 초기 속도 (행성의 중력장으로 진입)
    ])
    colors = ["orange", "green", "red"]
    view_range = 15.0

elif SCENARIO == 2:
    # [라그랑주 점] 5개의 라그랑주 점
    
    # L4, L5의 안정성 조건(m1/m2 > 24.96)을 만족하기 위해 질량비 100:3 설정
    m1 = 100.0
    m2 = 3.0 
    
    r_distance = 10.0
    r1 = r_distance * (m2 / (m1 + m2)) # 질량 중심으로부터 m1까지의 거리
    r2 = r_distance * (m1 / (m1 + m2)) # 질량 중심으로부터 m2까지의 거리
    
    omega = np.sqrt(G * (m1 + m2) / (r_distance**3)) # 계의 공전 각속도
    
    # 1. L4, L5 계산 (해석적 해 - 정삼각형 배치)
    l4_x = -r1 + r_distance * 0.5
    l4_y = r_distance * np.sqrt(3) / 2
    l5_x = l4_x
    l5_y = -l4_y

    # 2. L1, L2, L3 계산 (수치해석적 해 - fsolve 활용)
    # 회전 좌표계에서의 원심력과 두 중력의 합이 0이 되는 지점(x)을 찾는 방정식
    def force_balance(x):
        centrifugal = omega**2 * x
        grav_m1 = -np.sign(x - (-r1)) * G * m1 / (x - (-r1))**2
        grav_m2 = -np.sign(x - r2) * G * m2 / (x - r2)**2
        return centrifugal + grav_m1 + grav_m2

    # L1, L2, L3가 있을 법한 위치를 초기 추정값(Guess)으로 제공
    hill_radius = r_distance * (m2 / (3 * m1))**(1/3) # 힐 구(Hill sphere) 근사
    l1_guess = r2 - hill_radius
    l2_guess = r2 + hill_radius
    l3_guess = -r1 - r_distance

    # fsolve 알고리즘을 통해 수치적으로 정확한 x좌표 산출
    l1_x = fsolve(force_balance, l1_guess)[0]
    l2_x = fsolve(force_balance, l2_guess)[0]
    l3_x = fsolve(force_balance, l3_guess)[0]

    # 총 7개의 천체: m1, m2, L1, L2, L3, L4, L5
    masses = np.array([m1, m2, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6])
    
    positions = np.array([
        [-r1, 0.0],
        [r2, 0.0],
        [l1_x, 0.0], [l2_x, 0.0], [l3_x, 0.0], # L1, L2, L3
        [l4_x, l4_y], [l5_x, l5_y]             # L4, L5
    ])
    
    # 모든 탐사선의 속도는 회전 좌표계에 정지해 있으므로 v = r * w 적용
    velocities = np.array([
        [0.0, -r1 * omega],
        [0.0, r2 * omega],
        [0.0, l1_x * omega], 
        [0.0, l2_x * omega], 
        [0.0, l3_x * omega],
        [-l4_y * omega, l4_x * omega], 
        [-l5_y * omega, l5_x * omega]
    ])
    
    # 색상 부여: L1, L2, L3(불안정)는 빨간색 / L4, L5(안정)는 파란색
    colors = ["orange", "cyan", "red", "red", "red", "blue", "blue"]
    view_range = 15.0

elif SCENARIO == 3:
    # [카오스] 기존에 주석 처리해두신 8자 궤도 데이터
    masses = np.array([1.0, 1.0, 1.0])
    positions = np.array([
        [0.97000436, -0.24308753],
        [-0.97000436, 0.24308753],
        [0.0, 0.0]
    ])
    velocities = np.array([
        [0.46620368 , 0.43236573],
        [0.46620368, 0.43236573],
        [-0.93240737, -0.86473146]
    ])
    colors = ["red", "green", "blue"]
    view_range = 2.0

# state 배열 형태: [위치_배열, 속도_배열] -> shape: (2, N_bodies, 2)
state = np.array([positions, velocities])
N_bodies = len(masses)

# --- 수치해석 (NumPy Vectorized) ---
def get_accelerations(pos):
    # 만유인력 법칙을 통해 모든 천체의 가속도를 계산
    acc = np.zeros((N_bodies, 2))
    for i in range(N_bodies):
        for j in range(N_bodies):
            if i != j:
                r_vec = pos[j] - pos[i]
                r_mag = np.linalg.norm(r_vec)
                # 충돌 시 0으로 나누는 오류(Singularity) 방지를 위한 최소 거리 설정
                if r_mag > 0.01:
                    acc[i] += G * masses[j] * r_vec / (r_mag**3)
    return acc

def derivatives(curr_state):
    pos = curr_state[0]
    vel = curr_state[1]
    acc = get_accelerations(pos)
    return np.array([vel, acc]) # 상태 변화량(미분값) 반환

def rk4_step(curr_state, time_step):
    # 모든 천체의 상태를 동시에 계산하는 RK4
    k1 = derivatives(curr_state)
    k2 = derivatives(curr_state + 0.5 * time_step * k1)
    k3 = derivatives(curr_state + 0.5 * time_step * k2)
    k4 = derivatives(curr_state + time_step * k3)
    return curr_state + (time_step / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

# --- 시각화 및 애니메이션 ---
tracking_body_index = -1

def on_key_press(event):
    global tracking_body_index, view_range
    if event.key.isdigit():
        idx = int(event.key)
        if idx < N_bodies:
            tracking_body_index = idx
    elif event.key == "c":
        tracking_body_index = -1 
    if event.key == "[":
        view_range *= 1.5
    if event.key == "]":
        view_range /= 1.5

fig, axplt = plt.subplots(figsize=(8, 8))
fig.canvas.mpl_connect('key_press_event', on_key_press)
axplt.set_facecolor('black')
axplt.set_xlabel("x (m)")
axplt.set_ylabel("y (m)")

titles = ["1. Gravitational Slingshot", "2. Lagrange Point", "3. Chaotic Figure-8 Orbit"]
axplt.set_title(f"{titles[SCENARIO-1]}")
axplt.set_aspect("equal")

body_markers = []
body_trackers = []
trail_length = 200
pos_history_x = np.zeros((N_bodies, trail_length))
pos_history_y = np.zeros((N_bodies, trail_length))

for i in range(N_bodies):
    marker, = axplt.plot([], [], "o", markersize=8 if i == 0 else 5, color=colors[i])
    tracker, = axplt.plot([], [], lw=1.2, color=colors[i], alpha=0.6)
    body_markers.append(marker)
    body_trackers.append(tracker)
    pos_history_x[i, :] = state[0, i, 0]
    pos_history_y[i, :] = state[0, i, 1]

def update_axes_limits():
    if 0 <= tracking_body_index < N_bodies:
        center_x = state[0, tracking_body_index, 0]
        center_y = state[0, tracking_body_index, 1]
    else: 
        center_x, center_y = 0.0, 0.0
        
    axplt.set_xlim(center_x - view_range, center_x + view_range)
    axplt.set_ylim(center_y - view_range, center_y + view_range)

def init():
    update_axes_limits()
    return tuple(body_markers + body_trackers)

def update(frame):
    global state
    
    # 궤적이 그려지는 속도를 높이기 위해 프레임당 시뮬레이션을 5회씩 연산
    steps_per_frame = 5
    for _ in range(steps_per_frame):
        state = rk4_step(state, dt)
        
    update_axes_limits()
    
    for i in range(N_bodies):
        body_markers[i].set_data([state[0, i, 0]], [state[0, i, 1]])
        
        pos_history_x[i] = np.roll(pos_history_x[i], -1)
        pos_history_y[i] = np.roll(pos_history_y[i], -1)
        pos_history_x[i, -1] = state[0, i, 0]
        pos_history_y[i, -1] = state[0, i, 1]
        body_trackers[i].set_data(pos_history_x[i], pos_history_y[i])
        
    return tuple(body_markers + body_trackers)

ani = animation.FuncAnimation(
    fig, update, init_func=init, interval=15, blit=True, repeat=False
)

plt.show()