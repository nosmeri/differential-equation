import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

m, k, g = 0.15, 0.2, 9.81  # 질량, 선형저항계수, 중력
v0, angle = 20.0, 60  # 초기속도(m/s), 발사각(도)
dt, t_max = 0.01, 3.0

theta = np.deg2rad(angle)
vx0, vy0 = v0 * np.cos(theta), v0 * np.sin(theta)

t_list = [0.0]
x_list = [0.0]
y_list = [0.0]
vx_list = [vx0]
vy_list = [vy0]

current_state = np.array([0.0, 0.0, vx0, vy0])

def derivatives(state, t):
    """
    상태 벡터 S에 대한 dS/dt를 반환
    state: [x, y, vx, vy]
    return: [vx, vy, ax, ay]
    """
    x, y, vx, vy = state
    
    ax = -(k / m) * vx
    ay = -g - (k / m) * vy
    
    return np.array([vx, vy, ax, ay])

def rk4_step(state, t, dt):
    """
    Runge-Kutta 4th Order Integration Step
    """
    k1 = derivatives(state, t)
    k2 = derivatives(state + 0.5 * dt * k1, t + 0.5 * dt)
    k3 = derivatives(state + 0.5 * dt * k2, t + 0.5 * dt)
    k4 = derivatives(state + dt * k3, t + dt)
    
    return state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

curr_t = 0.0
while curr_t < t_max:
    current_state = rk4_step(current_state, curr_t, dt)
    curr_t += dt
    
    rx, ry, rvx, rvy = current_state
    
    if ry < 0:
        break
    
    # 데이터 저장
    t_list.append(curr_t)
    x_list.append(rx)
    y_list.append(ry)
    vx_list.append(rvx)
    vy_list.append(rvy)

x = np.array(x_list)
y = np.array(y_list)

fig, axplt = plt.subplots()
axplt.set_xlim(0, max(x) * 1.1) 
axplt.set_ylim(0, max(y) * 1.2) 
axplt.set_xlabel("x (m)")
axplt.set_ylabel("y (m)")
axplt.set_title("Projectile motion")
axplt.set_aspect("equal")
axplt.grid(True)

(traj_line,) = axplt.plot([], [], "-", lw=1)
(ball,) = axplt.plot([], [], "o", markersize=10)

def init():
    traj_line.set_data([], [])
    ball.set_data([], [])
    return traj_line, ball

def update(frame):
    if frame >= len(x):
        return traj_line, ball
        
    traj_line.set_data(x[:frame], y[:frame])
    ball.set_data([x[frame]], [y[frame]])
    return traj_line, ball

ani = animation.FuncAnimation(
    fig,
    update,
    frames=len(x),
    init_func=init,
    interval=int(1000 * dt),
    blit=True,
    repeat=False,
)

plt.show()