import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# 파라미터
m, k, g = 0.15, 0.2, 9.81  # 질량, 선형저항계수, 중력
v0, angle = 20.0, 60  # 초기속도(m/s), 발사각(도)
dt, t_max = 0.01, 3.0

# 초기 속도 성분
theta = np.deg2rad(angle)
vx0, vy0 = v0 * np.cos(theta), v0 * np.sin(theta)

# 배열
t = np.arange(0, t_max + dt, dt)
x = np.zeros_like(t)
y = np.zeros_like(t)
vx = np.zeros_like(t)
vy = np.zeros_like(t)

# 초기조건
vx[0], vy[0] = vx0, vy0
x[0], y[0] = 0.0, 0.0

# 수치해석 (Euler)
for i in range(len(t) - 1):
    ax = -(k / m) * vx[i]
    ay = -g - (k / m) * vy[i]
    vx[i + 1] = vx[i] + ax * dt
    vy[i + 1] = vy[i] + ay * dt
    x[i + 1] = x[i] + vx[i + 1] * dt
    y[i + 1] = y[i] + vy[i + 1] * dt
    if y[i + 1] < 0:  # 땅에 닿으면 중단
        x = x[: i + 2]
        y = y[: i + 2]
        break

# 애니메이션 준비
fig, axplt = plt.subplots()
axplt.set_xlim(0, 10)
axplt.set_ylim(0, 10)
axplt.set_xlabel("x (m)")
axplt.set_ylabel("y (m)")
axplt.set_title("Projectile motion with linear drag")
axplt.set_aspect("equal")

(traj_line,) = axplt.plot([], [], "-", lw=1)
(ball,) = axplt.plot([], [], "o", markersize=10)


def init():
    traj_line.set_data([], [])
    ball.set_data([], [])
    return traj_line, ball


def update(frame):
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
