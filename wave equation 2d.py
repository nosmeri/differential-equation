import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# 파라미터
v=2
dx, x_range=0.01, 5.0
dt = 0.001

# 배열
x = np.arange(-x_range, x_range + dx, dx)
y = np.zeros_like(x)

vy = np.zeros_like(x)

# 초기조건
y = np.exp(-(x*5) ** 2)

# 애니메이션 준비
fig, axplt = plt.subplots()
axplt.set_xlim(-x_range, x_range)
axplt.set_ylim(-1.2, 1.2)
axplt.set_xlabel("x (m)")
axplt.set_ylabel("y (m)")
axplt.set_title("Wave equation")

(line,) = axplt.plot([], [], "-", lw=1.5)


def init():
    line.set_data([], [])
    return line,


def update(frame):
    global y, vy, x
    ddy=np.zeros_like(x)
    ddy[1:-1]=(y[:-2] - 2 * y[1:-1] + y[2:])/dx**2
    ay=(v**2) * ddy
    vy+=ay*dt
    y+=vy*dt
    line.set_data(x, y)
    return line, 


ani = animation.FuncAnimation(
    fig,
    update,
    init_func=init,
    interval=int(1000 * dt),
    blit=True,
    repeat=False,
)
plt.show()
