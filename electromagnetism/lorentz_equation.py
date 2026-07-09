import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

dt=1e-2
sigma=10
rho=28
beta=8/3

line_length=1000

x=.6
y=-.4
z=0

x = np.full(line_length, x)
y = np.full(line_length, y)
z = np.full(line_length, z)

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.set(
    xlim=(-30, 30),
    ylim=(-30, 30),
    zlim=(-30, 30),
    xlabel="x",
    ylabel="y",
    zlabel="z",
    title="lorentz",
)

(dot,) = ax.plot([], [], [], "o", markersize=10)
(traj_line,) = ax.plot([], [], [], lw=1)


def update(frame):
    global x, y, z
    x[1:] = x[:-1]
    y[1:] = y[:-1]
    z[1:] = z[:-1]
    x_dot = sigma * (y[0] - x[0])
    y_dot = x[0] * (rho - z[0]) - y[0]
    z_dot = x[0] * y[0] - beta * z[0]
    x[0] += x_dot * dt
    y[0] += y_dot * dt
    z[0] += z_dot * dt
    dot.set_data([x[0]],[y[0]])
    dot.set_3d_properties([z[0]])
    traj_line.set_data(x,y)
    traj_line.set_3d_properties(z)
    return dot

ani = FuncAnimation(fig, update, frames=int(60 / dt), interval=1e-3, blit=False)

plt.show()
