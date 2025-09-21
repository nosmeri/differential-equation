import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

c = .5
Lx, Ly = 2.0, 2.0
nx, ny = 71, 71 
x = np.linspace(0, Lx, nx)
y = np.linspace(0, Ly, ny)
dx = x[1] - x[0]
dy = y[1] - y[0]

dt = 0.01

z = np.zeros((ny, nx), dtype=float)
v = np.zeros_like(z) 

X, Y = np.meshgrid(x, y)
x0, y0, sigma = Lx * 0.5, Ly * 0.5, 0.18
z[:] = np.exp(-((X - x0) ** 2 + (Y - y0) ** 2) / (2 * sigma**2))
v[:] = 0.0


def laplacian(Z):
    L = np.zeros_like(Z)
    L[1:-1, 1:-1] = (Z[2:, 1:-1] - 2 * Z[1:-1, 1:-1] + Z[:-2, 1:-1]) / dy**2 + (
        Z[1:-1, 2:] - 2 * Z[1:-1, 1:-1] + Z[1:-1, :-2]
    ) / dx**2
    return L


def apply_bc(Z, V):
    Z[0, :] = Z[-1, :] = 0.0
    Z[:, 0] = Z[:, -1] = 0.0
    V[0, :] = V[-1, :] = 0.0
    V[:, 0] = V[:, -1] = 0.0


fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.set(
    xlim=(0, Lx),
    ylim=(0, Ly),
    zlim=(-1.0, 1.0),
    xlabel="x",
    ylabel="y",
    zlabel="z",
    title="3D Wave",
)

# 초기 surface
surf = ax.plot_surface(
    X,
    Y,
    z,
    cmap="RdBu_r",
    rstride=1,
    cstride=1,
    linewidth=0,
    antialiased=True,
    vmin=-np.max(np.abs(z)),
    vmax=np.max(np.abs(z)),
)


def update(frame):
    global z, v, surf
    # a = c^2 * ∇^2 z
    a = c**2 * laplacian(z)
    v += a * dt
    z += v * dt
    apply_bc(z, v)

    surf.remove()
    surf = ax.plot_surface(
        X,
        Y,
        z,
        cmap="RdBu_r",
        rstride=1,
        cstride=1,
        linewidth=0,
        antialiased=True,
        vmin=-np.max(np.abs(z)),
        vmax=np.max(np.abs(z)),
    )
    return (surf,)


ani = FuncAnimation(fig, update, frames=int(60/dt), interval=dt*1000, blit=False)
ani.save("wave.mp4", writer="ffmpeg", fps=int(1/dt))