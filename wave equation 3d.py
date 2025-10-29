import argparse


parser = argparse.ArgumentParser(description="3D Wave Simulation")
parser.add_argument("--Lx", type=float, default=2.0, help="X 길이")
parser.add_argument("--Ly", type=float, default=2.0, help="Y 길이")
parser.add_argument("--nx", type=int, default=201, help="X 격자 개수")
parser.add_argument("--ny", type=int, default=201, help="Y 격자 개수")
parser.add_argument("--dt", type=float, default=0.002, help="시간 간격 Δt")
parser.add_argument("--c", type=float, default=0.5, help="파동 속도")
parser.add_argument("--sigma", type=float, default=0.18, help="초기 가우시안 폭")
parser.add_argument("--t", type=int, default=60, help="렌더링 길이")
args = parser.parse_args()


from cnpy import xp, to_cpu, on_gpu
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

c = args.c
Lx, Ly = args.Lx, args.Ly
nx, ny = args.nx, args.ny 
x = xp.linspace(0, Lx, nx, dtype=xp.float32)
y = xp.linspace(0, Ly, ny, dtype=xp.float32)
dx = float(x[1] - x[0])
dy = float(y[1] - y[0])
t=args.t
sigma=args.sigma

dt = args.dt
frames = int(t / dt)

z = xp.zeros((ny, nx), dtype=xp.float32)
v = xp.zeros_like(z) 

X_cpu, Y_cpu = np.meshgrid(np.linspace(0, Lx, nx), np.linspace(0, Ly, ny))

x0, y0 = Lx * 0.5, Ly * 0.5
Xb = xp.asarray(X_cpu) if on_gpu else X_cpu
Yb = xp.asarray(Y_cpu) if on_gpu else Y_cpu
z[:] = xp.exp(-((Xb - x0) ** 2 + (Yb - y0) ** 2) / (2 * sigma**2))


def laplacian(Z):
    L = xp.zeros_like(Z)
    L[1:-1, 1:-1] = (Z[2:, 1:-1] - 2 * Z[1:-1, 1:-1] + Z[:-2, 1:-1]) / dy**2 + (
        Z[1:-1, 2:] - 2 * Z[1:-1, 1:-1] + Z[1:-1, :-2]
    ) / dx**2
    return L


def apply_bc(Z, V):
    Z[0, :] = Z[-1, :] = 0.0
    Z[:, 0] = Z[:, -1] = 0.0
    V[0, :] = V[-1, :] = 0.0
    V[:, 0] = V[:, -1] = 0.0


amp0 = float((xp.max(xp.abs(z))).item() if on_gpu else np.max(np.abs(z)))

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.set(
    xlim=(0, Lx),
    ylim=(0, Ly),
    zlim=(-amp0, amp0),
    xlabel="x",
    ylabel="y",
    zlabel="z",
    title="3D Wave",
)

# 초기 surface
z0_cpu = to_cpu(z)
surf = ax.plot_surface(
    X_cpu,
    Y_cpu,
    z0_cpu,
    cmap="RdBu_r",
    rstride=1,
    cstride=1,
    linewidth=0,
    antialiased=True,
    vmin=-amp0,
    vmax=amp0,
)


def update(frame):
    print(f"{frame}/{frames}")
    global z, v, surf
    # a = c^2 * ∇^2 z
    a = c**2 * laplacian(z)
    v += a * dt
    z += v * dt
    apply_bc(z, v)

    z_cpu=to_cpu(z)

    surf.remove()
    surf = ax.plot_surface(
        X_cpu,
        Y_cpu,
        z_cpu,
        cmap="RdBu_r",
        rstride=1,
        cstride=1,
        linewidth=0,
        antialiased=True,
        vmin=-amp0,
        vmax=amp0,
    )
    return (surf,)


ani = FuncAnimation(fig, update, frames=frames, interval=dt*1000, blit=False)
ani.save("wave.mp4", writer="ffmpeg", fps=int(1/dt))
