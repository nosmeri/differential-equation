import argparse

parser = argparse.ArgumentParser(
    description="Wave Simulation (3D surface / 2D colormap)"
)
parser.add_argument("--Lx", type=float, default=2.0, help="X 길이")
parser.add_argument("--Ly", type=float, default=2.0, help="Y 길이")
parser.add_argument("--nx", type=int, default=201, help="X 격자 개수")
parser.add_argument("--ny", type=int, default=201, help="Y 격자 개수")
parser.add_argument("--dt", type=float, default=0.002, help="시간 간격 Δt")
parser.add_argument("--c", type=float, default=0.5, help="파동 속도")
parser.add_argument("--sigma", type=float, default=0.18, help="초기 가우시안 폭")
parser.add_argument("--t", type=float, default=60, help="물리 시뮬 길이(초)")
parser.add_argument(
    "--mode", choices=["3d", "2d"], default="3d", help="시각화 모드 선택"
)
parser.add_argument(
    "--substeps", type=int, default=1, help="렌더 1프레임당 타임스텝 수(2D는 5~10 권장)"
)
parser.add_argument("--fps", type=int, default=60, help="프레임레이트(미리보기/저장)")
parser.add_argument(
    "--ds", type=int, default=1, help="3D 표시 다운샘플 (1=원본, 2/3/4… 권장)"
)
parser.add_argument(
    "--outfile", type=str, default=None, help="저장 파일명 (기본: mode에 따라 자동)"
)
args = parser.parse_args()

from cnpy import xp, to_cpu, on_gpu
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import math

# ---------------- 공통 파라미터/필드 ----------------
c = args.c
Lx, Ly = args.Lx, args.Ly
nx, ny = args.nx, args.ny
dt = args.dt
sigma = args.sigma
substeps = max(1, args.substeps)
fps = args.fps
ds = max(1, args.ds)

x = xp.linspace(0, Lx, nx, dtype=xp.float32)
y = xp.linspace(0, Ly, ny, dtype=xp.float32)
dx = float(x[1] - x[0])
dy = float(y[1] - y[0])

# 필드
z = xp.zeros((ny, nx), dtype=xp.float32)
v = xp.zeros_like(z)

# CPU mesh (렌더링 extent/좌표용)
X_cpu, Y_cpu = np.meshgrid(np.linspace(0, Lx, nx), np.linspace(0, Ly, ny))

# 초기 조건: 가우시안
x0, y0 = Lx * 0.5, Ly * 0.5
Xb = xp.asarray(X_cpu) if on_gpu else X_cpu
Yb = xp.asarray(Y_cpu) if on_gpu else Y_cpu
z[:] = xp.exp(-((Xb - x0) ** 2 + (Yb - y0) ** 2) / (2 * sigma**2))

# 초기 진폭(컬러/스케일 고정용)
amp0 = float((xp.max(xp.abs(z))).item() if on_gpu else np.max(np.abs(z)))
amp0 = max(amp0, 1e-6)  # 0 방지

# 프레임 수: 물리시간 t를 substeps*dt 간격으로 렌더링
frames = int(args.t / (substeps * dt))
frames = max(frames, 1)

# CFL 경고 (참고)
cfl = c * dt * math.sqrt(1 / dx**2 + 1 / dy**2)
if cfl > 1.0:
    print(
        f"[WARN] CFL={cfl:.3f} > 1.0 (불안정할 수 있음). dt를 줄이거나 격자를 조정하세요."
    )


def laplacian(Z):
    L = xp.zeros_like(Z)
    L[1:-1, 1:-1] = (Z[2:, 1:-1] - 2 * Z[1:-1, 1:-1] + Z[:-2, 1:-1]) / (dy * dy) + (
        Z[1:-1, 2:] - 2 * Z[1:-1, 1:-1] + Z[1:-1, :-2]
    ) / (dx * dx)
    return L


def apply_bc(Z, V):
    Z[0, :] = Z[-1, :] = 0.0
    Z[:, 0] = Z[:, -1] = 0.0
    V[0, :] = V[-1, :] = 0.0
    V[:, 0] = V[:, -1] = 0.0


def step():
    global z, v
    a = (c * c) * laplacian(z)
    v += a * dt
    z += v * dt
    apply_bc(z, v)


# ---------------- 시각화 분기 ----------------
if args.mode == "2d":
    # 2D imshow (빠름)
    fig, ax = plt.subplots()
    im = ax.imshow(
        to_cpu(z),
        origin="lower",
        extent=[0, Lx, 0, Ly],
        cmap="RdBu_r",
        vmin=-amp0,
        vmax=amp0,
        animated=True,
    )
    ax.set(
        xlabel="x",
        ylabel="y",
        title=f"2D Wave (imshow, {'GPU' if on_gpu else 'CPU'})",
        aspect="equal",
    )
    fig.colorbar(im, ax=ax, label="z")

    def update(_):
        for _ in range(substeps):
            step()
        im.set_data(to_cpu(z))
        return (im,)

    ani = FuncAnimation(fig, update, frames=frames, interval=1000 / fps, blit=True)
    out = args.outfile or "wave_2d.mp4"
    ani.save(out, writer="ffmpeg", fps=fps)

else:
    # 3D surface (느림: ds로 다운샘플 권장)
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.set(
        xlim=(0, Lx),
        ylim=(0, Ly),
        zlim=(-amp0, amp0),
        xlabel="x",
        ylabel="y",
        zlabel="z",
        title=f"3D Wave (surface, ds={ds}, {'GPU' if on_gpu else 'CPU'})",
    )

    Xds = X_cpu[::ds, ::ds]
    Yds = Y_cpu[::ds, ::ds]
    z0 = to_cpu(z)[::ds, ::ds]
    surf = ax.plot_surface(
        Xds,
        Yds,
        z0,
        cmap="RdBu_r",
        rstride=1,
        cstride=1,
        linewidth=0,
        antialiased=False,
        vmin=-amp0,
        vmax=amp0,
    )

    def update(frame):
        global surf
        for _ in range(substeps):
            step()
        zcpu_ds = to_cpu(z)[::ds, ::ds]
        surf.remove()
        surf = ax.plot_surface(
            Xds,
            Yds,
            zcpu_ds,
            cmap="RdBu_r",
            rstride=1,
            cstride=1,
            linewidth=0,
            antialiased=False,
            vmin=-amp0,
            vmax=amp0,
        )
        return (surf,)

    ani = FuncAnimation(fig, update, frames=frames, interval=1000 / fps, blit=False)
    out = args.outfile or "wave_3d.mp4"
    ani.save(out, writer="ffmpeg", fps=fps)
