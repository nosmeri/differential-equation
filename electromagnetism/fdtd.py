import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

size = 200
c = 3e8
dx = 0.01
dt = dx / (c * np.sqrt(3))

eps0 = 8.854e-12
mu0 = 1.256e-6

current_z = size // 2

Ex = np.zeros((size, size, size))
Ey = np.zeros((size, size, size))
Ez = np.zeros((size, size, size))
Hx = np.zeros((size, size, size))
Hy = np.zeros((size, size, size))
Hz = np.zeros((size, size, size))

pml_len = 10
sigma = np.zeros(size)
for i in range(pml_len):
    sigma[pml_len - i - 1] = 0.5 * ((i+1)/pml_len)**3
    sigma[size - pml_len + i] = 0.5 * ((i+1)/pml_len)**3

fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(Ez[:, :, current_z], cmap='RdBu', vmin=-0.01, vmax=0.01, 
               extent=[0, size*dx, 0, size*dx], animated=True)
plt.colorbar(im, label='Ez Field Intensity')
info_text = ax.text(0.02, 0.95, f"Z-Slice: {current_z}", transform=ax.transAxes, 
                     color='black', fontsize=10, fontweight='bold')

def on_press(event):
    global current_z
    if event.key == 'up':
        current_z = min(size - 1, current_z + 1)
    elif event.key == 'down':
        current_z = max(0, current_z - 1)

fig.canvas.mpl_connect('key_press_event', on_press)

def update(frame):
    global Ex, Ey, Ez, Hx, Hy, Hz

    Hx[:, :-1, :-1] -= (dt/mu0) * ((Ez[:, 1:, :-1] - Ez[:, :-1, :-1])/dx - 
                                   (Ey[:, :-1, 1:] - Ey[:, :-1, :-1])/dx)
    Hy[:-1, :, :-1] -= (dt/mu0) * ((Ex[:-1, :, 1:] - Ex[:-1, :, :-1])/dx - 
                                   (Ez[1:, :, :-1] - Ez[:-1, :, :-1])/dx)
    Hz[:-1, :-1, :] -= (dt/mu0) * ((Ey[1:, :-1, :] - Ey[:-1, :-1, :])/dx - 
                                   (Ex[:-1, 1:, :] - Ex[:-1, :-1, :])/dx)

    Ex[1:, 1:, 1:] += (dt/eps0) * ((Hz[1:, 1:, 1:] - Hz[1:, :-1, 1:])/dx - 
                                   (Hy[1:, 1:, 1:] - Hy[1:, 1:, :-1])/dx)
    Ey[1:, 1:, 1:] += (dt/eps0) * ((Hx[1:, 1:, 1:] - Hx[1:, 1:, :-1])/dx - 
                                   (Hz[1:, 1:, 1:] - Hz[:-1, 1:, 1:])/dx)
    Ez[1:, 1:, 1:] += (dt/eps0) * ((Hy[1:, 1:, 1:] - Hy[:-1, 1:, 1:])/dx - 
                                   (Hx[1:, 1:, 1:] - Hx[1:, :-1, 1:])/dx)

    pulse = np.sin(2 * np.pi * 1e9 * frame * dt)
    Ez[size//2, size//2, size//2] += pulse

    im.set_array(Ez[:, :, current_z])
    info_text.set_text(f"Z-Slice Index: {current_z} (Use Up/Down Keys)")
    return [im, info_text]

ani = animation.FuncAnimation(fig, update, interval=20, blit=True)
plt.show()
ani.save("wave.mp4", writer="ffmpeg", fps=30)