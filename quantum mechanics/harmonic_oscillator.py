import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

hbar = 1.0
m = 1.0
omega = 1.0

N = 512
L = 12.0
x = np.linspace(-L/2, L/2, N, endpoint=False)
dx = x[1] - x[0]

k = np.fft.fftfreq(N, d=dx) * 2.0 * np.pi

V = 0.5 * m * (omega**2) * (x**2)

x0 = 2.5
sigma0 = 0.707

psi = np.exp(-((x - x0)**2) / (4.0 * sigma0**2)).astype(complex)

psi /= np.sqrt(np.sum(np.abs(psi)**2) * dx)
dt = 0.02

exp_V2 = np.exp(-1j * V * (dt / 2.0) / hbar)

T_k = (hbar**2 * k**2) / (2.0 * m)
exp_T = np.exp(-1j * T_k * dt / hbar)

def step_split_operator(psi_in):
    psi_temp = psi_in * exp_V2
    
    psi_k = np.fft.fft(psi_temp)
    psi_k_evolved = psi_k * exp_T
    
    psi_x = np.fft.ifft(psi_k_evolved)
    return psi_x * exp_V2

fig, ax = plt.subplots(figsize=(8, 5))
ax.set_xlim(-6, 6)
ax.set_ylim(-0.1, 1.2)

ax.plot(x, V, 'k--', alpha=0.5, label=r'Potential $V(x) = \frac{1}{2}m\omega^2 x^2$')

line_prob, = ax.plot([], [], 'b-', lw=2, label=r'Probability Density $|\Psi(x,t)|^2$')
line_real, = ax.plot([], [], 'g:', alpha=0.6, label=r'Real Part Re$(\Psi(x,t))$')
fill_prob = ax.fill_between(x, 0, 0, color='blue', alpha=0.2)

ax.set_title('Quantum Harmonic Oscillator (Pure Numerical Split-Operator FFT)', fontsize=12)
ax.set_xlabel('Position x')
ax.set_ylabel('Amplitude / Density')
ax.legend(loc='upper right')
ax.grid(True, linestyle=':', alpha=0.6)

time_text = ax.text(0.03, 0.88, '', transform=ax.transAxes, fontsize=11, fontweight='bold')

steps_per_frame = 3
current_psi = psi.copy()
current_t = 0.0

def init():
    line_prob.set_data([], [])
    line_real.set_data([], [])
    time_text.set_text('')
    return line_prob, line_real, time_text

def update(frame):
    global current_psi, current_t, fill_prob
    
    for _ in range(steps_per_frame):
        current_psi = step_split_operator(current_psi)
        current_t += dt
    
    prob = np.abs(current_psi)**2
    
    line_prob.set_data(x, prob)
    line_real.set_data(x, np.real(current_psi))
    
    fill_prob.remove()
    fill_prob = ax.fill_between(x, 0, prob, color='blue', alpha=0.2)
    
    time_text.set_text(f'Time t = {current_t:.2f} s')
    return line_prob, line_real, fill_prob, time_text

anim = FuncAnimation(fig, update, init_func=init, frames=200, interval=30, blit=False)

plt.show()