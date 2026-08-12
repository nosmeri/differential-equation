import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

hbar = 1.0
m = 1.0

N = 1024        
L = 40.0          
x = np.linspace(-L/2, L/2, N, endpoint=False)
dx = x[1] - x[0]

k = np.fft.fftfreq(N, d=dx) * 2.0 * np.pi # 운동량 공간에서 각파수 (p/h*2pi)  p/h가 주파수

V_height = 15.0   
barrier_width = 0.6  
V = np.where(np.abs(x) < barrier_width / 2.0, V_height, 0.0)

x0 = -5.0         
sigma0 = 0.6     
k0 = 5.0 # 초기 운동량 기댓값

psi = np.exp(-((x - x0)**2) / (4.0 * sigma0**2)) * np.exp(1j * k0 * x)
psi /= np.sqrt(np.sum(np.abs(psi)**2) * dx)

dt = 0.001

exp_V2 = np.exp(-1j * V * (dt / 2.0) / hbar)
T_k = (hbar**2 * k**2) / (2.0 * m)
exp_T = np.exp(-1j * T_k * dt / hbar)

def step_split_operator(psi_in):
    psi_temp = psi_in * exp_V2
    psi_k = np.fft.fft(psi_temp)
    psi_k_evolved = psi_k * exp_T
    psi_x = np.fft.ifft(psi_k_evolved)
    return psi_x * exp_V2

fig, ax = plt.subplots(figsize=(9, 5))
ax.set_xlim(-8, 8)
ax.set_ylim(-0.1, 1.2)

ax.plot(x, V / V_height * 0.8, 'r-', lw=2, label=f'Barrier $V_0 = {V_height}$ (scaled)')
ax.fill_between(x, 0, V / V_height * 0.8, color='red', alpha=0.15)

line_prob, = ax.plot([], [], 'b-', lw=2, label=r'Probability Density $|\Psi(x,t)|^2$')
line_real, = ax.plot([], [], 'g:', alpha=0.5, label=r'Real Part Re$(\Psi(x,t))$')
line_imag, = ax.plot([], [], 'm:', alpha=0.5, label=r'Imaginary Part Im$(\Psi(x,t))$')
fill_prob = ax.fill_between(x, 0, 0, color='blue', alpha=0.2)

ax.set_title('Quantum Tunneling & Scattering Simulation (Split-Operator FFT)', fontsize=12)
ax.set_xlabel('Position x')
ax.set_ylabel('Amplitude / Density')
ax.legend(loc='upper right')
ax.grid(True, linestyle=':', alpha=0.6)

time_text = ax.text(0.03, 0.88, '', transform=ax.transAxes, fontsize=10, fontweight='bold')
prob_text = ax.text(0.03, 0.80, '', transform=ax.transAxes, fontsize=10, color='darkblue')

steps_per_frame = 8
current_psi = psi.copy()
current_t = 0.0

def init():
    line_prob.set_data([], [])
    line_real.set_data([], [])
    line_imag.set_data([], [])
    time_text.set_text('')
    prob_text.set_text('')
    return line_prob, line_real, time_text, prob_text

def update(frame):
    global current_psi, current_t, fill_prob
    
    for _ in range(steps_per_frame):
        current_psi = step_split_operator(current_psi)
        current_t += dt
    
    prob = np.abs(current_psi)**2
    
    line_prob.set_data(x, prob)
    line_real.set_data(x, np.real(current_psi))
    line_imag.set_data(x, np.imag(current_psi))
    
    fill_prob.remove()
    fill_prob = ax.fill_between(x, 0, prob, color='blue', alpha=0.2)
    
    R = np.sum(prob[x < 0]) * dx
    T_prob = np.sum(prob[x >= 0]) * dx
    
    time_text.set_text(f'Time t = {current_t:.2f} s')
    prob_text.set_text(f'Reflection (R): {R*100:.1f}%  |  Transmission (T): {T_prob*100:.1f}%')
    return line_prob, line_real, fill_prob, time_text, prob_text

anim = FuncAnimation(fig, update, init_func=init, frames=200, interval=30, blit=False)

plt.show()