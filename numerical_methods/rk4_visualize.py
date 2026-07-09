import numpy as np
import matplotlib.pyplot as plt

# 미분방정식 dy/dt = cos(t) - 0.5*y
def f(t, y):
    return np.cos(t) - 0.5 * y

# 실제 참값 곡선 함수
def exact(t):
    return 0.6 * np.exp(-0.5 * t) + 0.4 * np.cos(t) + 0.8 * np.sin(t)

# 단일 RK4 스텝 매개변수 (시각화를 위해 큰 h 사용)
t0, y0 = 0.0, 1.0
h = 1.6 

# 4개의 기울기 및 가중평균 계산
k1 = f(t0, y0)
t_mid = t0 + h/2
y_mid1 = y0 + (h/2) * k1

k2 = f(t_mid, y_mid1)
y_mid2 = y0 + (h/2) * k2

k3 = f(t_mid, y_mid2)
t_end = t0 + h
y_end = y0 + h * k3

k4 = f(t_end, y_end)
k_avg = (k1 + 2*k2 + 2*k3 + k4) / 6
y1 = y0 + h * k_avg

# 시각화 설정
plt.figure(figsize=(12, 8), dpi=120)
t_fine = np.linspace(-0.2, t_end + 0.4, 300)
plt.plot(t_fine, exact(t_fine), label='True Solution Curve', color='gray', linestyle=':', alpha=0.7)

# 시작점
plt.scatter([t0], [y0], color='black', zorder=5, s=100)
plt.text(t0 - 0.05, y0 + 0.05, 'Start (t0, y0)', fontsize=11, fontweight='bold')

# k1 표현
t_k1 = np.linspace(t0, t_mid, 100)
plt.plot(t_k1, y0 + k1 * (t_k1 - t0), color='crimson', linestyle='--', label=f'k1 slope ({k1:.2f})')
plt.scatter([t_mid], [y_mid1], color='crimson', marker='o', s=80, zorder=5)

# k2 표현
t_k2 = np.linspace(t0, t_mid, 100)
plt.plot(t_k2, y0 + k2 * (t_k2 - t0), color='orange', linestyle='--', label=f'k2 slope ({k2:.2f})')
t_tangent2 = np.linspace(t_mid - 0.3, t_mid + 0.3, 50)
plt.plot(t_tangent2, y_mid1 + k2 * (t_tangent2 - t_mid), color='orange', alpha=0.5)
plt.scatter([t_mid], [y_mid2], color='orange', marker='s', s=80, zorder=5)

# k3 표현
t_k3 = np.linspace(t0, t_end, 100)
plt.plot(t_k3, y0 + k3 * (t_k3 - t0), color='forestgreen', linestyle='--', label=f'k3 slope ({k3:.2f})')
t_tangent3 = np.linspace(t_mid - 0.3, t_mid + 0.3, 50)
plt.plot(t_tangent3, y_mid2 + k3 * (t_tangent3 - t_mid), color='forestgreen', alpha=0.5)
plt.scatter([t_end], [y_end], color='forestgreen', marker='^', s=80, zorder=5)

# k4 표현
t_tangent4 = np.linspace(t_end - 0.3, t_end + 0.3, 50)
plt.plot(t_tangent4, y_end + k4 * (t_tangent4 - t_end), color='purple', alpha=0.5, linestyle='--', label=f'k4 slope ({k4:.2f})')

# 최종 RK4 결과선
plt.plot([t0, t_end], [y0, y1], color='blue', linewidth=3, label=f'Final RK4 Step (slope={k_avg:.2f})')
plt.scatter([t_end], [y1], color='blue', zorder=6, s=120, edgecolors='black')

# 축 및 스타일 설정
plt.axvline(x=t_mid, color='black', linestyle=':', alpha=0.3)
plt.axvline(x=t_end, color='black', linestyle=':', alpha=0.3)
plt.xticks([t0, t_mid, t_end], ['t_n', 't_n + h/2', 't_n + h'], fontsize=11)
plt.title('Geometric Interpretation of a Single RK4 Step ($k_1, k_2, k_3, k_4$)', fontsize=14, fontweight='bold')
plt.xlabel('Time (t)')
plt.ylabel('y')
plt.grid(True, alpha=0.3)
plt.legend(loc='upper left')
plt.xlim(-0.1, t_end + 0.5)
plt.ylim(0.4, 1.6)
plt.show()