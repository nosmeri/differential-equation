import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def f(x):
    return np.sin(x)

def diff(f, x):
    h=1e-6
    return (f(x+h)-f(x-h))/(2*h)


interval=10
dt=interval/1000


x = np.linspace(0, 10, 100)
y = np.zeros_like(x)

fig, ax = plt.subplots()
ax.set_ylim(-2,2)
(line,) = ax.plot(x, y)


def animate(frame):
    t=frame*dt
    dy2=diff()
    dy+=dy2*dt
    y = y + dy*dt
    line.set_ydata(y)
    return (line,)


ani = FuncAnimation(fig, animate, interval=interval, blit=True)
plt.show()
