import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

m1=1
m2=1
l1=1.5
l2=1
g=9.8

dt=5e-3

theta1=3*np.pi/4
theta1_dot=0
theta1_dot_dot=0

theta2=np.pi
theta2_dot = 0
theta2_dot_dot = 0

x = np.full(500, l1 * np.sin(theta1) + l2 * np.sin(theta2))
y = np.full(500, -l1 * np.cos(theta1) - l2 * np.cos(theta2))

def cal():
    global theta1, theta1_dot, theta1_dot_dot, theta2, theta2_dot, theta2_dot_dot

    theta1_dot_dot = (
        -g * (2 * m1 + m2) * np.sin(theta1)
        - m2 * g * np.sin(theta1 - 2 * theta2)
        - 2
        * np.sin(theta1 - theta2)
        * m2
        * (theta2_dot**2 * l2 + theta1_dot**2 * l1 * np.cos(theta1 - theta2))
    ) / (l1 * (2 * m1 + m2 - m2 * np.cos(2 * theta1 - 2 * theta2)))
    theta2_dot_dot = (
        2
        * np.sin(theta1 - theta2)
        * (
            (m1 + m2)*(theta1_dot**2 * l1 + g * np.cos(theta1))
            + theta2_dot**2 * l2 * m2 * np.cos(theta1 - theta2)
        )
    ) / (l2 * (2 * m1 + m2 - m2 * np.cos(2 * theta1 - 2 * theta2)))

    theta1_dot += theta1_dot_dot*dt
    theta2_dot += theta2_dot_dot*dt

    theta1 += theta1_dot*dt
    theta2 += theta2_dot*dt


# 애니메이션 준비
fig, axplt = plt.subplots()
axplt.set_xlim(-3, 3)
axplt.set_ylim(-3, 3)
axplt.set_xlabel("x (m)")
axplt.set_ylabel("y (m)")
axplt.set_title("Double pendulum")
axplt.set_aspect("equal")

(rod,) = axplt.plot([], [], lw=2)
(bob1,) = axplt.plot([], [], "o", markersize=10)
(bob2,) = axplt.plot([], [], "o", markersize=10)
(traj_line,) = axplt.plot([], [], lw=1)

def init():
    rod.set_data([], [])
    bob1.set_data([], [])
    bob2.set_data([], [])
    traj_line.set_data([], [])
    return rod, bob1, bob2, traj_line


def update(frame):
    global x, y
    x[1:] = x[:-1]
    y[1:] = y[:-1]
    x[0] = l1 * np.sin(theta1) + l2 * np.sin(theta2)
    y[0] = -l1 * np.cos(theta1) - l2 * np.cos(theta2)
    cal()
    rod.set_data(
        [0, l1 * np.sin(theta1), l1 * np.sin(theta1) + l2 * np.sin(theta2)],
        [0, -l1 * np.cos(theta1), -l1 * np.cos(theta1) - l2 * np.cos(theta2)],
    )
    bob1.set_data([l1*np.sin(theta1)], [-l1*np.cos(theta1)])
    bob2.set_data([l1*np.sin(theta1) + l2*np.sin(theta2)], [-l1*np.cos(theta1) - l2*np.cos(theta2)])
    traj_line.set_data(x, y)
    return rod, bob1, bob2, traj_line


ani = animation.FuncAnimation(
    fig,
    update,
    init_func=init,
    interval=int(1000 * dt),
    blit=True,
    repeat=False,
)
plt.show()
