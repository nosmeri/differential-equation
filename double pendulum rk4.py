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

theta2=np.pi
theta2_dot = 0

x = np.full(500, l1 * np.sin(theta1) + l2 * np.sin(theta2))
y = np.full(500, -l1 * np.cos(theta1) - l2 * np.cos(theta2))

def derivation(state):
    theta1_dot_dot = (
        -g * (2 * m1 + m2) * np.sin(state[0])
        - m2 * g * np.sin(state[0] - 2 * state[2])
        - 2
        * np.sin(state[0] - state[2])
        * m2
        * (state[3]**2 * l2 + state[1]**2 * l1 * np.cos(state[0] - state[2]))
    ) / (l1 * (2 * m1 + m2 - m2 * np.cos(2 * state[0] - 2 * state[2])))
    theta2_dot_dot = (
        2
        * np.sin(state[0] - state[2])
        * (
            (m1 + m2)*(state[1]**2 * l1 + g * np.cos(state[0]))
            + state[3]**2 * l2 * m2 * np.cos(state[0] - state[2])
        )
    ) / (l2 * (2 * m1 + m2 - m2 * np.cos(2 * state[0] - 2 * state[2])))
    return np.array([state[1], theta1_dot_dot, state[3], theta2_dot_dot])

def rk4_step(state):
    k1=derivation(state)
    k2=derivation(state+dt/2*k1)
    k3=derivation(state+dt/2*k2)
    k4=derivation(state+dt*k3)
    return state + dt*(k1+2*k2+2*k3+k4)/6


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
energy_text = axplt.text(0.5, 0.9, "Double pendulum", transform=axplt.transAxes, ha='center', va='bottom', fontsize=12)

def init():
    rod.set_data([], [])
    bob1.set_data([], [])
    bob2.set_data([], [])
    traj_line.set_data([], [])
    return rod, bob1, bob2, traj_line, energy_text


def update(frame):
    global x, y, theta1, theta1_dot, theta2, theta2_dot
    x[1:] = x[:-1]
    y[1:] = y[:-1]
    x[0] = l1 * np.sin(theta1) + l2 * np.sin(theta2)
    y[0] = -l1 * np.cos(theta1) - l2 * np.cos(theta2)
    
    state=np.array([theta1, theta1_dot, theta2, theta2_dot])
    state=rk4_step(state)
    theta1, theta1_dot, theta2, theta2_dot=state

    T1 = 0.5 * m1 * (l1 * theta1_dot)**2
    V1 = -m1 * g * l1 * np.cos(theta1)
    E1 = T1 + V1

    v2_sq = (l1 * theta1_dot)**2 + (l2 * theta2_dot)**2 + 2 * l1 * l2 * theta1_dot * theta2_dot * np.cos(theta1 - theta2)
    T2 = 0.5 * m2 * v2_sq
    V2 = -m2 * g * (l1 * np.cos(theta1) + l2 * np.cos(theta2))
    E2 = T2 + V2
    
    total_energy = E1 + E2
    
    energy_text.set_text(f"E1={E1:.2f}, E2={E2:.2f}, Total={total_energy:.2f}")

    rod.set_data(
        [0, l1 * np.sin(theta1), l1 * np.sin(theta1) + l2 * np.sin(theta2)],
        [0, -l1 * np.cos(theta1), -l1 * np.cos(theta1) - l2 * np.cos(theta2)],
    )
    bob1.set_data([l1*np.sin(theta1)], [-l1*np.cos(theta1)])
    bob2.set_data([l1*np.sin(theta1) + l2*np.sin(theta2)], [-l1*np.cos(theta1) - l2*np.cos(theta2)])
    traj_line.set_data(x, y)
    return rod, bob1, bob2, traj_line, energy_text


ani = animation.FuncAnimation(
    fig,
    update,
    init_func=init,
    interval=1000 * dt,
    blit=True,
    repeat=False,
)
plt.show()
