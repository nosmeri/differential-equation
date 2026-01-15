import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

k=1
m=1
x=3

dt=0.1

A=x

euler_x=x
euler_v=0
rk4_x=x
rk4_v=0

t=-50

real_x_history=[x]
euler_x_history=[euler_x]
rk4_x_history=[rk4_x]
t_history=[t]

fig, axplt = plt.subplots()
axplt.set_xlim(0, 20)
axplt.set_ylim(-10, 10)
axplt.set_xlabel("t (s)")
axplt.set_ylabel("x (m)")
axplt.set_title("Spring motion")
axplt.set_aspect("equal")

(real_graph,) = axplt.plot([], [], "-", lw=1, color="blue", label="real")
(euler_graph,) = axplt.plot([], [], "-", lw=1, color="green", label="euler")
(rk4_graph,) = axplt.plot([], [], "-", lw=1, color="red", label="rk4")

axplt.legend(loc="upper right")

def derivation(state):
    return np.array([state[1],-k/m*state[0]])

def rk4_step(state):
    k1=derivation(state)
    k2=derivation(state+k1*dt/2)
    k3=derivation(state+k2*dt/2)
    k4=derivation(state+k3*dt)
    return (state+k1*dt/6+k2*dt/3+k3*dt/3+k4*dt/6)

def init():
    real_graph.set_data([], [])
    euler_graph.set_data([], [])
    rk4_graph.set_data([], [])
    return real_graph, euler_graph, rk4_graph

def update(frame):
    global t, euler_x, euler_v, rk4_x, rk4_v

    t+=dt
    t_history.append(t)

    real_x=A*np.cos((k/m)**0.5 *t)
    real_x_history.append(real_x)
    
    euler_a=-k/m*euler_x
    euler_v+=euler_a*dt
    euler_x+=euler_v*dt
    euler_x_history.append(euler_x)
    
    state=np.array([rk4_x,rk4_v])
    state=rk4_step(state)
    rk4_x=state[0]
    rk4_v=state[1]
    rk4_x_history.append(state[0])
    
    real_graph.set_data(t_history, real_x_history)
    euler_graph.set_data(t_history, euler_x_history)
    rk4_graph.set_data(t_history, rk4_x_history)
    return real_graph, euler_graph, rk4_graph

ani = animation.FuncAnimation(
    fig,
    update,
    init_func=init,
    interval=1,
    blit=True,
    repeat=False,
)
plt.show()
