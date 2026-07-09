import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class Light_ray:
    def __init__(self, L, E, r, phi):

        self.L = L
        self.E = E
        self.s = np.array([r, -(E**2/c**2-L**2/r**2 * (1-R_s/r))**0.5, phi])

        self.x=np.full(200, r*np.cos(phi))
        self.y=np.full(200, r*np.sin(phi))

    def derivation(self, s):
        return np.array([s[1], (self.L**2/s[0]**3 - 3*self.L**2*R_s/(2*s[0]**4)), self.L / s[0] ** 2])

    def rk4_step(self):
        k1 = self.derivation(self.s)
        k2 = self.derivation(self.s+d_lamda/2*k1)
        k3 = self.derivation(self.s+d_lamda/2*k2)
        k4 = self.derivation(self.s+d_lamda*k3)
        self.s += d_lamda*(k1+2*k2+2*k3+k4)/6

    def update(self):
        self.rk4_step()

        self.x[1:] = self.x[:-1]
        self.y[1:] = self.y[:-1]
        self.x[0] = self.s[0]*np.cos(self.s[2])
        self.y[0] = self.s[0]*np.sin(self.s[2])



G=1
c=1
M=1

R_s=2*G*M/c**2

d_lamda=0.01

light_ray_list=[Light_ray(3*3**.5, 1, 5, 0)]#[Light_ray(5.0,1,5,0), Light_ray(5.1,1,5,0), Light_ray(5.2,1,5,0), Light_ray(5.3,1,5,0)]

fig, axplt = plt.subplots(figsize=(16, 12))
axplt.set_xlim(-5, 5)
axplt.set_ylim(-5, 5)
axplt.set_xlabel("x (m)")
axplt.set_ylabel("y (m)")
axplt.set_title("Black hole")
axplt.set_aspect("equal")

(black_hole,) = axplt.plot([0], [0], "o", markersize=1, label="black hole")
(event_horizon,) = axplt.plot(R_s*np.cos(np.linspace(0, 2*np.pi)), R_s*np.sin(np.linspace(0, 2*np.pi)), lw=1, label="event horizon")
(photon_sphere,) = axplt.plot(3/2*R_s*np.cos(np.linspace(0, 2*np.pi)), 3/2*R_s*np.sin(np.linspace(0, 2*np.pi)), lw=1, label="photon sphere", linestyle="dotted")

#energy_text = axplt.text(0.5, 0.9, "Double pendulum", transform=axplt.transAxes, ha='center', va='bottom', fontsize=12)

light_markers= []
traj_lines = []
for light_ray in light_ray_list:
    marker, = axplt.plot([], [], "o", markersize=1)
    traj_line, = axplt.plot([], [], lw=1, label=f"light ray({light_ray.L},{light_ray.E})")
    light_markers.append(marker)
    traj_lines.append(traj_line)

def init():
    for light_marker, traj_line in zip(light_markers, traj_lines):
        light_marker.set_data([], [])
        traj_line.set_data([], [])
    return tuple(light_markers+traj_lines)

def update(frame):
    for light_ray, light_marker, traj_line in zip(light_ray_list, light_markers, traj_lines):
        
        if light_ray.s[0] >= R_s * 0.99: 
            light_ray.update()
            
            light_marker.set_data([light_ray.s[0] * np.cos(light_ray.s[2])], 
                                  [light_ray.s[0] * np.sin(light_ray.s[2])])
            traj_line.set_data(light_ray.x, light_ray.y)
            
    return tuple(light_markers + traj_lines)

ani = animation.FuncAnimation(
    fig,
    update,
    init_func=init,
    frames=3000,
    interval=1000 * d_lamda,
    blit=True,
    repeat=False,
)
plt.legend()
#plt.show()


ani.save("blackhole rk4 3r3.mp4", writer="ffmpeg", fps=100)