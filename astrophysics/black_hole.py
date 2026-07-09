import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class Light_ray:
    def __init__(self, L, E, r, phi):
        self.L = L
        self.E = E
        self.r = r
        self.phi = phi

        self.r_dot = -(E**2/c**2-L**2/r**2 * (1-R_s/r))**0.5

        self.x=np.full(200, r*np.cos(phi))
        self.y=np.full(200, r*np.sin(phi))

    def update(self):
        self.r_dot_dot= self.L**2/self.r**3 - 3 *self.L**2 *R_s / (2*self.r**4) 
        self.phi_dot = self.L / self.r**2
        
        self.r_dot+=self.r_dot_dot * d_lamda

        self.phi+=self.phi_dot * d_lamda
        self.r+=self.r_dot * d_lamda

        self.x[1:] = self.x[:-1]
        self.y[1:] = self.y[:-1]
        self.x[0] = self.r*np.cos(self.phi)
        self.y[0] = self.r*np.sin(self.phi)



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
        
        if light_ray.r >= R_s * 0.99: 
            light_ray.update()
            
            light_marker.set_data([light_ray.r * np.cos(light_ray.phi)], 
                                  [light_ray.r * np.sin(light_ray.phi)])
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


ani.save("blackhole e 3r3.mp4", writer="ffmpeg", fps=100)