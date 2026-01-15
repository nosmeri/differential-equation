import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class Vector2D:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __add__(self, other):
        return Vector2D(self.x + other.x, self.y + other.y)

    def __sub__(self, other):
        return Vector2D(self.x - other.x, self.y - other.y)

    def size(self):
        return np.sqrt(self.x**2 + self.y**2)

    def __mul__(self, other):
        return Vector2D(self.x * other, self.y * other)
    
    def __rmul__(self, other):
        return Vector2D(self.x * other, self.y * other)

    def normalized(self):
        return self * (1 / self.size())

    def __repr__(self):
        return f"x= {self.x}, y={self.y}"

G=1#6.67430e-11
dt=0.005#60*60*24

class Body:
    def __init__(self, mass, position, velocity = Vector2D(0, 0), color="blue"):
        self.mass = mass
        self.position = position
        self.velocity = velocity
        self.color = color

bodies = [
    Body(mass=1,position=Vector2D(-1., 0.), velocity=Vector2D(0.347111, 0.532728), color="red"),
    Body(mass=1,position=Vector2D(1., 0.), velocity=Vector2D(0.347111, 0.532728), color="green"),
    Body(mass=1,position=Vector2D(0., 0.), velocity=Vector2D(-0.694222, -1.065456), color="blue"),
]
"""
Body(mass=1,position=Vector2D(0.97000436, -0.24308753), velocity=Vector2D(0.46620368, 0.43236573), color="red"),
Body(mass=1,position=Vector2D(-0.97000436, 0.24308753), velocity=Vector2D(0.46620368, 0.43236573), color="green"),
Body(mass=1,position=Vector2D(0.0,0.0), velocity=Vector2D(-0.93240737, -0.86473146), color="blue")
"""

tracking_body_index = 0 
view_range = 3#3e11 

def on_key_press(event):
    global tracking_body_index, view_range

    if event.key in ['0', '1', '2','3']:
        new_index = int(event.key)
        if new_index < len(bodies):
            tracking_body_index = new_index
    if event.key == "[":
        view_range *= 1.5
    if event.key == "]":
        view_range /= 1.5

fig, axplt = plt.subplots()
axplt.set_facecolor('black')
axplt.set_xlabel("x (m)")
axplt.set_ylabel("y (m)")
axplt.set_title("N-body problem")
axplt.set_aspect("equal")

body_markers = []
body_trackers = []
pos_x = []
pos_y = []
for body in bodies:
    marker, = axplt.plot([], [], "o", markersize=5, color=body.color)
    tracker, = axplt.plot([],[], lw=1, color=body.color)
    body_markers.append(marker)
    body_trackers.append(tracker)
    pos_x.append(np.full(10, body.position.x))
    pos_y.append(np.full(10, body.position.y))

def init():
    # 초기 화면을 추적 중인 천체 중심으로 설정
    tracking_body = bodies[tracking_body_index]
    center_x = tracking_body.position.x
    center_y = tracking_body.position.y
    axplt.set_xlim(center_x - view_range, center_x + view_range)
    axplt.set_ylim(center_y - view_range, center_y + view_range)
    for i, body in enumerate(bodies):
        body_markers[i].set_data([body.position.x], [body.position.y])
    return tuple(body_markers+body_trackers)

def derivation(body, state):
    
    a_sum=Vector2D(0,0)
    for other_body in bodies:
        if body != other_body:
            distance = (state[0] - other_body.position).size()
            a = (
                G
                * other_body.mass
                / distance**2
                * (other_body.position - state[0]).normalized()
            )
            a_sum += a
    return np.array([state[1], a_sum])
    

def rk4_step(body):
    state=np.array([body.position, body.velocity])
    k1=derivation(body, state)
    k2=derivation(body, state+dt/2*k1)
    k3=derivation(body, state+dt/2*k2)
    k4=derivation(body, state+dt*k3)
    return state + dt*(k1+2*k2+2*k3+k4)*(1/6)


def update(frame):
    for body in bodies:
        state=rk4_step(body)
        body.position=state[0]
        body.velocity=state[1]
    tracking_body = bodies[tracking_body_index-1] if tracking_body_index>=1 else Body(0,Vector2D(0,0))
    center_x = tracking_body.position.x
    center_y = tracking_body.position.y
    axplt.set_xlim(center_x - view_range, center_x + view_range)
    axplt.set_ylim(center_y - view_range, center_y + view_range)
    
    # 각 천체 마커 업데이트
    for i, body in enumerate(bodies):
        body_markers[i].set_data([body.position.x], [body.position.y])
        pos_x[i] = np.roll(pos_x[i], -1)
        pos_y[i] = np.roll(pos_y[i], -1)
        pos_x[i][-1] = body.position.x
        pos_y[i][-1] = body.position.y
        body_trackers[i].set_data(pos_x[i], pos_y[i])
    return tuple(body_markers+body_trackers)


# 키보드 이벤트 연결
fig.canvas.mpl_connect('key_press_event', on_key_press)

ani = animation.FuncAnimation(
    fig,
    update,
    init_func=init,
    interval=0.1,#1000 * dt,#/60/60/24/30,
    blit=True,
    repeat=False,
)

plt.show()
