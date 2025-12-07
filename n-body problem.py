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

G=6.67430e-11
dt=60*60*24

class Body:
    def __init__(self, mass, position, velocity = Vector2D(0, 0), color="blue"):
        self.mass = mass
        self.position = position
        self.velocity = velocity
        self.color = color

bodies = [
        Body(mass=1.9891e30, position=Vector2D(0, 0), color="orange"),  # 태양
    Body(
        mass=7.36e22,
        position=Vector2D(1.5e11 + 384400e3, 0),
        velocity=Vector2D(0, 29.78e3 + 1.022e3),
        color="gray",
    ),  # 달
    Body(
        mass=5.972e24,
        position=Vector2D(1.5e11, 0),
        velocity=Vector2D(0, 29.78e3),
        color="blue",
    ),  # 지구
    Body(
        mass=3.3e23,
        position=Vector2D(0.58e11, 0),
        velocity=Vector2D(0, 47e3),
        color="gray",
    ),  # 수성
    Body(
        mass=4.86e24,
        position=Vector2D(1.082e11, 0),
        velocity=Vector2D(0, 35e3),
        color="red",
    ),  # 금성
]

tracking_body_index = 0 
view_range = 3e11 

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
axplt.set_xlabel("x (m)")
axplt.set_ylabel("y (m)")
axplt.set_title("N-body problem")
axplt.set_aspect("equal")

body_markers = []
for body in bodies:
    marker, = axplt.plot([], [], "o", markersize=5, color=body.color)
    body_markers.append(marker)

def init():
    # 초기 화면을 추적 중인 천체 중심으로 설정
    tracking_body = bodies[tracking_body_index]
    center_x = tracking_body.position.x
    center_y = tracking_body.position.y
    axplt.set_xlim(center_x - view_range, center_x + view_range)
    axplt.set_ylim(center_y - view_range, center_y + view_range)
    for i, body in enumerate(bodies):
        body_markers[i].set_data([body.position.x], [body.position.y])
    return tuple(body_markers)


def update(frame):
    for body in bodies:
        for other_body in bodies:
            if body != other_body:
                distance = (body.position - other_body.position).size()
                a = (
                    G
                    * other_body.mass
                    / distance**2
                    * (other_body.position - body.position).normalized()
                )
                body.velocity += a * dt
    for body in bodies:
        body.position += body.velocity * dt
    tracking_body = bodies[tracking_body_index-1] if tracking_body_index>=1 else Body(0,Vector2D(0,0))
    center_x = tracking_body.position.x
    center_y = tracking_body.position.y
    axplt.set_xlim(center_x - view_range, center_x + view_range)
    axplt.set_ylim(center_y - view_range, center_y + view_range)
    
    # 각 천체 마커 업데이트
    for i, body in enumerate(bodies):
        body_markers[i].set_data([body.position.x], [body.position.y])
    return tuple(body_markers)


# 키보드 이벤트 연결
fig.canvas.mpl_connect('key_press_event', on_key_press)

ani = animation.FuncAnimation(
    fig,
    update,
    init_func=init,
    interval=1000 * dt/60/60/24/30,
    blit=True,
    repeat=False,
)

plt.show()
