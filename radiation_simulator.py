import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

dt=1e-2

time_limit=120

sigma=5.670374419e-8

R=6371000 # 지구 반지름
S=1361 # 태양상수
E=S*np.pi*R**2 # 지구로 들어오는 총 에너지
CM_surface=1e16 # 지표 열용량
CM_air=1e16 # 대기 열용량
albedo_air=0.31 # 대기 반사율
transmittance_space_to_surface=0.64 # 단파 복사 대기 투과율
transmittance_surface_to_space=0.09 # 장파 복사 대기 투과율
radiation_air_to_surface=0.56 # 대기->지표 복사율


t=np.arange(0,time_limit,dt)
T_air=np.full_like(t,500)
T_surface=np.full_like(t,0)


fig, axplt = plt.subplots()
axplt.set_xlabel("t (s)")
axplt.set_ylabel("T (K)")
axplt.set_title("Radiation simulation")

air, = axplt.plot([], [], lw=2, color="blue", label="air temperature")
surface, = axplt.plot([], [], lw=2, color="green", label="surface temperature")

axplt.set_xlim(0,time_limit)
axplt.set_ylim(0,500)

def init():
    air.set_data([], [])
    surface.set_data([], [])
    return air, surface


def update(frame):
    E_reflection=albedo_air*E # 대기 반사
    E_transmission=transmittance_space_to_surface*(1-albedo_air)*E # 대기 투과
    E_absorption=E-E_reflection-E_transmission # 대기 흡수
    E_radiation_surface=sigma*T_surface[frame-1]**4*4*np.pi*R**2 # 지표 복사
    E_radiation_air=sigma*T_air[frame-1]**4*4*np.pi*R**2 # 대기 복사
    
    T_air[frame] = T_air[frame-1] + dt*(E_absorption+E_radiation_surface*(1-transmittance_surface_to_space) - E_radiation_air)/CM_air
    T_surface[frame] = T_surface[frame-1] + dt*(E_transmission + E_radiation_air*radiation_air_to_surface - E_radiation_surface)/CM_surface
    air.set_data(t[:frame], T_air[:frame])
    surface.set_data(t[:frame], T_surface[:frame])
    print(T_surface[frame])
    return air, surface


ani = animation.FuncAnimation(
    fig,
    update,
    init_func=init,
    frames=len(t),
    interval=1,
    blit=True,
    repeat=False,
)

plt.legend()
plt.show()
