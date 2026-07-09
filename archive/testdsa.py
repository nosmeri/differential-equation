import numpy as np
import matplotlib.pyplot as plt

def normalize_angle(angle):
    """각도를 -pi와 pi 사이로 정규화하는 함수"""
    return (angle + np.pi) % (2 * np.pi) - np.pi

# ==========================================
# 1. 시뮬레이션 환경 구축 및 가상 데이터 생성
# ==========================================
# 정답 궤적 (Ground Truth): 4개의 노드로 이루어진 정사각형 경로 (x, y, theta)
true_poses = np.array([
    [0.0, 0.0, 0.0],          # 노드 0 (시작점)
    [1.0, 0.0, np.pi/2],      # 노드 1
    [1.0, 1.0, np.pi],        # 노드 2
    [0.0, 1.0, 3*np.pi/2]     # 노드 3
])
num_nodes = len(true_poses)

# 그래프의 에지(Edge) 정의: (시작 노드, 끝 노드)
# 0->1->2->3 연속 주행 후, 3번 노드에서 다시 0번 노드를 관측하는 '루프 폐쇄(Loop Closure)' 추가
edges = [(0, 1), (1, 2), (2, 3), (3, 0)]

#np.random.seed(42) # 결과 재현성을 위한 시드 고정
measurements = []
Omega = np.diag([100.0, 100.0, 100.0]) # 정보 행렬 (측정 신뢰도, 센서 오차의 역수)

# 각 에지별 노이즈가 포함된 상대 변위(Odometry) 측정값 생성
for i, j in edges:
    # 정답 데이터 기반의 상대 변위 계산 (Global frame)
    dx_g = true_poses[j][0] - true_poses[i][0]
    dy_g = true_poses[j][1] - true_poses[i][1]
    dtheta = normalize_angle(true_poses[j][2] - true_poses[i][2])
    
    # 로봇의 현재 로컬 좌표계(Local frame)로 변환
    theta_i = true_poses[i][2]
    dx_l = np.cos(theta_i) * dx_g + np.sin(theta_i) * dy_g
    dy_l = -np.sin(theta_i) * dx_g + np.cos(theta_i) * dy_g
    
    # 인위적인 가우시안 센서 노이즈 삽입 (드리프트 오차 모사)
    dx_l += np.random.normal(0, 0.08)
    dy_l += np.random.normal(0, 0.08)
    dtheta += np.random.normal(0, 0.03)
    
    measurements.append((i, j, np.array([dx_l, dy_l, normalize_angle(dtheta)])))

# ==========================================
# 2. 초기 추정치(Initial Guess) 설정: 오차가 누적된 궤적
# ==========================================
# 루프 폐쇄 없이 센서 값만 단순 누적하여 왜곡된 초기 경로를 계산합니다.
estimated_poses = np.zeros_like(true_poses)
for idx in range(1, num_nodes):
    for i, j, meas in measurements:
        if i == idx-1 and j == idx:
            theta_prev = estimated_poses[i][2]
            # Local to Global 좌표 변환
            dx_g = np.cos(theta_prev) * meas[0] - np.sin(theta_prev) * meas[1]
            dy_g = np.sin(theta_prev) * meas[0] + np.cos(theta_prev) * meas[1]
            estimated_poses[j][0] = estimated_poses[i][0] + dx_g
            estimated_poses[j][1] = estimated_poses[i][1] + dy_g
            estimated_poses[j][2] = normalize_angle(estimated_poses[i][2] + meas[2])

# 최적화 전 그래프 시각화를 위해 왜곡된 상태를 저장해둠
drifted_poses = estimated_poses.copy()

# ==========================================
# 3. 가우스-뉴턴(Gauss-Newton) 비선형 최적화 루프
# ==========================================
iterations = 5
print("--- 가우스-뉴턴 최적화 시작 ---")

for it in range(iterations):
    # 전역 헤시안 행렬 H와 gradient 벡터 b 초기화 (크기: 노드 수 * 3차원)
    H = np.zeros((num_nodes * 3, num_nodes * 3))
    b = np.zeros(num_nodes * 3)
    
    # 첫 번째 노드를 원점(Anchor)으로 고정하기 위해 강한 제약조건 부여
    H[0:3, 0:3] += np.identity(3) * 1e6
    
    for i, j, meas in measurements:
        x_i = estimated_poses[i]
        x_j = estimated_poses[j]
        
        # 현재 추정치 기반의 예측 변위 계산 (비선형 함수 f(x))
        dx_g = x_j[0] - x_i[0]
        dy_g = x_j[1] - x_i[1]
        theta_i = x_i[2]
        
        f_x = np.array([
            np.cos(theta_i) * dx_g + np.sin(theta_i) * dy_g,
            -np.sin(theta_i) * dx_g + np.cos(theta_i) * dy_g,
            normalize_angle(x_j[2] - x_i[2])
        ])
        
        # 잔차(Residual) 오차 계산: e = f(x) - z
        e = f_x - meas
        e[2] = normalize_angle(e[2])
        
        # 비선형 함수에 대한 자코비안(Jacobian) 행렬 A(i에 대한 미분)와 B(j에 대한 미분) 도출
        A = np.array([
            [-np.cos(theta_i), -np.sin(theta_i), -np.sin(theta_i)*dx_g + np.cos(theta_i)*dy_g],
            [np.sin(theta_i),  -np.cos(theta_i), -np.cos(theta_i)*dx_g - np.sin(theta_i)*dy_g],
            [0, 0, -1]
        ])
        
        B = np.array([
            [np.cos(theta_i), np.sin(theta_i), 0],
            [-np.sin(theta_i), np.cos(theta_i), 0],
            [0, 0, 1]
        ])
        
        # 전역 행렬 인덱스 매핑 슬라이스
        idx_i = slice(i*3, (i+1)*3)
        idx_j = slice(j*3, (j+1)*3)
        
        # 가우스-뉴턴 수식에 따른 전역 H와 b 누적 업데이트
        H[idx_i, idx_i] += A.T @ Omega @ A
        H[idx_i, idx_j] += A.T @ Omega @ B
        H[idx_j, idx_i] += B.T @ Omega @ A
        H[idx_j, idx_j] += B.T @ Omega @ B
        
        b[idx_i] += A.T @ Omega @ e
        b[idx_j] += B.T @ Omega @ e
        
    # 선형 시스템 정규 방정식 (H * delta_x = -b) 풀기
    delta_x = np.linalg.solve(H, -b)
    
    # 상태 변수 업데이트
    for idx in range(num_nodes):
        estimated_poses[idx] += delta_x[idx*3:(idx+1)*3]
        estimated_poses[idx][2] = normalize_angle(estimated_poses[idx][2])
        
    # 오차 제곱합(Cost) 출력
    cost = np.sum(b**2)
    print(f"반복 {it+1}회차: 전역 Cost (오차 제곱합) = {cost:.4f}")

# ==========================================
# 4. 결과 시각화 (Matplotlib)
# ==========================================
plt.figure(figsize=(10, 8))

# 궤적 플롯팅 함수 정의
def plot_trajectory(poses, label, color, marker):
    # 경로를 닫아서 그리기 위해 시작점을 끝에 추가
    x_coords = list(poses[:, 0]) + [poses[0, 0]]
    y_coords = list(poses[:, 1]) + [poses[0, 1]]
    plt.plot(x_coords, y_coords, color=color, linestyle='-', marker=marker, label=label, linewidth=2)
    for idx, pos in enumerate(poses):
        plt.text(pos[0]+0.02, pos[1]+0.02, f"X{idx}", fontsize=12, color=color, weight='bold')

plot_trajectory(true_poses, "Ground Truth", "green", "o")
plot_trajectory(drifted_poses, "Before Optimization", "red", "x")
plot_trajectory(estimated_poses, "After Optimization", "blue", "s")

plt.title("Pose Graph Optimization using Gauss-Newton Simulation", fontsize=14, weight='bold')
plt.xlabel("X Coordinate (m)", fontsize=12)
plt.ylabel("Y Coordinate (m)", fontsize=12)
plt.legend(fontsize=12, loc='best')
plt.grid(True, linestyle='--', alpha=0.6)
plt.axis('equal')
plt.show()