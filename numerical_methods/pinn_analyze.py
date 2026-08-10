import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import time

# ---------------------------------------------------------
# 1. 시드 고정 및 기본 설정
# ---------------------------------------------------------
torch.manual_seed(42)
np.random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
alpha = 0.01  # 열전도 계수 (Thermal diffusivity)

# ---------------------------------------------------------
# 2. 열 방정식 해석해 (Exact Solution)
# ---------------------------------------------------------
def heat_exact(x, t, alpha_val=alpha):
    # u(x, t) = exp(-alpha * pi^2 * t) * sin(pi * x)
    return np.exp(-alpha_val * (np.pi**2) * t) * np.sin(np.pi * x)

# ---------------------------------------------------------
# 3. 신경망 아키텍처 (MLP)
# ---------------------------------------------------------
class BasePINN(nn.Module):
    def __init__(self, hidden_dim=20, num_layers=4):
        super(BasePINN, self).__init__()
        layers = [nn.Linear(2, hidden_dim), nn.Tanh()]
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x, t):
        inputs = torch.cat([x, t], dim=1)
        return self.net(inputs)

# ---------------------------------------------------------
# 4. Soft vs Hard Constraint 모델 정의
# ---------------------------------------------------------
class SoftHeatPINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.pinn = BasePINN()

    def forward(self, x, t):
        return self.pinn(x, t)

class HardHeatPINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.pinn = BasePINN()

    def forward(self, x, t):
        # Ansatz: u(x,t) = sin(pi*x) + (1 - x^2) * t * NN(x,t)
        # - t = 0 일 때: sin(pi*x) (초기 조건 100% 보장)
        # - x = ±1 일 때: sin(±pi) + 0 = 0 (경계 조건 100% 보장)
        B = torch.sin(np.pi * x)
        envelope = (1.0 - x**2) * t
        return B + envelope * self.pinn(x, t)

# ---------------------------------------------------------
# 5. 열 방정식 PDE 잔차 계산 (Autograd)
# ---------------------------------------------------------
def compute_pde_residual(model, x, t, alpha_val):
    x_g = x.clone().detach().requires_grad_(True)
    t_g = t.clone().detach().requires_grad_(True)
    
    u = model(x_g, t_g)
    u_t = torch.autograd.grad(u, t_g, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_x = torch.autograd.grad(u, x_g, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x_g, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
    
    # 열 방정식 잔차: f = u_t - alpha * u_xx
    return u_t - alpha_val * u_xx

# ---------------------------------------------------------
# 6. 검증용 참값 격자 미리 계산
# ---------------------------------------------------------
x_val_np = np.linspace(-1.0, 1.0, 50)
t_val_np = np.linspace(0.0, 1.0, 20)
x_grid, t_grid = np.meshgrid(x_val_np, t_val_np)

x_eval_flat = x_grid.flatten()
t_eval_flat = t_grid.flatten()

u_exact_flat = heat_exact(x_eval_flat, t_eval_flat)
u_exact_tensor = torch.tensor(u_exact_flat, dtype=torch.float32).unsqueeze(1).to(device)

x_eval_tensor = torch.tensor(x_eval_flat, dtype=torch.float32).unsqueeze(1).to(device)
t_eval_tensor = torch.tensor(t_eval_flat, dtype=torch.float32).unsqueeze(1).to(device)

# ---------------------------------------------------------
# 7. Epoch별 트래킹 학습 루프
# ---------------------------------------------------------
def train_and_track(model_type='soft', epochs=4000, eval_every=40):
    # 학습 데이터 생성
    x_pde = (torch.rand(1500, 1) * 2.0 - 1.0).to(device)
    t_pde = torch.rand(1500, 1).to(device)
    
    t_bc = torch.rand(200, 1).to(device)
    x_bc = torch.cat([torch.full((100, 1), -1.0), torch.full((100, 1), 1.0)], dim=0).to(device)
    u_bc = torch.zeros_like(x_bc).to(device)
    
    x_ic = (torch.rand(200, 1) * 2.0 - 1.0).to(device)
    t_ic = torch.zeros((200, 1)).to(device)
    u_ic = torch.sin(np.pi * x_ic).to(device)
    
    model = SoftHeatPINN().to(device) if model_type == 'soft' else HardHeatPINN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    epoch_history = []
    l2_history = []
    
    model.train()
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        
        loss_pde = torch.mean(compute_pde_residual(model, x_pde, t_pde, alpha)**2)
        
        if model_type == 'soft':
            loss_bc = torch.mean((model(x_bc, t_bc) - u_bc)**2)
            loss_ic = torch.mean((model(x_ic, t_ic) - u_ic)**2)
            # 가중치 밸런싱
            total_loss = loss_pde + 20.0 * loss_bc + 20.0 * loss_ic
        else:
            total_loss = loss_pde
            
        total_loss.backward()
        optimizer.step()
        
        # L2 오차 트래킹
        if epoch % eval_every == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                u_pred_eval = model(x_eval_tensor, t_eval_tensor)
                rel_l2 = (torch.norm(u_pred_eval - u_exact_tensor, 2) / torch.norm(u_exact_tensor, 2)).item()
                
            epoch_history.append(epoch)
            l2_history.append(rel_l2)
            model.train()
            
    # 최종 경계 오차 측정
    model.eval()
    with torch.no_grad():
        bc_err = torch.mean(torch.abs(model(x_bc, t_bc) - u_bc)).item()
        
    return epoch_history, l2_history, bc_err

# ---------------------------------------------------------
# 8. 학습 실행 및 결과 시각화
# ---------------------------------------------------------
total_epochs = 4000
eval_step = 40

print("=== 1차원 열 방정식 PINN 실증 실험 시작 ===")
print(f"\n[1/2] Soft Constraint 모델 학습 중 ({total_epochs} Epochs)...")
soft_epochs, soft_l2, soft_bc_err = train_and_track('soft', epochs=total_epochs, eval_every=eval_step)

print(f"[2/2] Hard Constraint 모델 학습 중 ({total_epochs} Epochs)...")
hard_epochs, hard_l2, hard_bc_err = train_and_track('hard', epochs=total_epochs, eval_every=eval_step)

print("\n" + "="*58)
print(f"{'검증 항목 (Metric)':<22} | {'Soft Constraint':<12} | {'Hard Constraint':<12}")
print("="*58)
print(f"{'경계 오차 (Boundary Error)':<20} | {soft_bc_err:<15.2e} | {hard_bc_err:<15.2e}")
print(f"{'최종 상대 L2 오차':<21} | {soft_l2[-1]:<15.4f} | {hard_l2[-1]:<15.4f}")
print("="*58)

# ---------------------------------------------------------
# 9. Matplotlib 시각화 그래프
# ---------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 6), dpi=120)

ax.plot(soft_epochs, soft_l2, label='Soft Constraint', color='#d62728', linewidth=2, linestyle='--')
ax.plot(hard_epochs, hard_l2, label='Hard Constraint', color='#1f77b4', linewidth=2.5, linestyle='-')

ax.set_title('Relative L2 Error', fontsize=14, fontweight='bold', pad=12)
ax.set_xlabel('Epochs', fontsize=12, fontweight='bold')
ax.set_ylabel('Relative L2 Error', fontsize=12, fontweight='bold')

# 로그 스케일 적용 시 수렴 양상이 더 뚜렷함
ax.set_yscale('log')

ax.legend(fontsize=11, loc='upper right', frameon=True, facecolor='white', framealpha=0.9)
ax.grid(True, which='both', linestyle=':', alpha=0.6)

plt.tight_layout()
plt.savefig('heat_equation_pinn_l2_convergence.png', dpi=300)
print("\n그래프가 'heat_equation_pinn_l2_convergence.png' 파일로 저장되었습니다.")
plt.show()