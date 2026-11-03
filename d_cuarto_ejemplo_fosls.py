import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# ==============================================================================
# PARÁMETRO MULTIESCALA
# ==============================================================================
EPSILON = 0.5  

def get_A_inv(x, eps):
    return 1.0 + 0.5 * torch.cos(2 * torch.pi * x / eps)

def solucion_exacta(x, eps):
    # Convertimos w a Tensor para evitar el TypeError en torch.sin / torch.cos
    w = torch.tensor(2 * torch.pi / eps, dtype=x.dtype)
    c1 = (0.5 + 0.5 * torch.sin(w) / w - 0.5 * (1.0 - torch.cos(w)) / (w**2)) / (1.0 + 0.5 * torch.sin(w) / w)
    osc = (-x + c1) / w * torch.sin(w * x) + (1.0 - torch.cos(w * x)) / (w**2)
    u_exact = -0.5 * (x**2) + c1 * x + 0.5 * osc
    j_exact = -x + c1
    return u_exact, j_exact

# ==============================================================================
# 1. DEFINICIÓN DE LAS REDES (Red principal u y Red auxiliar de flujo j)
# ==============================================================================
class RedBase(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 30), nn.Tanh(),
            nn.Linear(30, 30), nn.Tanh(),
            nn.Linear(30, 1)
        )
    def forward(self, x):
        return self.net(x)

model_u = RedBase()
model_j = RedBase()

optimizer = torch.optim.Adam(
    list(model_u.parameters()) + list(model_j.parameters()),
    lr=0.005
)

# Condiciones de contorno para u: x = 0 y x = 1
x_bc = torch.tensor([[0.0], [1.0]])

# ==============================================================================
# 2. ENTRENAMIENTO FOSLS CON MUESTREO ALEATORIO
# ==============================================================================
print(f"Iniciando entrenamiento FOSLS con epsilon = {EPSILON}...")

for epoch in range(5001):
    optimizer.zero_grad()

    # Muestreo aleatorio de 200 puntos en (0, 1)
    x = torch.rand(200, 1, requires_grad=True)

    u = model_u(x)
    j = model_j(x)

    # Derivadas exclusivamente de PRIMER ORDEN
    du_dx = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
    dj_dx = torch.autograd.grad(j, x, torch.ones_like(j), create_graph=True)[0]

    # --- A) PÉRDIDA FÍSICA FOSLS ---
    # 1) Ecuación constitutiva: u' - A_inv(x) * j = 0 (sin derivar A_eps)
    res_flujo = du_dx - get_A_inv(x, EPSILON) * j
    # 2) Ecuación de conservación: -j' = 1  =>  j' + 1 = 0
    res_conservacion = dj_dx + 1.0

    loss_f = torch.mean(res_flujo**2) + torch.mean(res_conservacion**2)

    # --- B) CONDICIONES DE CONTORNO: u(0) = 0, u(1) = 0 ---
    u_bc = model_u(x_bc)
    loss_bc = torch.mean(u_bc**2)

    # --- C) OPTIMIZACIÓN ---
    loss = loss_f + 20.0 * loss_bc
    loss.backward()
    optimizer.step()

    if epoch % 500 == 0:
        print(f"Época {epoch:4d} | Pérdida Total: {loss.item():.5f} (FOSLS: {loss_f.item():.5f} | BC: {loss_bc.item():.5f})")

# ==============================================================================
# 3. RESULTADOS Y GRÁFICAS
# ==============================================================================
with torch.no_grad():
    x_plot = torch.linspace(0, 1, 400).view(-1, 1)
    u_pred = model_u(x_plot)
    j_pred = model_j(x_plot)
    u_real, j_real = solucion_exacta(x_plot, EPSILON)

    fig, axs = plt.subplots(1, 2, figsize=(12, 4))

    # Gráfica 1: Solución u(x)
    axs[0].plot(x_plot.numpy(), u_real.numpy(), 'b-', label="u Exacta", linewidth=2)
    axs[0].plot(x_plot.numpy(), u_pred.numpy(), 'r--', label="u Predicha (FOSLS)", linewidth=2)
    axs[0].set_title(f"Campo u(x) [eps = {EPSILON}]")
    axs[0].set_xlabel("Espacio (x)")
    axs[0].set_ylabel("u(x)")
    axs[0].legend()
    axs[0].grid(True, linestyle=':', alpha=0.7)

    # Gráfica 2: Flujo j(x)
    axs[1].plot(x_plot.numpy(), j_real.numpy(), 'g-', label="Flujo j Exacto", linewidth=2)
    axs[1].plot(x_plot.numpy(), j_pred.numpy(), 'm--', label="Flujo j Predicho (FOSLS)", linewidth=2)
    axs[1].set_title("Flujo j(x) = A_eps(x) * u'(x)")
    axs[1].set_xlabel("Espacio (x)")
    axs[1].set_ylabel("j(x)")
    axs[1].legend()
    axs[1].grid(True, linestyle=':', alpha=0.7)

    plt.tight_layout()
    plt.show()

    # Gráfica 3: Error absoluto en u(x)
    plt.figure(figsize=(8, 3))
    plt.plot(x_plot.numpy(), torch.abs(u_real - u_pred).numpy(), color='purple', linewidth=2)
    plt.xlabel("Espacio (x)")
    plt.ylabel("Error Absoluto")
    plt.title("Error Absoluto en u(x) (FOSLS)")
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.tight_layout()
    plt.show()
