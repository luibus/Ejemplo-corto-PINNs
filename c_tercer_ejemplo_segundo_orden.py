import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# ==============================================================================
# PARÁMETRO MULTIESCALA
# Nota: En 2º orden converge bien para EPSILON = 0.5, pero colapsa a u=0 
# (Loss_f = 1.0) para EPSILON <= 0.1 debido a la derivada de A_eps(x).
# ==============================================================================
EPSILON = 0.5  # Prueba a cambiarlo a 0.5 para ver cómo converge el de 2º orden

def get_A(x, eps):
    A_inv = 1.0 + 0.5 * torch.cos(2 * torch.pi * x / eps)
    return 1.0 / A_inv

def solucion_exacta(x, eps):
    # Convertimos w a Tensor para que torch.sin y torch.cos no den TypeError
    w = torch.tensor(2 * torch.pi / eps, dtype=x.dtype)
    c1 = (0.5 + 0.5 * torch.sin(w) / w - 0.5 * (1.0 - torch.cos(w)) / (w**2)) / (1.0 + 0.5 * torch.sin(w) / w)
    osc = (-x + c1) / w * torch.sin(w * x) + (1.0 - torch.cos(w * x)) / (w**2)
    return -0.5 * (x**2) + c1 * x + 0.5 * osc

# ==============================================================================
# 1. DEFINICIÓN DE LA RED NEURONAL
# ==============================================================================
class PINNMultiescala(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 30), nn.Tanh(),
            nn.Linear(30, 30), nn.Tanh(),
            nn.Linear(30, 1)
        )
    def forward(self, x):
        return self.net(x)

model = PINNMultiescala()
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

# Condiciones de contorno: x = 0 y x = 1
x_bc = torch.tensor([[0.0], [1.0]])

# ==============================================================================
# 2. ENTRENAMIENTO CON MUESTREO ALEATORIO
# ==============================================================================
print(f"Iniciando entrenamiento PINN (2º Orden) con epsilon = {EPSILON}...")

for epoch in range(5001):
    optimizer.zero_grad()

    # Muestreo aleatorio de 200 puntos en (0, 1)
    x = torch.rand(200, 1, requires_grad=True)

    # --- A) PÉRDIDA FÍSICA: -(A_eps(x) * u')' = 1 ---
    u = model(x)
    du_dx = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
    
    flujo = get_A(x, EPSILON) * du_dx
    dflujo_dx = torch.autograd.grad(flujo, x, torch.ones_like(flujo), create_graph=True)[0]
    
    loss_f = torch.mean((-dflujo_dx - 1.0)**2)

    # --- B) CONDICIONES DE CONTORNO: u(0) = 0, u(1) = 0 ---
    u_bc = model(x_bc)
    loss_bc = torch.mean(u_bc**2)

    # --- C) OPTIMIZACIÓN ---
    loss = loss_f + 20.0 * loss_bc
    loss.backward()
    optimizer.step()

    if epoch % 500 == 0:
        print(f"Época {epoch:4d} | Pérdida Total: {loss.item():.5f} (Física: {loss_f.item():.5f} | BC: {loss_bc.item():.5f})")

# ==============================================================================
# 3. RESULTADOS Y GRÁFICAS
# ==============================================================================
with torch.no_grad():
    x_plot = torch.linspace(0, 1, 400).view(-1, 1)
    u_pred = model(x_plot)
    u_real = solucion_exacta(x_plot, EPSILON)

    plt.figure(figsize=(8, 4))
    plt.plot(x_plot.numpy(), u_real.numpy(), 'b-', label="Solución Exacta", linewidth=2)
    plt.plot(x_plot.numpy(), u_pred.numpy(), 'r--', label=f"PINN 2º Orden (eps={EPSILON})", linewidth=2)
    plt.xlabel("Espacio (x)")
    plt.ylabel("u(x)")
    plt.title(f"PINN 2º Orden: Problema Elíptico Multiescala (eps = {EPSILON})")
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 3))
    plt.plot(x_plot.numpy(), torch.abs(u_real - u_pred).numpy(), color='purple', linewidth=2)
    plt.xlabel("Espacio (x)")
    plt.ylabel("Error Absoluto")
    plt.title("Error Absoluto de la PINN (2º Orden)")
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.tight_layout()
    plt.show()
