import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# ==============================================================================
# PARÁMETROS MULTIESCALA Y DE FRECUENCIA SIREN
# ==============================================================================
EPSILON = 0.1  # Escala de heterogeneidad del material (prueba 0.5, 0.1, 0.05)
W0 = 10.0      # Frecuencia base de la activación Seno (SIREN)

def get_A(x, eps):
    A_inv = 1.0 + 0.5 * torch.cos(2 * torch.pi * x / eps)
    return 1.0 / A_inv

def solucion_exacta(x, eps):
    w = torch.tensor(2 * torch.pi / eps, dtype=x.dtype)
    c1 = (0.5 + 0.5 * torch.sin(w) / w - 0.5 * (1.0 - torch.cos(w)) / (w**2)) / (1.0 + 0.5 * torch.sin(w) / w)
    osc = (-x + c1) / w * torch.sin(w * x) + (1.0 - torch.cos(w * x)) / (w**2)
    return -0.5 * (x**2) + c1 * x + 0.5 * osc

# ==============================================================================
# 1. DEFINICIÓN DE CAPA Y RED CON ACTIVACIÓN SENO (CUIDADO ESPECIAL SIREN)
# ==============================================================================
class CapaSeno(nn.Module):
    def __init__(self, in_features, out_features, es_primera=False, w0=10.0):
        super().__init__()
        self.w0 = w0
        self.linear = nn.Linear(in_features, out_features)
        
        # Inicialización especial SIREN para estabilidad en las derivadas
        with torch.no_grad():
            if es_primera:
                limite = 1.0 / in_features
            else:
                limite = (6.0 / in_features)**0.5 / w0
            self.linear.weight.uniform_(-limite, limite)
            self.linear.bias.uniform_(-limite, limite)

    def forward(self, x):
        return torch.sin(self.w0 * self.linear(x))

class PINNSeno(nn.Module):
    def __init__(self, w0=10.0):
        super().__init__()
        self.net = nn.Sequential(
            CapaSeno(1, 30, es_primera=True, w0=w0),
            CapaSeno(30, 30, es_primera=False, w0=w0),
            nn.Linear(30, 1)
        )
        # Escalar la capa de salida evita que u''(x) explote en la época 0
        with torch.no_grad():
            limite_out = (6.0 / 30)**0.5 / w0
            self.net[-1].weight.uniform_(-limite_out, limite_out)
            self.net[-1].bias.zero_()

    def forward(self, x):
        return self.net(x)

model = PINNSeno(w0=W0)
optimizer = torch.optim.Adam(model.parameters(), lr=0.002)

# Puntos de contorno: x = 0 y x = 1
x_bc = torch.tensor([[0.0], [1.0]])

# ==============================================================================
# 2. ENTRENAMIENTO CON MUESTREO ALEATORIO
# ==============================================================================
print(f"Iniciando entrenamiento PINN-Seno (2º Orden) | eps = {EPSILON}, w0 = {W0}...")

for epoch in range(3001):
    optimizer.zero_grad()

    # Muestreo aleatorio de 250 puntos en (0, 1)
    x = torch.rand(250, 1, requires_grad=True)

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
    loss = loss_f + 50.0 * loss_bc
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
    plt.plot(x_plot.numpy(), u_pred.numpy(), 'r--', label=f"PINN Seno 2º Orden (eps={EPSILON})", linewidth=2)
    plt.xlabel("Espacio (x)")
    plt.ylabel("u(x)")
    plt.title(f"PINN 2º Orden con Activación Seno (eps = {EPSILON}, w0 = {W0})")
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 3))
    plt.plot(x_plot.numpy(), torch.abs(u_real - u_pred).numpy(), color='purple', linewidth=2)
    plt.xlabel("Espacio (x)")
    plt.ylabel("Error Absoluto")
    plt.title("Error Absoluto (PINN Seno 2º Orden)")
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.tight_layout()
    plt.show()
