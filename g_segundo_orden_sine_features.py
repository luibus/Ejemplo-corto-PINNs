import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# ==============================================================================
# PARÁMETROS MULTIESCALA Y SIREN
# ==============================================================================
EPSILON = 0.1  # Escala de heterogeneidad (prueba 0.1, 0.05)
W0 = 2.0       # Frecuencia moderada en capas Seno (Fourier ya inyecta 2*pi/eps)

def get_A(x, eps):
    A_inv = 1.0 + 0.5 * torch.cos(2 * torch.pi * x / eps)
    return 1.0 / A_inv

def solucion_exacta(x, eps):
    w = torch.tensor(2 * torch.pi / eps, dtype=x.dtype)
    c1 = (0.5 + 0.5 * torch.sin(w) / w - 0.5 * (1.0 - torch.cos(w)) / (w**2)) / (1.0 + 0.5 * torch.sin(w) / w)
    osc = (-x + c1) / w * torch.sin(w * x) + (1.0 - torch.cos(w * x)) / (w**2)
    return -0.5 * (x**2) + c1 * x + 0.5 * osc

# ==============================================================================
# 1. FOURIER FEATURES REPRESENTATIVAS + RED CON ACTIVACIÓN SENO
# ==============================================================================
class CapaFourierFeatures(nn.Module):
    def __init__(self, eps):
        super().__init__()
        # Frecuencias representativas:
        # - Macroescala (tendencia global suave): f = 1.0, 2.0
        # - Microescala (heterogeneidad del material): f = 1.0 / eps
        freqs = torch.tensor([[1.0, 2.0, 1.0 / eps]], dtype=torch.float32)
        self.register_buffer('w', 2.0 * torch.pi * freqs)

    def forward(self, x):
        # Escalamos por 1/w para que la derivada primera tenga amplitud O(1)
        wx = x @ self.w
        sin_f = torch.sin(wx) / self.w
        cos_f = torch.cos(wx) / self.w
        # Salida de dimensión: 1 (x) + 3 (sin) + 3 (cos) = 7 características
        return torch.cat([x, sin_f, cos_f], dim=-1)

class CapaSeno(nn.Module):
    def __init__(self, in_features, out_features, es_primera=False, w0=2.0):
        super().__init__()
        self.w0 = w0
        self.linear = nn.Linear(in_features, out_features)
        
        with torch.no_grad():
            if es_primera:
                limite = 1.0 / in_features
            else:
                limite = (6.0 / in_features)**0.5 / w0
            self.linear.weight.uniform_(-limite, limite)
            self.linear.bias.uniform_(-limite, limite)

    def forward(self, x):
        return torch.sin(self.w0 * self.linear(x))

class PINNSenoFourier(nn.Module):
    def __init__(self, eps, w0=2.0):
        super().__init__()
        self.fourier = CapaFourierFeatures(eps)
        self.net = nn.Sequential(
            CapaSeno(7, 30, es_primera=True, w0=w0),
            CapaSeno(30, 30, es_primera=False, w0=w0),
            nn.Linear(30, 1)
        )
        with torch.no_grad():
            limite_out = (6.0 / 30)**0.5 / w0
            self.net[-1].weight.uniform_(-limite_out, limite_out)
            self.net[-1].bias.zero_()

    def forward(self, x):
        feat = self.fourier(x)
        return self.net(feat)

model = PINNSenoFourier(eps=EPSILON, w0=W0)
optimizer = torch.optim.Adam(model.parameters(), lr=0.002)

# Puntos de contorno: x = 0 y x = 1
x_bc = torch.tensor([[0.0], [1.0]])

# ==============================================================================
# 2. ENTRENAMIENTO CON MUESTREO ALEATORIO
# ==============================================================================
print(f"Iniciando entrenamiento PINN 2º Orden (Fourier + Seno) | eps = {EPSILON}...")

for epoch in range(3001):
    optimizer.zero_grad()

    # Muestreo aleatorio de 300 puntos en (0, 1)
    x = torch.rand(300, 1, requires_grad=True)

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

    # Evaluamos también la derivada u'(x) para comprobar las micro-oscilaciones
    x_grad = torch.linspace(0, 1, 400).view(-1, 1).requires_grad_(True)
    with torch.enable_grad():
        u_g = model(x_grad)
        du_pred = torch.autograd.grad(u_g, x_grad, torch.ones_like(u_g))[0].detach()
    w = torch.tensor(2 * torch.pi / EPSILON)
    c1 = (0.5 + 0.5 * torch.sin(w) / w - 0.5 * (1.0 - torch.cos(w)) / (w**2)) / (1.0 + 0.5 * torch.sin(w) / w)
    du_real = (1.0 + 0.5 * torch.cos(w * x_plot)) * (-x_plot + c1)

    fig, axs = plt.subplots(1, 2, figsize=(12, 4))

    axs[0].plot(x_plot.numpy(), u_real.numpy(), 'b-', label="u Exacta", linewidth=2)
    axs[0].plot(x_plot.numpy(), u_pred.numpy(), 'r--', label="u Predicha (Fourier+Seno)", linewidth=2)
    axs[0].set_title(f"Solución u(x) - 2º Orden [eps = {EPSILON}]")
    axs[0].set_xlabel("Espacio (x)")
    axs[0].set_ylabel("u(x)")
    axs[0].legend()
    axs[0].grid(True, linestyle=':', alpha=0.7)

    axs[1].plot(x_plot.numpy(), du_real.numpy(), 'b-', label="u' Exacta", linewidth=1.5)
    axs[1].plot(x_plot.numpy(), du_pred.numpy(), 'r--', label="u' Predicha (Fourier+Seno)", linewidth=1.5)
    axs[1].set_title("Gradiente u'(x) [Captura de Microescala]")
    axs[1].set_xlabel("Espacio (x)")
    axs[1].set_ylabel("du/dx")
    axs[1].legend()
    axs[1].grid(True, linestyle=':', alpha=0.7)

    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 3))
    plt.plot(x_plot.numpy(), torch.abs(u_real - u_pred).numpy(), color='purple', linewidth=2)
    plt.xlabel("Espacio (x)")
    plt.ylabel("Error Absoluto")
    plt.title("Error Absoluto en u(x) (2º Orden con Fourier Features + Seno)")
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.tight_layout()
    plt.show()
