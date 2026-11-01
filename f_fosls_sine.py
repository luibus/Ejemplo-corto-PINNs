import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# ==============================================================================
# PARÁMETROS MULTIESCALA Y DE FRECUENCIA SIREN
# ==============================================================================
EPSILON = 0.1  # Prueba 0.1, 0.05 o 0.02
W0_U = 15.0    # Frecuencia alta para u(x) (captura oscilaciones multiescala)
W0_J = 1.0     # Frecuencia suave para el flujo j(x) (es lineal)

def get_A_inv(x, eps):
    return 1.0 + 0.5 * torch.cos(2 * torch.pi * x / eps)

def solucion_exacta(x, eps):
    w = torch.tensor(2 * torch.pi / eps, dtype=x.dtype)
    c1 = (0.5 + 0.5 * torch.sin(w) / w - 0.5 * (1.0 - torch.cos(w)) / (w**2)) / (1.0 + 0.5 * torch.sin(w) / w)
    osc = (-x + c1) / w * torch.sin(w * x) + (1.0 - torch.cos(w * x)) / (w**2)
    u_exact = -0.5 * (x**2) + c1 * x + 0.5 * osc
    j_exact = -x + c1
    return u_exact, j_exact

# ==============================================================================
# 1. DEFINICIÓN DE REDES CON ACTIVACIÓN SENO (SIREN)
# ==============================================================================
class CapaSeno(nn.Module):
    def __init__(self, in_features, out_features, es_primera=False, w0=10.0):
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

class RedSeno(nn.Module):
    def __init__(self, w0=10.0):
        super().__init__()
        self.net = nn.Sequential(
            CapaSeno(1, 30, es_primera=True, w0=w0),
            CapaSeno(30, 30, es_primera=False, w0=w0),
            nn.Linear(30, 1)
        )
        with torch.no_grad():
            limite_out = (6.0 / 30)**0.5 / w0
            self.net[-1].weight.uniform_(-limite_out, limite_out)
            self.net[-1].bias.zero_()

    def forward(self, x):
        return self.net(x)

# Instanciamos cada red con su frecuencia óptima
model_u = RedSeno(w0=W0_U)
model_j = RedSeno(w0=W0_J)

optimizer = torch.optim.Adam(
    list(model_u.parameters()) + list(model_j.parameters()),
    lr=0.002
)

# Condiciones de contorno para u: x = 0 y x = 1
x_bc = torch.tensor([[0.0], [1.0]])

# ==============================================================================
# 2. ENTRENAMIENTO FOSLS CON MUESTREO ALEATORIO
# ==============================================================================
print(f"Iniciando entrenamiento FOSLS-Seno | eps = {EPSILON}, w0_u = {W0_U}...")

for epoch in range(3001):
    optimizer.zero_grad()

    # Muestreo aleatorio de 250 puntos en (0, 1)
    x = torch.rand(250, 1, requires_grad=True)

    u = model_u(x)
    j = model_j(x)

    # Derivadas exclusivamente de PRIMER ORDEN
    du_dx = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
    dj_dx = torch.autograd.grad(j, x, torch.ones_like(j), create_graph=True)[0]

    # --- A) PÉRDIDA FÍSICA FOSLS ---
    # 1) Relación constitutiva: u' - A_inv(x) * j = 0
    res_flujo = du_dx - get_A_inv(x, EPSILON) * j
    # 2) Conservación: -j' = 1  =>  j' + 1 = 0
    res_conservacion = dj_dx + 1.0

    loss_f = torch.mean(res_flujo**2) + torch.mean(res_conservacion**2)

    # --- B) CONDICIONES DE CONTORNO: u(0) = 0, u(1) = 0 ---
    u_bc = model_u(x_bc)
    loss_bc = torch.mean(u_bc**2)

    # --- C) OPTIMIZACIÓN ---
    loss = loss_f + 50.0 * loss_bc
    loss.backward()
    optimizer.step()

    if epoch % 500 == 0:
        print(f"Época {epoch:4d} | Pérdida Total: {loss.item():.5f} (FOSLS: {loss_f.item():.5f} | BC: {loss_bc.item():.5f})")

# ==============================================================================
# 3. RESULTADOS Y GRÁFICAS (CAPTURA DE MICRO-OSCILACIONES)
# ==============================================================================
with torch.no_grad():
    x_plot = torch.linspace(0, 1, 400).view(-1, 1)
    u_pred = model_u(x_plot)
    j_pred = model_j(x_plot)
    u_real, j_real = solucion_exacta(x_plot, EPSILON)

    # Calculamos también la derivada u'(x) para visualizar las oscilaciones de escala epsilon
    x_grad = torch.linspace(0, 1, 400).view(-1, 1).requires_grad_(True)
    with torch.enable_grad():
        u_g = model_u(x_grad)
        du_pred = torch.autograd.grad(u_g, x_grad, torch.ones_like(u_g))[0].detach()
    du_real = get_A_inv(x_plot, EPSILON) * j_real

    fig, axs = plt.subplots(1, 3, figsize=(15, 4))

    # Gráfica 1: Solución u(x)
    axs[0].plot(x_plot.numpy(), u_real.numpy(), 'b-', label="u Exacta", linewidth=2)
    axs[0].plot(x_plot.numpy(), u_pred.numpy(), 'r--', label="u Predicha (FOSLS-Seno)", linewidth=2)
    axs[0].set_title(f"Campo u(x) [eps = {EPSILON}]")
    axs[0].set_xlabel("Espacio (x)")
    axs[0].set_ylabel("u(x)")
    axs[0].legend()
    axs[0].grid(True, linestyle=':', alpha=0.7)

    # Gráfica 2: Derivada u'(x) (donde se aprecia la rugosidad multiescala)
    axs[1].plot(x_plot.numpy(), du_real.numpy(), 'b-', label="u' Exacta", linewidth=1.5)
    axs[1].plot(x_plot.numpy(), du_pred.numpy(), 'r--', label="u' Predicha (FOSLS-Seno)", linewidth=1.5)
    axs[1].set_title("Gradiente u'(x) [Micro-oscilaciones]")
    axs[1].set_xlabel("Espacio (x)")
    axs[1].set_ylabel("du/dx")
    axs[1].legend()
    axs[1].grid(True, linestyle=':', alpha=0.7)

    # Gráfica 3: Flujo j(x)
    axs[2].plot(x_plot.numpy(), j_real.numpy(), 'g-', label="Flujo j Exacto", linewidth=2)
    axs[2].plot(x_plot.numpy(), j_pred.numpy(), 'm--', label="Flujo j Predicho", linewidth=2)
    axs[2].set_title("Flujo j(x) = A_eps(x) * u'(x)")
    axs[2].set_xlabel("Espacio (x)")
    axs[2].set_ylabel("j(x)")
    axs[2].legend()
    axs[2].grid(True, linestyle=':', alpha=0.7)

    plt.tight_layout()
    plt.show()

    # Gráfica 4: Error absoluto en u(x)
    plt.figure(figsize=(8, 3))
    plt.plot(x_plot.numpy(), torch.abs(u_real - u_pred).numpy(), color='purple', linewidth=2)
    plt.xlabel("Espacio (x)")
    plt.ylabel("Error Absoluto")
    plt.title("Error Absoluto en u(x) (FOSLS con Activación Seno)")
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.tight_layout()
    plt.show()
