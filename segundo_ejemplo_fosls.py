import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# ==============================================================================
# 1. DEFINICIÓN DE LAS REDES NEURONALES (FOSLS)
# Sistema de 1er orden:
#   (1) dy/dt - j = 0   (Definición del flujo / velocidad)
#   (2) dj/dt + y = 0   (Ecuación de movimiento)
# ==============================================================================

class RedBase(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 20), nn.Tanh(),
            nn.Linear(20, 20), nn.Tanh(),
            nn.Linear(20, 1)
        )
    def forward(self, t):
        return self.net(t)

# Red principal para la posición y(t) y red auxiliar para el flujo j(t) = y'(t)
model_y = RedBase()
model_j = RedBase()

# El optimizador ajusta los parámetros de ambas redes en paralelo
optimizer = torch.optim.Adam(
    list(model_y.parameters()) + list(model_j.parameters()), 
    lr=0.01
)

# Punto de condición inicial (t0 = 0)
t0 = torch.tensor([[0.0]], requires_grad=True)

# ==============================================================================
# 2. ENTRENAMIENTO CON MUESTREO ALEATORIO (FOSLS)
# ==============================================================================
print("Iniciando entrenamiento FOSLS...")

for epoch in range(2000):
    optimizer.zero_grad()

    # Muestreo aleatorio de 100 puntos en [0, 2*pi]
    t = (torch.rand(100, 1) * 6.28).requires_grad_(True)

    # Evaluación de ambas redes
    y = model_y(t)
    j = model_j(t)

    # Derivadas de PRIMER ORDEN (FOSLS evita calcular segundas derivadas)
    dy_dt = torch.autograd.grad(y, t, torch.ones_like(y), create_graph=True)[0]
    dj_dt = torch.autograd.grad(j, t, torch.ones_like(j), create_graph=True)[0]

    # --- A) PÉRDIDA FÍSICA FOSLS (Residuos L2 del sistema acoplado) ---
    res_flujo = dy_dt - j      # Ecuación constitutiva: y' = j
    res_momento = dj_dt + y    # Ecuación de movimiento: j' + y = 0
    loss_f = torch.mean(res_flujo**2) + torch.mean(res_momento**2)

    # --- B) CONDICIONES INICIALES: y(0) = 0, j(0) = 1 ---
    # Al tener la red auxiliar j, la condición de velocidad j(0)=1 es puramente algebraica
    y0 = model_y(t0)
    j0 = model_j(t0)
    loss_ic = (y0 - 0.0)**2 + (j0 - 1.0)**2

    # --- C) OPTIMIZACIÓN ---
    loss = loss_f + loss_ic
    loss.backward()
    optimizer.step()

    # Diagnóstico en pantalla
    if epoch % 200 == 0:
        print(f"Época {epoch:4d} | Pérdida Total: {loss.item():.5f} "
              f"(FOSLS: {loss_f.item():.5f} | ICs: {loss_ic.item():.5f})")

# ==============================================================================
# 3. EVALUACIÓN Y GRÁFICAS DE RESULTADOS
# ==============================================================================
with torch.no_grad():
    t_plot = torch.linspace(0, 7, 100).view(-1, 1)
    
    # Predicciones de ambas redes
    y_pred = model_y(t_plot)
    j_pred = model_j(t_plot)

    # Soluciones exactas: y(t) = sin(t), j(t) = y'(t) = cos(t)
    y_real = torch.sin(t_plot)
    j_real = torch.cos(t_plot)

    # Gráfica 1: Comparativa de posición y flujo (velocidad)
    plt.figure(figsize=(9, 4.5))
    plt.plot(t_plot.numpy(), y_real.numpy(), 'b-', label="Exacta y(t) = sin(t)", linewidth=2)
    plt.plot(t_plot.numpy(), y_pred.numpy(), 'r--', label="PINN y(t) [Red Principal]", linewidth=2)
    plt.plot(t_plot.numpy(), j_real.numpy(), 'g-', label="Exacta j(t) = cos(t)", linewidth=2, alpha=0.7)
    plt.plot(t_plot.numpy(), j_pred.numpy(), 'm:', label="PINN j(t) [Red Auxiliar Flujo]", linewidth=2)
    plt.xlabel("Tiempo (t)")
    plt.ylabel("Amplitud")
    plt.title("FOSLS-PINN: Solución Simultánea de Posición y Flujo")
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.tight_layout()
    plt.show()

    # Gráfica 2: Errores absolutos
    plt.figure(figsize=(9, 3.5))
    plt.plot(t_plot.numpy(), torch.abs(y_real - y_pred).numpy(), 'purple', label="Error en y(t)", linewidth=2)
    plt.plot(t_plot.numpy(), torch.abs(j_real - j_pred).numpy(), 'orange', label="Error en j(t)", linewidth=2)
    plt.xlabel("Tiempo (t)")
    plt.ylabel("Error Absoluto")
    plt.title("FOSLS-PINN: Error Absoluto en Ambas Variables")
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.tight_layout()
    plt.show()
