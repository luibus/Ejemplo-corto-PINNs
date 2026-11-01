# Physics-Informed Neural Network (PINN): Oscilador Armónico Simple

Guía para la implementación de una Red Neuronal Informada en Física (PINN) en PyTorch para resolver una ecuación diferencial ordinaria (EDO) sin necesidad de datos experimentales etiquetados.

---

## 1. El Problema Físico y Matemático

El script resuelve el movimiento de un **oscilador armónico simple ideal** (un sistema masa-resorte sin amortiguamiento ni fuerzas externas):

$$y''(t) + y(t) = 0$$

Con las siguientes **condiciones iniciales** en $t_0 = 0$:
* **Posición inicial:** $y(0) = 0$
* **Velocidad inicial:** $y'(0) = 1$

La **solución analítica exacta** para este problema de valor inicial es:
$$y(t) = \sin(t)$$

---

## 2. ¿Qué es una PINN y cómo aprende?

A diferencia del Machine Learning supervisado clásico (que entrena comparando predicciones contra pares de datos etiquetados), una PINN aprende **incorporando las leyes de la física directamente en la función de pérdida**:

1. **Aproximador funcional:** La red recibe el tiempo $t$ como entrada y predice la posición escalar $y(t)$.
2. **Diferenciación automática (Autograd):** PyTorch calcula las derivadas exactas $\frac{dy}{dt}$ y $\frac{d^2y}{dt^2}$ respecto a la entrada $t$ recorriendo el grafo computacional.
3. **Pérdida residual:** Se penaliza a la red en la medida en que sus salidas y derivadas violen la ecuación diferencial o las condiciones iniciales.

---

## 3. Desglose del Código

### A. Arquitectura de la Red (`PINNOscilador`)
```python
self.net = nn.Sequential(
    nn.Linear(1, 20), nn.Tanh(),
    nn.Linear(20, 20), nn.Tanh(),
    nn.Linear(20, 1)
)
```
* **Entrada/Salida:** Entrada unidimensional ($t$) $\to$ salida unidimensional ($y$).
* **Capas intermedias:** 2 capas ocultas con 20 neuronas cada una.
* **Activación (`Tanh`):** No se utiliza `ReLU` porque su segunda derivada es nula en casi todo su dominio ($d^2/dt^2 = 0$). `Tanh` es suave, continua e infinitamente diferenciable ($C^\infty$), requisito indispensable para calcular derivadas de orden superior.

### B. Puntos de Colocación (Collocation Points)
```python
t = (torch.rand(100, 1) * 6.28).requires_grad_(True)
```
* En cada época se muestrean 100 puntos pseudoaleatorios en el intervalo $[0, 2\pi]$.
* `requires_grad_(True)` habilita el rastreo de gradientes sobre el tensor de tiempo $t$.

### C. Pérdida Física ($Loss_f$)
```python
dy_dt = torch.autograd.grad(y, t, torch.ones_like(y), create_graph=True)[0]
d2y_dt2 = torch.autograd.grad(dy_dt, t, torch.ones_like(dy_dt), create_graph=True)[0]
loss_f = torch.mean((d2y_dt2 + y)**2)
```
* El argumento `create_graph=True` construye el grafo de la primera derivada para permitir calcular la segunda derivada ($y''$).
* Si la red satisface la dinámica física, $y'' + y \to 0$, minimizando este término cuadrático medio.

### D. Pérdida por Condiciones Iniciales ($Loss_{ic}$)
```python
y0 = model(t0)
dy0_dt = torch.autograd.grad(y0, t0, torch.ones_like(y0), create_graph=True)[0]
loss_ic = (y0 - 0.0)**2 + (dy0_dt - 1.0)**2
```
* Evalúa y penaliza las desviaciones puntuales en $t=0$: asegura $y(0) = 0$ y $y'(0) = 1$.
* Sin este término, la red convergería a la solución trivial $y(t) = 0$.

### E. Optimización
```python
loss = loss_f + loss_ic
loss.backward()
optimizer.step()
```
* La pérdida total suma ambas restricciones. El optimizador Adam ajusta los pesos mediante descenso de gradiente.

---

## 4. Instalación y Ejecución

### Requisitos previos
Instala las dependencias en tu entorno de Python:
```bash
pip install torch matplotlib
```

### Ejecución
Guarda el código en un archivo (por ejemplo, `pinn_oscilador.py`) y ejecútalo:
```bash
python pinn_oscilador.py
```

---

## 5. Interpretación de Salidas y Gráficas

* **Épocas 0–400:** La pérdida global desciende con rapidez; el modelo fija prioritariamente las condiciones iniciales en $t=0$.
* **Épocas 400–2000:** La pérdida residual física decae por debajo de $10^{-4}$, alcanzando la dinámica oscilatoria periódica.
* **Curva de posición:** La predicción (línea punteada roja) reproduce la trayectoria de $\sin(t)$ (azul continuo) en todo el intervalo entrenado $[0, 2\pi]$.
* **Error absoluto:** Se mantiene en órdenes de $10^{-2}$ o inferiores dentro del dominio de colocación. Se observa un ligero incremento cerca de $t \approx 7$, correspondiente a la extrapolación fuera del rango de entrenamiento ($6.28 \approx 2\pi$).

---

## 6. Retos Propuestos

1. **Oscilador Amortiguado:** Incorpora fricción modificando el residuo físico a:
   $$y'' + 0.2y' + y = 0$$
   *¿Cómo se adapta la red a la pérdida paulatina de amplitud?*
2. **Ponderación de Pérdidas:** Modifica la función objetivo usando coeficientes:
   $$\mathcal{L} = \mathcal{L}_f + \lambda_{ic}\mathcal{L}_{ic} \quad (\text{con } \lambda_{ic} = 10)$$
   *Compara la velocidad de convergencia y el error en el origen.*
3. **Extensión del Dominio:** Modifica el muestreo para cubrir el intervalo $[0, 4\pi]$ y analiza si 20 neuronas por capa siguen siendo suficientes para mantener la precisión.
