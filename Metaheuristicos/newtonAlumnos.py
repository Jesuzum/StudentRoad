import numpy as np
import matplotlib.pyplot as plt

# ================================
# Función 1: f(x) = sin(2x)
# ================================
def f1(x):
    return np.sin(2*x)

def f1p(x):  # primera derivada
    return 2*np.cos(2*x)

def f1pp(x):  # segunda derivada
    return -4*np.sin(2*x)


# ================================
# Función 2: f(x) = sin(x) + x cos(x)
# ================================
def f2(x):
    return np.sin(x) + x*np.cos(x)

def f2p(x):  # primera derivada
    return 2*np.cos(x) - x*np.sin(x)

def f2pp(x):  # segunda derivada
    return -3*np.sin(x) - x*np.cos(x)


# ================================
# Método de Newton aplicado a derivadas
# ================================
def newton_extremos(fprima, fsegunda, x0, tol=1e-6, maxiter=20):
    xi = x0
    for i in range(maxiter):
        if fsegunda(xi) == 0:  # evitar división por cero
            break
        xi_new = xi - fprima(xi)/fsegunda(xi)
        if abs(xi_new - xi) < tol:  # convergencia
            xi = xi_new
            break
        xi = xi_new
    # Clasificación con la segunda derivada
    tipo = "mínimo" if fsegunda(xi) > 0 else "máximo"
    return xi, tipo


# ================================
# Graficar una función con sus derivadas
# ================================
def graficar(func, fprima, fsegunda, rango, titulo, puntos_iniciales):
    x = np.linspace(rango[0], rango[1], 500)
    plt.figure(figsize=(8,6))
    plt.plot(x, func(x), 'k-', label='f(x)')
    plt.plot(x, fprima(x), 'g-', label="f'(x)")
    plt.plot(x, fsegunda(x), 'b--', label="f''(x)")

    # Aplicamos Newton a cada valor inicial y marcamos en la gráfica
    for x0 in puntos_iniciales:
        raiz, tipo = newton_extremos(fprima, fsegunda, x0)
        plt.plot(raiz, fprima(raiz), 'r*', markersize=12)
        print(f"Con x0 = {x0}, se encontró raíz en {raiz:.4f} → {tipo}")

    plt.title(titulo)
    plt.legend()
    plt.grid(True)
    plt.savefig(titulo.replace(" ", "_") + ".eps", format="eps")  # para puntos extra
    plt.show()


# ================================
# Ejecutamos para las dos funciones
# ================================

print("\n=== Función 1: f(x) = sin(2x) ===")
graficar(f1, f1p, f1pp, [-4, 4], "Funcion 1: sin(2x)", [-3, -1, 1, 3])

print("\n=== Función 2: f(x) = sin(x) + x cos(x) ===")
graficar(f2, f2p, f2pp, [-5, 5], "Funcion 2: sin(x)+xcos(x)", [-4, -2, 0, 2, 4])
