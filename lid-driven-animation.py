import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from PIL import Image

# Create a list to store the individual frames
frames = []


N_GRIDPOINTS = 41                       # Anzahl Gitterpunkte
DOMAIN_SIZE = 1.                        # Länge des Gebietes
N_ITERATIONS = 5000                     # Anzahl Zeitschritte
TIME_STEP_LENGTH = 0.001                # Länge des Zeitschrittes, aufgrund der CFL-Bedingung und der Stabilität
DENSITY = 1.                            # Dichte
KINEMATIC_VISCOSITY = 0.01              # kinematische Viskosität
HORIZONTAL_VELOCITY_TOP = 1.            # Geschwindigkeit oben (Deckel bzw. Wand)

N_PRESSURE_ITERATIONS = 100             # Anzahl Iterationen für den Druck
STABILITY_SAFETY_FACTOR = 0.5           # Sicherheitsfaktor für die Stabilität
ELEMENT_LENGTH = DOMAIN_SIZE / (N_GRIDPOINTS - 1)   # Länge eines Elements

def central_difference_x(f):
    diff = np.zeros_like(f)
    diff[1:-1, 1:-1] = (f[1:-1, 2:]-f[1:-1, 0:-2]) / (2 * ELEMENT_LENGTH)
    return diff

def central_difference_y(f):
    diff = np.zeros_like(f)
    diff[1:-1, 1:-1] = (f[2:, 1:-1]-f[0:-2, 1:-1]) / (2 * ELEMENT_LENGTH)
    return diff

def laplace(f):
    diff = np.zeros_like(f)
    diff[1:-1, 1:-1] = (f[1:-1, 0:-2]+f[0:-2, 1:-1]-4*f[1:-1, 1:-1]+f[1:-1, 2:]+f[2:, 1:-1]) / (ELEMENT_LENGTH**2)
    return diff

def enforce_boundary_conditions(u, v):
    # Enforce boundary conditions for u
    u[0, :] = 0.0   # bottom boundary
    u[-1, :] = HORIZONTAL_VELOCITY_TOP  # top boundary
    u[:, 0] = 0.0   # left boundary
    u[:, -1] = 0.0  # right boundary
    
    # Enforce boundary conditions for v
    v[0, :] = 0.0   # bottom boundary
    v[-1, :] = 0.0  # top boundary
    v[:, 0] = 0.0   # left boundary
    v[:, -1] = 0.0  # right boundary
    return u, v

def plot_veloctiy_and_pressure(X, Y, p_next, u_next, v_next):
    plt.figure(figsize=(8, 6))
    plt.contourf(X[::2, ::2], Y[::2, ::2], p_next[::2, ::2])
    plt.quiver(X[::2, ::2], Y[::2, ::2], u_next[::2, ::2], v_next[::2, ::2])
    plt.colorbar()

    # Set title and axis labels
    plt.title('Velocity and Pressure Contours')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.xlim((0, 1))
    plt.ylim((0, 1))

    # Show the plot
    plt.show()

# 0. Initialisierung
x = np.linspace(0.0, DOMAIN_SIZE, N_GRIDPOINTS)
y = np.linspace(0.0, DOMAIN_SIZE, N_GRIDPOINTS)

X, Y = np.meshgrid(x, y)

u = np.zeros_like(X)   # velocity in x direction
v = np.zeros_like(X)   # velocity in y direction
p = np.zeros_like(X)   # pressure

for i in tqdm(range(N_ITERATIONS)):
    # 0. Initialisierung
    du_dx = central_difference_x(u)
    du_dy = central_difference_y(u)
    dv_dx = central_difference_x(v)
    dv_dy = central_difference_y(v)
    laplace_u = laplace(u)
    laplace_v = laplace(v)
    
    # 1. Lösung Impulsgleichung ohne Druck
    u_tentative = (u + TIME_STEP_LENGTH*(KINEMATIC_VISCOSITY*laplace_u - (u*du_dx + v*du_dy)))
    v_tentative = (v + TIME_STEP_LENGTH*(KINEMATIC_VISCOSITY*laplace_v - (u*dv_dx + v*dv_dy)))
    # 1. Randbedingungen erzwingen
    u_tentative, v_tentative = enforce_boundary_conditions(u_tentative, v_tentative)
    
    du_tentative_dx = central_difference_x(u_tentative)
    dv_tentative_dy = central_difference_y(v_tentative)
    
    # 2. Pressure-Poisson Equation
    # 2. rechte Seite berechnen
    rhs = (DENSITY / TIME_STEP_LENGTH * (du_tentative_dx+dv_tentative_dy))
    
    # 2. Druck Laplace lösen
    for _ in range(N_PRESSURE_ITERATIONS):
        p_next = np.zeros_like(p)
        p_next[1:-1, 1:-1] = 1/4 * (+p[1:-1, 0:-2]+p[0:-2, 1:-1]+p[1:-1, 2:]+p[2:, 1:-1]-ELEMENT_LENGTH**2*rhs[1:-1, 1:-1])
        # 2. Randbedingungen erzwingen
        p_next[:, -1] = p_next[:, -2]   # right boundary (Neumann)
        p_next[0,  :] = p_next[1,  :]   # bottom boundary (Neumann)
        p_next[:,  0] = p_next[:,  1]   # left boundary (Neumann)
        p_next[-1, :] = 0.0             # top boundary (Dirichlet)
        p = p_next
    
    dp_next_dx = central_difference_x(p_next)
    dp_next_dy = central_difference_y(p_next)
    
    # 3. Geschwindigkeit mit Druck korrigieren
    u_next = (u_tentative - TIME_STEP_LENGTH/DENSITY * dp_next_dx)
    v_next = (v_tentative - TIME_STEP_LENGTH/DENSITY * dp_next_dy)
    
    # 3. Randbedingungen erzwingen
    u_next, v_next = enforce_boundary_conditions(u_next, v_next)
    u = u_next
    v = v_next
    p = p_next
    plt.contourf(X[::, ::], Y[::, ::], p_next[::, ::])
    plt.quiver(X[::, ::], Y[::, ::], u_next[::, ::], v_next[::, ::])
    plt.colorbar()
    plt.title(f'Velocity and Pressure Contours')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.xlim((0, 1))
    plt.ylim((0, 1))
    plt.text(0.5, 0.01, f'Iteration: {i}/{N_ITERATIONS}', horizontalalignment='center', fontsize=12)
    fig = plt.gcf()
    fig.canvas.draw()
    image = Image.frombytes('RGB', fig.canvas.get_width_height(), fig.canvas.tostring_rgb())
    plt.clf()
    frames.append(image)
    
frames[0].save('animation.gif', format='GIF', append_images=frames[1:], save_all=True, duration=200, loop=0)
