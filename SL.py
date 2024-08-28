import numpy as np
import matplotlib.pyplot as plt
import quantecon as qe
from numba import njit, prange
import matplotlib.cm as cm
from matplotlib.lines import Line2D

# Parameters
nb = 30
Bs_grid_min = -0.3
Bs_grid_max = 0.3
Bl_grid_min = -2
Bl_grid_max = 2
ny = 20
beta = 0.98
sigma = 2
r = 0.01
rho = 0.945
sigma_E = 0.015
mu = -0.5 * sigma_E**2
psi = 0.083
d0 = 0.17
d1 = 1.2
delta = 0.035

# Grids
B_grid_s = np.linspace(Bs_grid_min, Bs_grid_max, nb)
B_grid_l = np.linspace(Bl_grid_min, Bl_grid_max, nb)
mc = qe.markov.tauchen(ny, rho, sigma_E, mu, 3)
y_grid, P = np.exp(mc.state_values), mc.P
B0_idx_s = np.searchsorted(B_grid_s, 1e-10)
B0_idx_l = np.searchsorted(B_grid_l, 1e-10)
ql_future = np.ones((nb,nb,ny))/(1+r)
B_star_S = np.zeros((nb,nb,ny),dtype=int)
B_star_L = np.zeros((nb,nb,ny),dtype=int)

@njit
def u(c, sigma):
    return (c**(1 - sigma)) / (1 - sigma)
@njit
def h(y, d0, d1):
    return d0 * y + d1 * y**2

@njit
def compute_q(v_c, v_d, qs, ql, beta, sigma, r, rho, sigma_E, psi, P, y_grid, B_grid_s, B_grid_l, B0_idx_s, B0_idx_l, delta):
    for B_idx_S in range(len(B_grid_s)):
        for B_idx_L in range(len(B_grid_l)):
            for y_idx in range(len(y_grid)):
                delta_default = (P[y_idx, :] * (v_c[B_idx_S, B_idx_L, :] < v_d)).sum()
                qs[B_idx_S, B_idx_L, y_idx] = (1 - delta_default) / (1 + r)    
                ql[B_idx_S, B_idx_L, y_idx] = 1/(1+r) * (1 - delta_default) * (delta + (1 - delta) * np.sum(P[y_idx, :] * ql_future[B_idx_S, B_idx_L, :]))

@njit
def T_d(y_idx, v_c, v_d, beta, sigma, r, rho, sigma_E, psi, P, y_grid, B_grid_s, B_grid_l,  B0_idx_s, B0_idx_l):
    if y_grid[y_idx] - h(y_grid[y_idx], d0, d1) > 0:
        current_utility = u(y_grid[y_idx]- h(y_grid[y_idx], d0, d1), sigma) 
    else: current_utility = u(1,sigma)
    v = np.maximum(v_c[B0_idx_s, B0_idx_l, :], v_d)
    cont_value = np.sum(((1-psi) * v + psi * v_d) * P[y_idx, :])
    return current_utility + beta * cont_value

@njit(parallel=True)
def T_c(B_idx_S, B_idx_L, y_idx, v_c, v_d, qs, ql, beta, sigma, r, rho, sigma_E, psi, P, y_grid, B_grid_s, B_grid_l, B0_idx_s, B0_idx_l, delta):
    B_S = B_grid_s[B_idx_S]
    B_L = B_grid_l[B_idx_L]
    y = y_grid[y_idx]
    current_max = -1e6
    for Bp_idx_S, Bp_S in enumerate(B_grid_s):
        for Bp_idx_L, Bp_L in enumerate(B_grid_l):
            if Bp_idx_S > 0 and Bp_idx_S < len(B_grid_s) - 1:
                dV_dbS_prime = np.sum(P[y_idx, :] * (np.maximum(v_c[Bp_idx_S + 1, Bp_idx_L, :], v_d) - np.maximum(v_c[Bp_idx_S - 1, Bp_idx_L, :], v_d)) / (B_grid_s[Bp_idx_S + 1] - B_grid_s[Bp_idx_S - 1]))
            elif Bp_idx_S == 0:
                dV_dbS_prime = np.sum(P[y_idx, :] * (np.maximum(v_c[Bp_idx_S + 1, Bp_idx_L, :], v_d) - np.maximum(v_c[Bp_idx_S, Bp_idx_L, :], v_d)) / (B_grid_s[Bp_idx_S + 1] - B_grid_s[Bp_idx_S]))
            else:  # Bp_idx_S == len(B_grid) - 1
                dV_dbS_prime = np.sum(P[y_idx, :] * (np.maximum(v_c[Bp_idx_S, Bp_idx_L, :], v_d) - np.maximum(v_c[Bp_idx_S - 1, Bp_idx_L, :], v_d)) / (B_grid_s[Bp_idx_S] - B_grid_s[Bp_idx_S - 1]))

            if Bp_idx_L > 0 and Bp_idx_L < len(B_grid_l) - 1:
                dV_dbL_prime = np.sum(P[y_idx, :] * (np.maximum(v_c[Bp_idx_S, Bp_idx_L + 1, :], v_d) - np.maximum(v_c[Bp_idx_S, Bp_idx_L - 1, :], v_d)) / (B_grid_l[Bp_idx_L + 1] - B_grid_l[Bp_idx_L - 1]))
            elif Bp_idx_L == 0:
                dV_dbL_prime = np.sum(P[y_idx, :] * (np.maximum(v_c[Bp_idx_S, Bp_idx_L + 1, :], v_d) - np.maximum(v_c[Bp_idx_S, Bp_idx_L, :], v_d)) / (B_grid_l[Bp_idx_L + 1] - B_grid_l[Bp_idx_L]))
            else:  # Bp_idx_L == len(B_grid) - 1
                dV_dbL_prime = np.sum(P[y_idx, :] * (np.maximum(v_c[Bp_idx_S, Bp_idx_L, :], v_d) - np.maximum(v_c[Bp_idx_S, Bp_idx_L - 1, :], v_d)) / (B_grid_l[Bp_idx_L] - B_grid_l[Bp_idx_L - 1]))
            FOC_bS =  (((y + B_S + delta * B_L
                      - qs[Bp_idx_S, Bp_idx_L, y_idx] * Bp_S
                + ql[Bp_idx_S, Bp_idx_L, y_idx] * (1 - delta) * B_L
            - ql[Bp_idx_S, Bp_idx_L, y_idx] * Bp_L)**(-sigma)) * (-1) * qs[Bp_idx_S, Bp_idx_L, y_idx] + beta * dV_dbS_prime)
            FOC_bL = (((y + B_S + delta * B_L
                      - qs[Bp_idx_S, Bp_idx_L, y_idx] * Bp_S
                + ql[Bp_idx_S, Bp_idx_L, y_idx] * (1 - delta) * B_L
            - ql[Bp_idx_S, Bp_idx_L, y_idx] * Bp_L)**(-sigma)) * (-1) * ql[Bp_idx_S, Bp_idx_L, y_idx] + beta * dV_dbL_prime)
            v = np.maximum(v_c[Bp_idx_S, Bp_idx_L, :], v_d)
            val = (((y + B_S + delta * B_L
                              - qs[Bp_idx_S, Bp_idx_L, y_idx] * Bp_S
                              + ql[Bp_idx_S, Bp_idx_L, y_idx] * (1 - delta) * B_L
                              - ql[Bp_idx_S, Bp_idx_L, y_idx] * Bp_L) ** (1 - sigma)) / (1 - sigma) + beta * np.sum(v * P[y_idx, :]))
            
            if  np.abs(FOC_bS) < 1e-8 and np.abs(FOC_bL) < 1e-8 and val > current_max:
                current_max = val
                Bp_star_idx_S = Bp_idx_S
                Bp_star_idx_L = Bp_idx_L

    return  current_max, Bp_star_idx_S, Bp_star_idx_L

@njit(parallel=True)
def update_values_and_prices(v_c, v_d, B_star_S, B_star_L, qs, ql, beta, sigma, r, rho, sigma_E, psi, P, y_grid, B_grid_s, B_grid_l, B0_idx_s, B0_idx_l, delta):
    compute_q(v_c, v_d, qs, ql, beta, sigma, r, rho, sigma_E, psi, P, y_grid, B_grid_s, B_grid_l, B0_idx_s, B0_idx_l, delta)
    new_v_c = np.empty_like(v_c)
    new_v_d = np.empty_like(v_d)
    for y_idx in prange(len(y_grid)):
        new_v_d[y_idx] = T_d(y_idx, v_c, v_d, beta, sigma, r, rho, sigma_E, psi, P, y_grid, B_grid_s, B_grid_l, B0_idx_s, B0_idx_l)
        for B_idx_S in range(len(B_grid_s)):
            for B_idx_L in range(len(B_grid_l)):
                new_v_c[B_idx_S, B_idx_L, y_idx], Bp_idx_S, Bp_idx_L = T_c(B_idx_S, B_idx_L, y_idx, v_c, v_d, qs, ql, beta, sigma, r, rho, sigma_E, psi, P, y_grid, B_grid_s, B_grid_l, B0_idx_s, B0_idx_l, delta)
                B_star_S[B_idx_S, B_idx_L, y_idx] = Bp_idx_S
                B_star_L[B_idx_S, B_idx_L, y_idx] = Bp_idx_L
                
    return new_v_c, new_v_d

def solve(B_grid_s, B_grid_l, y_grid, P, B0_idx_s, B0_idx_l, beta, sigma, r, rho, sigma_E, psi, delta, tol=1e-6, max_iter=5000):
    v_c = np.zeros((nb, nb,ny))
    v_d = np.zeros(ny)
    qs = np.zeros_like(v_c)
    ql = np.zeros_like(v_c)
    ql_future = np.ones((nb, nb,ny))/(1+r)
    B_star_S = np.empty_like(v_c, dtype=int)
    B_star_L = np.empty_like(v_c, dtype=int)
    current_iter = 0
    dist = np.inf
    while (current_iter < max_iter) and (dist > tol):
        new_v_c, new_v_d = update_values_and_prices(v_c, v_d, B_star_S, B_star_L, qs, ql, beta, sigma, r, rho, sigma_E, psi, P, y_grid, B_grid_s, B_grid_l, B0_idx_s, B0_idx_l, delta)
        dist = np.max(np.abs(new_v_c - v_c)) + np.max(np.abs(new_v_d - v_d)) 
        print(f'It # {current_iter} finished. Distance: {dist}')
        v_c = new_v_c
        v_d = new_v_d
        ql_future = np.copy(ql)
        current_iter += 1
    return v_c, v_d, qs, ql, B_star_S, B_star_L, ql_future

v_c, v_d, qs, ql, B_star_S, B_star_L, ql_future = solve(B_grid_s, B_grid_l, y_grid, P, B0_idx_s, B0_idx_l, beta, sigma, r, rho, sigma_E, psi, delta)



# Graphs
# Create "Y High" and "Y Low" values as 10% devs from mean
high, low = np.mean(y_grid) * 1.05, np.mean(y_grid) * 0.95
iy_high, iy_low = (np.searchsorted(y_grid, x) for x in (high, low))

# Short bonds
fig, ax = plt.subplots(figsize=(10, 6.5))
ax.set_title("$B_S$ pricing function $q(y, B_s', 0)$")
x = []
q_low = []
q_high = []
for i, B in enumerate(B_grid_s):
    x.append(B)
    q_low.append(qs[i,B0_idx_l, iy_low])
    q_high.append(qs[i,B0_idx_l, iy_high])
ax.plot(x, q_high, label="$y_H$", lw=2, alpha=0.7)
ax.plot(x, q_low, label="$y_L$", lw=2, alpha=0.7)
ax.set_xlabel("$B_s'$")
ax.legend(loc='upper left', frameon=False)
plt.show()

# Long bonds
fig, ax = plt.subplots(figsize=(10, 7))
ax.set_title(r"$B_L$ pricing function $q(y, 0, B_l')$")
x = []
q_low = []
q_high = []
for i, B in enumerate(B_grid_l):
    x.append(B)
    q_low.append(ql[B0_idx_s, i, iy_low])  # Promedio sobre B_S para y bajo
    q_high.append(ql[B0_idx_s, i, iy_high])  # Promedio sobre B_S para y alto
ax.plot(x, q_high, label="$y_H$", lw=2)
ax.plot(x, q_low, label="$y_L$", lw=2)
ax.set_xlabel("$B_l'$")
ax.legend(loc='upper left')
plt.show()


# Value function 2D and 3D
v = np.maximum(v_c, np.reshape(v_d, (1,1, ny)))

## 2D
fig, ax = plt.subplots(figsize=(10, 7))
ax.set_title("Value Functions")
ax.plot(B_grid_s, v[:, B0_idx_l, iy_high], label="$y_H$", lw=2)
ax.plot(B_grid_s, v[:,B0_idx_l ,iy_low], label="$y_L$", lw=2)
ax.legend(loc='upper left')
ax.set(xlabel="$B_S$", ylabel="$v(y, B_s, 0)$")
ax.set_xlim(min(B_grid_s), max(B_grid_s))
plt.show()
fig, ax = plt.subplots(figsize=(10, 7))
ax.set_title("Value Functions")
ax.plot(B_grid_l, v[B0_idx_s, : , iy_high], label="$y_H$", lw=2)
ax.plot(B_grid_l, v[B0_idx_s,: ,iy_low], label="$y_L$", lw=2)
ax.legend(loc='upper left')
ax.set(xlabel="$B_L$", ylabel="$v(y, 0, B_l)$")
ax.set_xlim(min(B_grid_l), max(B_grid_l))
plt.show()

## 3D
fig = plt.figure(figsize=(10, 6.5))
ax = fig.add_subplot(111, projection='3d')
B_S, B_L = np.meshgrid(B_grid_s, B_grid_l)
Y = np.tile(y_grid, (len(B_grid_s), 1)).T
v_high = v[:, :, iy_high]
v_low = v[:, :, iy_low]
ax.plot_surface(B_S, B_L, v_high, cmap='viridis', alpha=0.7, label="$y_H$")
ax.plot_surface(B_S, B_L, v_low, cmap='plasma', alpha=0.7, label="$y_L$")
viridis_color = cm.viridis(0.5)
plasma_color = cm.plasma(0.5)
legend_elements = [
    Line2D([0], [0], color=viridis_color, marker='o', markersize=10, label='$y_H$'),
    Line2D([0], [0], color=plasma_color, marker='o', markersize=10, label='$y_L$')
]
ax.set_title("Value Functions")
ax.set_xlabel("$B_S$")
ax.set_ylabel("$B_L$")
ax.set_zlabel("$v(y, B_S, B_L)$")
plt.show()
