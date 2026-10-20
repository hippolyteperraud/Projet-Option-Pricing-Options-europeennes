"""
GREEKS PAR MONTE CARLO - Annexe au pricing Monte Carlo d'un call européen
=========================================================================

Trois familles d'estimateurs pour une dérivée dC/dθ = d/dθ E[e^{-rT} f(S_T)] :

1. Différences finies (FD)  : (C(θ+h) - C(θ-h)) / 2h
     - avec tirages indépendants  -> variance en 1/h^2 (catastrophique)
     - avec nombres aléatoires communs (CRN) -> variance contrôlée
2. Pathwise (IPA)           : on dérive le payoff le long de la trajectoire
     - non biaisé, très faible variance, mais exige un payoff lipschitzien
       (échoue pour le gamma : la dérivée seconde du payoff est un Dirac)
3. Likelihood Ratio (LR)    : on dérive la densité de S_T, pas le payoff
     - marche pour tout payoff, y compris discontinu, mais variance plus élevée
4. Mixte pathwise-LR        : pour le gamma, on dérive le delta pathwise par LR

Modèle : S_T = S0 exp((r - sigma^2/2) T + sigma sqrt(T) Z),  Z ~ N(0,1)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

rng = np.random.default_rng(42)

# ----------------------------------------------------------------------------
# Black-Scholes : formules fermées de référence
# ----------------------------------------------------------------------------

def bs_greeks(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    delta = norm.cdf(d1)
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    vega = S * norm.pdf(d1) * np.sqrt(T)
    theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T))
             - r * K * np.exp(-r * T) * norm.cdf(d2))
    rho = K * T * np.exp(-r * T) * norm.cdf(d2)
    return dict(price=price, delta=delta, gamma=gamma, vega=vega,
                theta=theta, rho=rho)

# ----------------------------------------------------------------------------
# Brique de base : S_T à partir de Z (on garde Z pour pouvoir le réutiliser)
# ----------------------------------------------------------------------------

def terminal(S, T, r, sigma, Z):
    return S * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)

def call_payoff(ST, K):
    return np.maximum(ST - K, 0.0)

def mean_se(x):
    return x.mean(), x.std(ddof=1) / np.sqrt(len(x))

# ----------------------------------------------------------------------------
# 1. Différences finies
# ----------------------------------------------------------------------------

def fd_delta(S, K, T, r, sigma, N, h, crn=True):
    Z1 = rng.standard_normal(N)
    Z2 = Z1 if crn else rng.standard_normal(N)
    up = call_payoff(terminal(S + h, T, r, sigma, Z1), K)
    dn = call_payoff(terminal(S - h, T, r, sigma, Z2), K)
    return mean_se(np.exp(-r * T) * (up - dn) / (2 * h))

def fd_gamma(S, K, T, r, sigma, N, h, crn=True):
    Z1 = rng.standard_normal(N)
    Z2 = Z1 if crn else rng.standard_normal(N)
    Z3 = Z1 if crn else rng.standard_normal(N)
    up = call_payoff(terminal(S + h, T, r, sigma, Z1), K)
    md = call_payoff(terminal(S, T, r, sigma, Z2), K)
    dn = call_payoff(terminal(S - h, T, r, sigma, Z3), K)
    return mean_se(np.exp(-r * T) * (up - 2 * md + dn) / h**2)

# ----------------------------------------------------------------------------
# 2. Pathwise : d payoff / d theta = 1{S_T>K} * dS_T/d theta
# ----------------------------------------------------------------------------

def pathwise_all(S, K, T, r, sigma, N):
    Z = rng.standard_normal(N)
    ST = terminal(S, T, r, sigma, Z)
    itm = (ST > K).astype(float)
    disc = np.exp(-r * T)

    price = mean_se(disc * call_payoff(ST, K))
    delta = mean_se(disc * itm * ST / S)
    vega = mean_se(disc * itm * ST * (np.sqrt(T) * Z - sigma * T))
    rho = mean_se(disc * itm * K * T)
    # dC/dT = -r C + e^{-rT} E[1{ST>K} ST ((r - s^2/2) + s Z / (2 sqrt(T)))]
    dCdT = disc * (-r * call_payoff(ST, K)
                   + itm * ST * ((r - 0.5 * sigma**2) + sigma * Z / (2 * np.sqrt(T))))
    theta = mean_se(-dCdT)
    return dict(price=price, delta=delta, vega=vega, rho=rho, theta=theta)

# ----------------------------------------------------------------------------
# 3. Likelihood ratio : d/dtheta de la densité lognormale de S_T
# ----------------------------------------------------------------------------

def lr_all(S, K, T, r, sigma, N):
    Z = rng.standard_normal(N)
    ST = terminal(S, T, r, sigma, Z)
    pay = np.exp(-r * T) * call_payoff(ST, K)
    sq = np.sqrt(T)

    delta = mean_se(pay * Z / (S * sigma * sq))
    gamma = mean_se(pay * (Z**2 - 1 - sigma * sq * Z) / (S**2 * sigma**2 * T))
    vega = mean_se(pay * ((Z**2 - 1) / sigma - sq * Z))
    return dict(delta=delta, gamma=gamma, vega=vega)

# ----------------------------------------------------------------------------
# 4. Gamma mixte : pathwise pour le delta, puis LR pour la dérivée du delta
# ----------------------------------------------------------------------------

def mixed_gamma(S, K, T, r, sigma, N):
    Z = rng.standard_normal(N)
    ST = terminal(S, T, r, sigma, Z)
    itm = (ST > K).astype(float)
    est = np.exp(-r * T) * itm * ST * (Z / (sigma * np.sqrt(T)) - 1.0) / S**2
    return mean_se(est)

# ----------------------------------------------------------------------------
# Expérience 1 : tableau des Greeks, MC vs Black-Scholes
# ----------------------------------------------------------------------------

S, K, T, r, sigma = 100.0, 105.0, 1.0, 0.05, 0.20
N = 1_000_000
ref = bs_greeks(S, K, T, r, sigma)
pw = pathwise_all(S, K, T, r, sigma, N)
lr = lr_all(S, K, T, r, sigma, N)
mg = mixed_gamma(S, K, T, r, sigma, N)

rows = [
    ("Prix",  ref["price"], pw["price"], None, None),
    ("Delta", ref["delta"], pw["delta"], lr["delta"], None),
    ("Gamma", ref["gamma"], None, lr["gamma"], mg),
    ("Vega",  ref["vega"],  pw["vega"],  lr["vega"], None),
    ("Theta", ref["theta"], pw["theta"], None, None),
    ("Rho",   ref["rho"],   pw["rho"],   None, None),
]

def fmt(x):
    return "      -        " if x is None else f"{x[0]:8.4f} ±{1.96*x[1]:.4f}"

print("=" * 96)
print(f"GREEKS PAR MONTE CARLO (N = {N:,}) — S={S}, K={K}, T={T}, r={r}, sigma={sigma}")
print("=" * 96)
print(f"{'':7}{'Black-Scholes':>15}{'Pathwise':>20}{'Likelihood ratio':>22}{'Mixte':>20}")
for name, b, p, l, m in rows:
    print(f"{name:7}{b:15.4f}  {fmt(p):>18}  {fmt(l):>20}  {fmt(m):>18}")

# ----------------------------------------------------------------------------
# Expérience 2 : variance des estimateurs (répétitions indépendantes)
# ----------------------------------------------------------------------------

n_rep, N_rep = 300, 10_000

def replicate(fn):
    return np.array([fn()[0] for _ in range(n_rep)])

h_list = np.array([5.0, 2.0, 1.0, 0.5, 0.2, 0.1, 0.05, 0.02])
std_fd_indep = [replicate(lambda h=h: fd_delta(S, K, T, r, sigma, N_rep, h, crn=False)).std() for h in h_list]
std_fd_crn = [replicate(lambda h=h: fd_delta(S, K, T, r, sigma, N_rep, h, crn=True)).std() for h in h_list]
# Biais exact de la différence finie centrée (calculé sur Black-Scholes, sans bruit MC)
bias_fd = [abs((bs_greeks(S + h, K, T, r, sigma)["price"] - bs_greeks(S - h, K, T, r, sigma)["price"])
               / (2 * h) - ref["delta"]) for h in h_list]
std_pw_delta = replicate(lambda: pathwise_all(S, K, T, r, sigma, N_rep)["delta"]).std()
std_lr_delta = replicate(lambda: lr_all(S, K, T, r, sigma, N_rep)["delta"]).std()

# Gamma par FD avec CRN : écart-type et biais exact en fonction de h
hg_list = np.array([5.0, 2.0, 1.0, 0.5, 0.2, 0.1])
std_fd_gamma_crn = [replicate(lambda h=h: fd_gamma(S, K, T, r, sigma, N_rep, h, crn=True)).std() for h in hg_list]
bias_fd_gamma = [abs((bs_greeks(S + h, K, T, r, sigma)["price"] - 2 * ref["price"]
                      + bs_greeks(S - h, K, T, r, sigma)["price"]) / h**2 - ref["gamma"]) for h in hg_list]

h_g = 1.0
g_std = {
    "FD indép.":  replicate(lambda: fd_gamma(S, K, T, r, sigma, N_rep, h_g, crn=False)).std(),
    "FD CRN":     replicate(lambda: fd_gamma(S, K, T, r, sigma, N_rep, h_g, crn=True)).std(),
    "LR pur":     replicate(lambda: lr_all(S, K, T, r, sigma, N_rep)["gamma"]).std(),
    "Mixte":      replicate(lambda: mixed_gamma(S, K, T, r, sigma, N_rep)).std(),
}

print("\nÉcart-type du delta estimé (N = 10 000, 300 répétitions)")
print(f"  Pathwise : {std_pw_delta:.5f}   |   LR : {std_lr_delta:.5f}")
print(f"  {'h':>6}{'FD indép.':>12}{'FD CRN':>12}{'biais exact':>13}")
for h, a, b, c in zip(h_list, std_fd_indep, std_fd_crn, bias_fd):
    print(f"  {h:6.2f}{a:12.5f}{b:12.5f}{c:13.6f}")
print("\nGamma par FD + CRN en fonction de h (N = 10 000)")
print(f"  {'h':>6}{'écart-type':>12}{'biais exact':>13}")
for h, a, c in zip(hg_list, std_fd_gamma_crn, bias_fd_gamma):
    print(f"  {h:6.2f}{a:12.6f}{c:13.7f}")
print("\nÉcart-type du gamma estimé (N = 10 000, h = 1)")
for k_, v in g_std.items():
    print(f"  {k_:10}: {v:.6f}   (x{v / g_std['Mixte']:.1f} vs mixte)")

# ----------------------------------------------------------------------------
# Expérience 3 : Delta et Gamma en fonction de S (balayage)
# ----------------------------------------------------------------------------

S_grid = np.linspace(70, 140, 36)
d_mc, g_mc, d_bs, g_bs = [], [], [], []
for s in S_grid:
    d_mc.append(pathwise_all(s, K, T, r, sigma, 200_000)["delta"][0])
    g_mc.append(mixed_gamma(s, K, T, r, sigma, 200_000)[0])
    b = bs_greeks(s, K, T, r, sigma)
    d_bs.append(b["delta"]); g_bs.append(b["gamma"])

# ----------------------------------------------------------------------------
# Figure
# ----------------------------------------------------------------------------

plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(2, 2, figsize=(12, 8.5))

ax[0, 0].plot(S_grid, d_bs, "k-", lw=1.5, label="Black-Scholes")
ax[0, 0].plot(S_grid, d_mc, "o", ms=4, color="tab:blue", label="Monte Carlo (pathwise)")
ax[0, 0].axvline(K, color="grey", ls=":", lw=1)
ax[0, 0].set(xlabel="$S_0$", ylabel="Delta", title="Delta : MC pathwise vs formule fermée")
ax[0, 0].legend(frameon=False)

ax[0, 1].plot(S_grid, g_bs, "k-", lw=1.5, label="Black-Scholes")
ax[0, 1].plot(S_grid, g_mc, "o", ms=4, color="tab:green", label="Monte Carlo (mixte)")
ax[0, 1].axvline(K, color="grey", ls=":", lw=1)
ax[0, 1].set(xlabel="$S_0$", ylabel="Gamma", title="Gamma : MC mixte pathwise-LR vs formule fermée")
ax[0, 1].legend(frameon=False)

ax[1, 0].loglog(h_list, std_fd_indep, "o-", color="tab:red", label="FD, tirages indépendants")
ax[1, 0].loglog(h_list, std_fd_crn, "s-", color="tab:orange", label="FD, CRN")
ax[1, 0].axhline(std_pw_delta, color="tab:blue", ls="--", label="Pathwise")
ax[1, 0].set(xlabel="pas de différence finie $h$", ylabel="écart-type de l'estimateur",
             title="Delta : le piège du pas $h$ (N = 10 000)")
ax[1, 0].legend(frameon=False)

names = list(g_std.keys()); vals = [g_std[k_] for k_ in names]
colors = ["tab:red", "tab:orange", "tab:purple", "tab:green"]
ax[1, 1].bar(names, vals, color=colors)
ax[1, 1].set_yscale("log")
ax[1, 1].set(ylabel="écart-type de l'estimateur",
             title="Gamma : variance des 4 estimateurs (N = 10 000)")
for i, v in enumerate(vals):
    ax[1, 1].text(i, v * 1.15, f"{v:.4f}", ha="center", fontsize=9)

plt.tight_layout()
plt.savefig("greeks_monte_carlo.png", dpi=160)
print("\nFigure enregistrée : greeks_monte_carlo.png")
