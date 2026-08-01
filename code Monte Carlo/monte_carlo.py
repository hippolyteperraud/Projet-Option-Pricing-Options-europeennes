import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# ============================================================================
# MÉTHODE MONTE CARLO
# ============================================================================

def monte_carlo_call(S, K, T, r, sigma, num_simulations=100000):
    """
    Estime le prix d'un call européen par simulation Monte Carlo
    
    Idée : Simuler N trajectoires du prix S_t jusqu'à maturité T,
    puis calculer l'espérance du payoff actualisé.
    
    E[C_0] = e^(-rT) * E[max(S_T - K, 0)]
    """
    
    # Générer les prix finaux S_T selon le mouvement brownien géométrique
    # S_T = S_0 * exp((r - σ²/2)*T + σ*√T*Z)
    # où Z ~ N(0,1)
    
    Z = np.random.standard_normal(num_simulations)
    S_T = S * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
    
    # Payoff final du call
    payoff = np.maximum(S_T - K, 0)
    
    # Actualiser et prendre la moyenne
    call_price = np.exp(-r * T) * np.mean(payoff)
    
    # Écart-type pour l'intervalle de confiance
    std_dev = np.std(payoff)
    std_error = std_dev / np.sqrt(num_simulations)
    ic_95 = 1.96 * std_error * np.exp(-r * T)
    
    return call_price, ic_95, S_T, payoff

def monte_carlo_put(S, K, T, r, sigma, num_simulations=100000):
    """
    Estime le prix d'un put européen par simulation Monte Carlo
    """
    
    Z = np.random.standard_normal(num_simulations)
    S_T = S * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
    
    # Payoff final du put
    payoff = np.maximum(K - S_T, 0)
    
    # Actualiser et prendre la moyenne
    put_price = np.exp(-r * T) * np.mean(payoff)
    
    # Intervalle de confiance
    std_dev = np.std(payoff)
    std_error = std_dev / np.sqrt(num_simulations)
    ic_95 = 1.96 * std_error * np.exp(-r * T)
    
    return put_price, ic_95, S_T, payoff

# ============================================================================
# CONVERGENCE DE MONTE CARLO
# ============================================================================

def monte_carlo_convergence(S, K, T, r, sigma):
    """
    Montre comment l'estimateur converge avec le nombre de simulations
    """
    
    num_sims_list = [100, 500, 1000, 5000, 10000, 50000, 100000]
    call_prices = []
    
    for num_sims in num_sims_list:
        call_price, _, _, _ = monte_carlo_call(S, K, T, r, sigma, num_sims)
        call_prices.append(call_price)
    
    return num_sims_list, call_prices

# ============================================================================
# UTILISATION
# ============================================================================

print("=" * 80)
print("MÉTHODE MONTE CARLO - VALORISATION D'OPTIONS")
print("=" * 80)

# Paramètres
S = 100
K = 105
T = 1
r = 0.05
sigma = 0.20

print(f"\nParamètres:")
print(f"  S₀ (Prix action)     = {S} €")
print(f"  K (Strike)           = {K} €")
print(f"  T (Maturité)         = {T} an")
print(f"  r (Taux sans risque) = {r*100}%")
print(f"  σ (Volatilité)       = {sigma*100}%")

# ========================================================================
# 1. CONVERGENCE AVEC LE NOMBRE DE SIMULATIONS
# ========================================================================

print(f"\n{'─' * 80}")
print("1. CONVERGENCE DE L'ESTIMATEUR MONTE CARLO")
print(f"{'─' * 80}\n")

num_sims_list, call_prices = monte_carlo_convergence(S, K, T, r, sigma)

print(f"{'Simulations':<15} {'Prix Call (€)':<20}")
print("─" * 35)

for num_sims, call_price in zip(num_sims_list, call_prices):
    print(f"{num_sims:<15} {call_price:<20.6f}")

# ========================================================================
# 2. ESTIMATION AVEC 100 000 SIMULATIONS
# ========================================================================

print(f"\n{'─' * 80}")
print("2. ESTIMATION AVEC 100 000 SIMULATIONS")
print(f"{'─' * 80}\n")

num_simulations = 100000

call_price, ic_call, S_T, payoff_call = monte_carlo_call(
    S, K, T, r, sigma, num_simulations
)
put_price, ic_put, _, payoff_put = monte_carlo_put(
    S, K, T, r, sigma, num_simulations
)

print(f"Prix du CALL = {call_price:.6f} € ± {ic_call:.6f} € (IC 95%)")
print(f"Prix du PUT  = {put_price:.6f} € ± {ic_put:.6f} € (IC 95%)")

# Vérifier la parité
parity_check = call_price - put_price
parity_theory = S - K * np.exp(-r * T)
print(f"\nVérification parité call-put:")
print(f"  C - P = {parity_check:.6f} €")
print(f"  S - Ke^(-rT) = {parity_theory:.6f} €")
print(f"  Différence = {abs(parity_check - parity_theory):.6f} €")

# ========================================================================
# 3. COMPARAISON AVEC BLACK-SCHOLES
# ========================================================================

print(f"\n{'─' * 80}")
print("3. COMPARAISON AVEC BLACK-SCHOLES")
print(f"{'─' * 80}\n")

def black_scholes_call(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

bs_call = black_scholes_call(S, K, T, r, sigma)
error = abs(call_price - bs_call)
error_pct = (error / bs_call) * 100

print(f"Black-Scholes (fermée) = {bs_call:.6f} €")
print(f"Monte Carlo            = {call_price:.6f} €")
print(f"Différence             = {error:.6f} € ({error_pct:.2f}%)")


# ========================================================================
# 4. VISUALISATION
# ========================================================================

print(f"\n{'─' * 80}")
print("4. VISUALISATION")
print(f"{'─' * 80}\n")

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Graphique 1 : Distribution des prix finaux S_T
axes[0, 0].hist(S_T, bins=100, alpha=0.7, color='blue', edgecolor='black')
axes[0, 0].axvline(K, color='red', linestyle='--', linewidth=2, label=f'Strike K={K}')
axes[0, 0].set_xlabel('Prix final $S_T$ (€)')
axes[0, 0].set_ylabel('Fréquence')
axes[0, 0].set_title('Distribution des prix finaux (100 000 simulations)')
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

# Graphique 2 : Distribution des payoffs du call
axes[0, 1].hist(payoff_call, bins=100, alpha=0.7, color='green', edgecolor='black')
axes[0, 1].set_xlabel('Payoff du call (€)')
axes[0, 1].set_ylabel('Fréquence')
axes[0, 1].set_title(f'Distribution des payoffs (Prix moyen = {call_price:.4f} €)')
axes[0, 1].grid(alpha=0.3)

# Graphique 3 : Convergence
axes[1, 0].semilogx(num_sims_list, call_prices, 'o-', linewidth=2, markersize=8)
axes[1, 0].axhline(bs_call, color='red', linestyle='--', linewidth=2, label='Black-Scholes')
axes[1, 0].set_xlabel('Nombre de simulations')
axes[1, 0].set_ylabel('Prix du call (€)')
axes[1, 0].set_title('Convergence de Monte Carlo')
axes[1, 0].legend()
axes[1, 0].grid(alpha=0.3)

# Graphique 4 : Quelques trajectoires
num_trajectories = 100
np.random.seed(42)
t = np.linspace(0, T, 100)
for i in range(num_trajectories):
    Z = np.random.standard_normal(len(t))
    S_path = S * np.exp((r - 0.5 * sigma**2) * t + sigma * np.sqrt(t) * Z)
    axes[1, 1].plot(t, S_path, alpha=0.1, color='blue')

axes[1, 1].set_xlabel('Temps (années)')
axes[1, 1].set_ylabel('Prix de l\'action (€)')
axes[1, 1].set_title(f'100 trajectoires simulées du prix (σ = {sigma*100}%)')
axes[1, 1].grid(alpha=0.3)

plt.tight_layout()

plt.show()

print(f"\n{'=' * 80}\n")