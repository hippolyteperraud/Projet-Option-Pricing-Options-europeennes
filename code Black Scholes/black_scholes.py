import numpy as np
from scipy.stats import norm

# ============================================================================
# FORMULE BLACK-SCHOLES
# ============================================================================

def black_scholes_call(S, K, T, r, sigma):
    """
    Calcule le prix d'un call européen avec la formule de Black-Scholes
    
    C_0 = S_0 * N(d_1) - K * e^(-rT) * N(d_2)
    
    Paramètres:
    S     : prix actuel de l'action
    K     : prix d'exercice (strike)
    T     : temps jusqu'à maturité (en années)
    r     : taux sans risque
    sigma : volatilité annualisée
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    
    return call_price, d1, d2

def black_scholes_put(S, K, T, r, sigma):
    """
    Calcule le prix d'un put européen avec la formule de Black-Scholes
    
    P_0 = K * e^(-rT) * N(-d_2) - S_0 * N(-d_1)
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    put_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    
    return put_price, d1, d2

# ============================================================================
# UTILISATION
# ============================================================================

print("=" * 70)
print("FORMULE BLACK-SCHOLES")
print("=" * 70)

# Paramètres de l'exemple du rapport
S = 100          # Prix action
K = 105          # Strike
T = 1            # Maturité (1 an)
r = 0.05         # Taux sans risque 5%
sigma = 0.20     # Volatilité 20%

print(f"\nParamètres:")
print(f"  S₀ (Prix action)     = {S} €")
print(f"  K (Strike)           = {K} €")
print(f"  T (Maturité)         = {T} an")
print(f"  r (Taux sans risque) = {r*100}%")
print(f"  σ (Volatilité)       = {sigma*100}%")

# Calcul
call_price, d1, d2 = black_scholes_call(S, K, T, r, sigma)
put_price, _, _ = black_scholes_put(S, K, T, r, sigma)

print(f"\nRésultats:")
print(f"  d₁ = {d1:.4f}")
print(f"  d₂ = {d2:.4f}")
print(f"  N(d₁) = {norm.cdf(d1):.4f}")
print(f"  N(d₂) = {norm.cdf(d2):.4f}")
print(f"\n  Prix du CALL = {call_price:.4f} €")
print(f"  Prix du PUT  = {put_price:.4f} €")

# Vérifier la parité call-put
parity_left = call_price - put_price
parity_right = S - K * np.exp(-r * T)
print(f"\nVérification parité call-put:")
print(f"  C - P = {parity_left:.4f} €")
print(f"  S - Ke^(-rT) = {parity_right:.4f} €")
print(f"  Différence = {abs(parity_left - parity_right):.6f} €")

print(f"\n{'=' * 70}\n")