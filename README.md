# Mathématiques financières : Évaluation des options européennes

Rapport de Projet Semestriel — Semestre 1, GM4
**INSA Rouen Normandie**

## 📋 Description

Ce projet étudie la valorisation (*pricing*) des options européennes à travers deux approches complémentaires :

- **Modèle discret** de Cox, Ross et Rubinstein (arbre binomial)
- **Modèle continu** de Black-Scholes

L'objectif est de comprendre comment déterminer le juste prix d'une option (call ou put) en s'appuyant sur le principe fondamental d'**absence d'opportunité d'arbitrage**, et comment construire une stratégie de couverture (*hedging*) associée.

## 🎯 Contenu du rapport

1. **Cadre général** — définitions du prix forward, des options, enjeux de pricing et hedging
2. **Fondements mathématiques** — notations et cadre de travail
3. **Parité call-put** — relation d'arbitrage entre call et put
4. **Modèle discret (Cox-Ross-Rubinstein)** — stratégies, martingales, prix des options
5. **Modèle continu (Black-Scholes)**
   - Passage du discret au continu (convergence, théorème limite central)
   - Mouvement brownien et calcul stochastique (intégrale d'Itô, formule d'Itô, EDS)
   - Deux approches de valorisation : EDP (Feynman-Kac) et probabiliste (Girsanov)
   - Options sur plusieurs sous-jacents
6. **Calcul effectif** — formules explicites, simulation Monte Carlo, résolution numérique de l'EDP
7. **Volatilité historique et implicite** — smile de volatilité
8. **Annexes** — construction de l'intégrale de Wiener

## 🧮 Exemple numérique traité

Pricing d'un call européen avec :
- Prix de l'action : S₀ = 100 €
- Strike : K = 105 €
- Maturité : T = 1 an
- Taux sans risque : r = 5%
- Volatilité : σ = 20%

**Résultats obtenus :**
| Méthode | Prix du call |
| Formule de Black-Scholes (fermée) | 8.02 € |
| Simulation Monte Carlo (100 000 tirages) | 8.05 € |

La relation de parité call-put est vérifiée à la précision numérique près.

## 👥 Auteurs

- **Hippolyte Perraud** — contributeur principal du projet (recherche, rédaction, implémentation)
- Yassine Mhiri — contribution mineure
- Ahmed Benlafya — contribution mineure

**Encadrement :** Ioana Ciotir

## 📚 Références principales

- Hull, J. C. (2021). *Options, Futures, and Other Derivatives* (11th ed.)
- Black, F., & Scholes, M. (1973). *The pricing of options and corporate liabilities*
- Shreve, S. E. (2004). *Stochastic Calculus for Finance II*
- Pardoux, E. (2012). *Processus stochastiques et applications*

## 📅 Période

Septembre 2025 - Janvier 2026