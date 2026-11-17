"""Optimal execution with dynamic programming (discrete-time Almgren-Chriss).

Buy Q shares over T steps. At each step choose x_t; inventory q_{t+1} = q_t - x_t.
Objective (mean-variance):

    min  sum_t eta * x_t^2  +  lambda * sigma^2 * dt * sum_t q_{t+1}^2

The first term is the expected impact cost, the second is lambda * Var[cost].
"""
import os

import numpy as np
import matplotlib.pyplot as plt


Q = 100            # total shares to buy
T = 20             # number of time steps
dt = 1.0 / T

sigma = 1.0        # price volatility
eta = 0.05         # temporary impact strength
lambda_risk = 0.01 # risk aversion weight (kappa*T = 2: neither TWAP nor all-at-once)

S0 = 100.0         # initial price for simulations
N_PATHS = 5000     # Monte Carlo paths
SEED = 0

AGGRESSIVENESS = 0.7

FIG_DIR = "figures"


def simulate_prices(S0, sigma, T, dt, n_paths, seed=0):
    rng = np.random.default_rng(seed)
    eps = rng.standard_normal((n_paths, T))
    dS = sigma * np.sqrt(dt) * eps
    S = np.zeros((n_paths, T + 1), dtype=float)
    S[:, 0] = S0
    S[:, 1:] = S0 + np.cumsum(dS, axis=1)
    return S


def compute_dp_policy(Q, T, eta, lambda_risk, sigma, dt):
    V = np.full((T + 1, Q + 1), np.inf, dtype=float)
    pi = np.zeros((T, Q + 1), dtype=int)

    V[T, 0] = 0.0   # the order must be completed: V[T, q > 0] = +inf

    for t in range(T - 1, -1, -1):
        for q in range(Q + 1):
            best_val = np.inf
            best_x = 0
            for x in range(q + 1):
                # risk is charged on the inventory held AFTER trading (q - x):
                # that is the position exposed to the next price move.
                cost = eta * (x ** 2) + lambda_risk * (sigma ** 2) * ((q - x) ** 2) * dt + V[t + 1, q - x]
                if cost < best_val:
                    best_val = cost
                    best_x = x
            V[t, q] = best_val
            pi[t, q] = best_x

    return pi, V


def reconstruct_dp_schedule(pi, Q, T):
    q = Q
    x = np.zeros(T, dtype=float)
    for t in range(T):
        xt = int(pi[t, q])
        x[t] = xt
        q -= xt
    return x


def closed_form_schedule(Q, T, eta, lambda_risk, sigma, dt):
    """Analytic solution (Almgren-Chriss, discrete time), used to check the DP.

    q_t = Q sinh(kappa (T - t)) / sinh(kappa T),  cosh(kappa) = 1 + lambda sigma^2 dt / (2 eta)
    """
    kappa = np.arccosh(1 + lambda_risk * sigma ** 2 * dt / (2 * eta))
    t = np.arange(T + 1)
    q = Q * np.sinh(kappa * (T - t)) / np.sinh(kappa * T)
    return -np.diff(q)


def uniform_strategy(Q, T):
    x = np.full(T, Q / T, dtype=float)
    x[-1] += Q - x.sum()
    return x


def adaptive_strategy(price_path, Q, T, aggressiveness=0.7):
    # Only uses prices up to time t (no look-ahead).
    x = np.zeros(T, dtype=float)
    q = float(Q)
    for t in range(T):
        remaining_steps = T - t
        baseline = q / remaining_steps
        mean_so_far = float(np.mean(price_path[: t + 1]))
        signal = (mean_so_far - price_path[t])  # positive => price is "cheap"
        factor = 1.0 + aggressiveness * np.tanh(signal)
        xt = baseline * factor
        xt = float(np.clip(xt, 0.0, q))
        x[t] = xt
        q -= xt
    x[-1] += Q - x.sum()
    return x


def execution_cost(price_path, x, eta):
    # Buy at P_t = S_t + eta*x_t
    T = len(x)
    return float(np.sum(x * (price_path[:T] + eta * x)))


def impact_cost(x, eta):
    return float(np.sum(eta * (x ** 2)))


def inventory_curve(x, Q):
    q = [float(Q)]
    rem = float(Q)
    for xt in x:
        rem -= xt
        q.append(rem)
    return np.array(q, dtype=float)


def theoretical_variance(x, Q, sigma, dt):
    """Var[cost] = sigma^2 * dt * sum_t q_{t+1}^2 (exact for a fixed schedule)."""
    q_after = inventory_curve(x, Q)[1:]
    return float(sigma ** 2 * dt * np.sum(q_after ** 2))


def evaluate_strategies(S_paths, schedules, eta, Q, S0, lambda_risk):
    """Mean-variance comparison. A schedule is either a fixed array x, or a function
    price_path -> x (adaptive strategy, recomputed on every path)."""
    results = {}
    for name, strat in schedules.items():
        costs = np.empty(S_paths.shape[0])
        for i, path in enumerate(S_paths):
            x = strat(path) if callable(strat) else strat
            costs[i] = execution_cost(path, x, eta)
        excess = costs - Q * S0          # cost above the arrival price Q*S0
        results[name] = {
            "mean_excess_cost": float(excess.mean()),
            "std_cost": float(excess.std(ddof=1)),
            "objective": float(excess.mean() + lambda_risk * excess.var(ddof=1)),
        }
    return results


def efficient_frontier(lambdas, Q, T, eta, sigma, dt):
    """For each lambda: optimal schedule -> (expected impact cost, std of cost)."""
    points = []
    for lam in lambdas:
        pi, _ = compute_dp_policy(Q, T, eta, lam, sigma, dt)
        x = reconstruct_dp_schedule(pi, Q, T)
        points.append((impact_cost(x, eta), np.sqrt(theoretical_variance(x, Q, sigma, dt))))
    return np.array(points)


def save(fig_name):
    os.makedirs(FIG_DIR, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, fig_name), dpi=150)  # file used by the README
    plt.show()                                             # display on screen


def plot_schedules(schedules, Q):
    plt.figure()
    for name, x in schedules.items():
        plt.plot(inventory_curve(x, Q), label=name)
    plt.xlabel("t")
    plt.ylabel("Remaining inventory q_t")
    plt.title("Remaining inventory")
    plt.legend()
    save("inventory.png")


def plot_frontier(frontier, others):
    plt.figure()
    plt.plot(frontier[:, 1], frontier[:, 0], "o-", label="DP (lambda varies)")
    for name, (mean, std) in others.items():
        plt.plot(std, mean, "s", markersize=9, label=name)
    plt.xlabel("Std of execution cost")
    plt.ylabel("Expected cost above Q*S0")
    plt.title("Mean-variance efficient frontier")
    plt.legend()
    save("efficient_frontier.png")


def main():
    # DP policy + check against the analytic solution
    pi, V = compute_dp_policy(Q=Q, T=T, eta=eta, lambda_risk=lambda_risk, sigma=sigma, dt=dt)
    x_dp = reconstruct_dp_schedule(pi, Q, T)
    x_cf = closed_form_schedule(Q, T, eta, lambda_risk, sigma, dt)
    gap = np.max(np.abs(inventory_curve(x_dp, Q) - inventory_curve(x_cf, Q)))

    # Benchmarks
    x_uniform = uniform_strategy(Q, T)

    # Simulate
    S_paths = simulate_prices(S0=S0, sigma=sigma, T=T, dt=dt, n_paths=N_PATHS, seed=SEED)

    schedules = {
        "DP_optimal": x_dp,
        "Uniform": x_uniform,
        "Adaptive": lambda path: adaptive_strategy(path, Q, T, aggressiveness=AGGRESSIVENESS),
    }
    results = evaluate_strategies(S_paths, schedules, eta=eta, Q=Q, S0=S0, lambda_risk=lambda_risk)

    print("=== PARAMETERS ===")
    print(f"Q={Q}, T={T}, dt={dt:.4f}, sigma={sigma}, eta={eta}, lambda_risk={lambda_risk}")
    print(f"\nDP vs closed form: max inventory gap = {gap:.2f} shares (integer grid)")
    print("\n=== RESULTS (Monte Carlo, cost measured above Q*S0) ===")
    print(f"{'Strategy':<12}{'mean':>10}{'std':>10}{'mean + lambda*var':>20}")
    for name, r in results.items():
        print(f"{name:<12}{r['mean_excess_cost']:>10.2f}{r['std_cost']:>10.2f}{r['objective']:>20.2f}")

    # Plots
    plot_schedules({"DP_optimal": x_dp, "Uniform": x_uniform,
                    "Adaptive (path 0)": schedules["Adaptive"](S_paths[0])}, Q)
    frontier = efficient_frontier(np.logspace(-4, 0, 12), Q, T, eta, sigma, dt)
    plot_frontier(frontier, {name: (r["mean_excess_cost"], r["std_cost"])
                             for name, r in results.items() if name != "DP_optimal"})
    print(f"\nFigures saved in ./{FIG_DIR}/")


if __name__ == "__main__":
    main()
