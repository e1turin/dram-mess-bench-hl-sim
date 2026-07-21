# %% [markdown]
# # M/M/1 Queue Simulation — Latency vs Throughput
#
# We simulate a single-server queue (M/M/1) with:
# - **Poisson arrivals** at rate λ (requests/sec)
# - **Exponential service times** at rate μ (service capacity)
# - **Traffic intensity** ρ = λ/μ
#
# Two regimes are explored:
# - **Normal** (ρ < 1): queue is stable, latency is finite
# - **Overloaded** (ρ ≥ 1): queue grows without bound, latency → ∞
#
# We compare the analytical M/M/1 formulas against a discrete-event simulation.

# %% [markdown]
# ## Cell 1 — Imports and constants

# %%
import heapq
import dataclasses as dc
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt

RNG_SEED = 42
SIM_DURATION = 50_000  # seconds of simulated time
WARMUP = 5_000  # discard initial transient
NUM_SAMPLES = 80  # resolution of λ sweep


# %% [markdown]
# ## Cell 2 — Analytical M/M/1 model
#
# For an M/M/1 queue:
#
# | Metric | Formula |
# |--------|---------|
# | Avg latency (response time) | W = 1 / (μ − λ) |
# | Avg queue length | L = ρ / (1 − ρ) |
# | Throughput | X = min(λ, μ) (in steady state) |

# %%
def mm1_latency(lam: float, mu: float) -> float:
    """Analytical average response time for M/M/1."""
    if lam >= mu:
        return float("inf")
    return 1.0 / (mu - lam)


def mm1_throughput(lam: float, mu: float) -> float:
    """Throughput = arrival rate capped at service rate."""
    return min(lam, mu)


# %% [markdown]
# ## Cell 3 — Discrete-event simulation engine
#
# We build a minimal event-driven simulator:
# - `arrival` events are scheduled with exponential inter-arrival times
# - `departure` events are scheduled after service completion
# - We track per-request latency after warmup

# %%
@dc.dataclass(slots=True)
class SimResult:
    latencies: np.ndarray  # seconds per completed request
    throughput: float  # requests / sec (in measurement window)


def simulate_mm1(lam: float, mu: float, rng: np.random.Generator) -> SimResult:
    """
    Discrete-event M/M/1 simulation.

    Parameters
    ----------
    lam : float — arrival rate (req/s)
    mu : float — service rate (req/s)
    rng : numpy random generator

    Returns
    -------
    SimResult with measured latencies and throughput.
    """
    # Event queue: (time, event_type, payload)
    # event_type: 'arrival' or 'departure'
    events: list = []
    t = 0.0

    # Schedule first arrival
    inter = rng.exponential(1.0 / lam)
    heapq.heappush(events, (t + inter, "arrival", None))

    server_busy_until = 0.0
    queue: list[float] = []  # arrival times of waiting requests

    completed_lats: list[float] = []

    while t < SIM_DURATION:
        t, etype, _ = heapq.heappop(events)

        if etype == "arrival":
            # Schedule next arrival
            inter = rng.exponential(1.0 / lam)
            heapq.heappush(events, (t + inter, "arrival", None))

            if t >= server_busy_until:
                # Server idle — depart immediately after service
                svc = rng.exponential(1.0 / mu)
                server_busy_until = t + svc
                heapq.heappush(events, (server_busy_until, "departure", t))
            else:
                # Server busy — enqueue
                queue.append(t)

        elif etype == "departure":
            arr_time = _  # arrival time of the departing request
            if t >= WARMUP:
                completed_lats.append(t - arr_time)

            if queue:
                next_arr = queue.pop(0)
                svc = rng.exponential(1.0 / mu)
                server_busy_until = t + svc
                heapq.heappush(events, (server_busy_until, "departure", next_arr))

    lat_arr = np.array(completed_lats)
    meas_window = SIM_DURATION - WARMUP
    throughput = len(lat_arr) / meas_window
    return SimResult(latencies=lat_arr, throughput=throughput)


# %% [markdown]
# ## Cell 4 — Run sweep over arrival rates
#
# We sweep λ from a small value up to 2×μ (well into overload) and collect
# both analytical and simulated latency + throughput pairs.

# %%
MU = 100.0  # service rate: 100 req/s (capacity)
LAM_MIN = 5.0
LAM_MAX = 200.0  # 2× overload

lam_values = np.linspace(LAM_MIN, LAM_MAX, NUM_SAMPLES)

# Analytical curves
analytical_throughput = np.array([mm1_throughput(l, MU) for l in lam_values])
analytical_latency = np.array([mm1_latency(l, MU) for l in lam_values])

# Simulation
rng = np.random.default_rng(RNG_SEED)
sim_throughput = np.empty_like(lam_values)
sim_latency = np.empty_like(lam_values)

for i, lam in enumerate(lam_values):
    res = simulate_mm1(lam, MU, rng)
    sim_throughput[i] = res.throughput
    sim_latency[i] = np.mean(res.latencies) if len(res.latencies) > 0 else np.nan
    if (i + 1) % 10 == 0:
        print(f"  λ = {lam:6.1f}  sim-throughput = {sim_throughput[i]:7.1f}  "
              f"sim-latency = {sim_latency[i]:8.4f}s")


# %% [markdown]
# ## Cell 5 — Plot: Latency vs Throughput
#
# The classic "bathtub" curve:
# - Throughput rises linearly with λ up to μ (saturation)
# - Latency stays low in the normal regime, then spikes near μ
# - In overload (ρ > 1) throughput plateaus at μ while latency → ∞

# %%
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# ── Left: Latency vs Throughput ──────────────────────────────────────
ax = axes[0]
# Mask infinite analytical latencies for plotting
mask_an = analytical_latency < 1e6

ax.plot(analytical_throughput[mask_an], analytical_latency[mask_an],
        "b-", linewidth=2, label="Analytical M/M/1")
ax.plot(sim_throughput, sim_latency, "ro", markersize=4, alpha=0.7,
        label="Simulation")

ax.axvline(MU, color="gray", linestyle="--", alpha=0.6, label=f"μ = {MU:.0f}")
ax.set_xlabel("Throughput (req/s)", fontsize=12)
ax.set_ylabel("Avg Latency (s)", fontsize=12)
ax.set_title("Latency vs Throughput", fontsize=13)
ax.set_ylim(0, 2.0)  # clip for readability; overload tails go higher
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# ── Right: Latency vs Arrival Rate (ρ) ──────────────────────────────
ax = axes[1]
rho = lam_values / MU
mask_sim = sim_latency < 1e6

ax.plot(rho[mask_an], analytical_latency[mask_an],
        "b-", linewidth=2, label="Analytical M/M/1")
ax.plot(rho[mask_sim], sim_latency[mask_sim], "ro", markersize=4, alpha=0.7,
        label="Simulation")

ax.axvline(1.0, color="red", linestyle="--", alpha=0.6, label="ρ = 1 (saturation)")
ax.set_xlabel("Traffic intensity ρ = λ/μ", fontsize=12)
ax.set_ylabel("Avg Latency (s)", fontsize=12)
ax.set_title("Latency vs Traffic Intensity", fontsize=13)
ax.set_ylim(0, 2.0)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("queue_latency_throughput.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved → queue_latency_throughput.png")


# %% [markdown]
# ## Cell 6 — Summary
#
# | Regime | ρ | Throughput | Latency |
# |--------|---|------------|---------|
# | Normal | < 1 | ≈ λ (linear growth) | 1/(μ − λ), low & stable |
# | Critical | ≈ 1 | ≈ μ (saturated) | spikes sharply |
# | Overloaded | > 1 | capped at μ | grows toward ∞ (queue explodes) |
#
# The simulation closely tracks the analytical M/M/1 curves, confirming the
# model. In the overloaded regime the simulation may show large variance due
# to finite simulation time — real systems would shed load or fail rather
# than let latency grow without bound.

# %%
# Print numeric summary at key operating points
key_rhos = [0.5, 0.9, 0.99, 1.0, 1.1, 1.5]
print(f"\n{'ρ':>6}  {'λ':>8}  {'Analytical W':>14}  {'Sim W':>10}  {'Sim X':>10}")
print("-" * 56)
for rho in key_rhos:
    lam = rho * MU
    a_w = mm1_latency(lam, MU)
    if lam >= LAM_MAX or a_w > 100:
        a_w_str = "∞"
    else:
        a_w_str = f"{a_w:.4f}"

    # Simulate at this ρ if within range
    if lam <= LAM_MAX:
        res = simulate_mm1(lam, MU, rng)
        s_w = f"{np.mean(res.latencies):.4f}"
        s_x = f"{res.throughput:.1f}"
    else:
        s_w = "—"
        s_x = "—"

    print(f"{rho:6.2f}  {lam:8.1f}  {a_w_str:>14}  {s_w:>10}  {s_x:>10}")
