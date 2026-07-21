# %%

import statistics as stats

import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import numpy as np
from simpy import Environment, Resource
from simpy.core import SimTime

SIM_TIME_UNIT: SimTime = 10  # ticks
NS = 1 * SIM_TIME_UNIT
US = 1000 * NS
MS = 1000 * US
SECOND = 1000 * MS

N_CPU = 4

# %%

def use(g):
    for _ in g:
        ...


# %%


class Dram:
    BASE_LATENCY = 10 * NS

    def __init__(self, env: Environment, gen_lat, channels: int = 1):
        self._env = env
        self._res = Resource(env, capacity=channels)
        self._lat = gen_lat
        self._service_latencies = []

    def read(self):
        with self._res.request() as read:
            yield read
            lat = self._lat()
            # lat = self.BASE_LATENCY
            self._service_latencies.append(lat)
            yield self._env.timeout(lat)
            return lat


# %%


class Cpu:
    FREQ = 3.3  # GHz
    FREQ__HZ = FREQ * 1e9

    def __init__(self, dram: Dram, id, freq: float, rng: np.random.Generator):
        self._dram = dram
        self._id = id
        self._time = SECOND // freq
        self._latencies = []
        self._intervals = []
        self._rng = rng

    def run(self, env: Environment):
        while True:
            latency = yield env.process(self._dram.read())
            self._latencies.append((self._id, latency))
            
            interval = self._rng.exponential(scale=self._time)
            self._intervals.append(interval)
            yield env.timeout(interval)

    @property
    def latencies(self):
        return self._latencies

    @property
    def intervals(self):
        return self._intervals


# %%

def gen_lat():
    return np.random.exponential(scale=Dram.BASE_LATENCY)

env = Environment()
dram = Dram(env, gen_lat)
cpus = [
    Cpu(
        dram=dram,
        id=i,
        freq=Cpu.FREQ__HZ,
        rng=np.random.default_rng(i),
    )
    for i in range(N_CPU)
]
use(env.process(c.run(env)) for c in cpus)

# %%
RUN_TIME = SECOND // 1_000_000 
env.run(until=RUN_TIME)

# %%

latencies = []
use(latencies.extend(c.latencies) for c in cpus)

# %%

lat_avg = stats.mean(lat for _, lat in latencies)
lat_avg

# %%

tput_avg = len(latencies) / RUN_TIME

# %%

def plot_dram_latency(service_latencies: list[float]):
    """KDE of DRAM service latencies."""
    x = np.linspace(min(service_latencies), max(service_latencies), 200)
    kde = gaussian_kde(service_latencies)
    fig, ax = plt.subplots()
    ax.plot(x, kde(x))
    ax.fill_between(x, kde(x), alpha=0.3)
    ax.set_title("DRAM Latency Distribution")
    ax.set_xlabel("Latency (sim ticks)")
    ax.set_ylabel("Density")
    ax.axvline(Dram.BASE_LATENCY, color="red", linestyle="--", label=f"mean={Dram.BASE_LATENCY}")
    ax.legend()
    fig.tight_layout()
    plt.show()

plot_dram_latency(dram._service_latencies)


# %%

def plot_cpu_request_intensity(cpus: list[Cpu]):
    """KDE of CPU request inter-arrival intensities."""
    fig, ax = plt.subplots()
    for c in cpus:
        intervals = np.array(c.intervals)
        if len(intervals) < 2:
            continue
        x = np.linspace(intervals.min(), intervals.max(), 200)
        kde = gaussian_kde(intervals)
        ax.plot(x, kde(x), label=f"CPU {c._id}")
        ax.fill_between(x, kde(x), alpha=0.2)
    ax.set_title("CPU Request Intensity Distribution")
    ax.set_xlabel("Inter-arrival time (sim ticks)")
    ax.set_ylabel("Density")
    ax.legend()
    fig.tight_layout()
    plt.show()
    
plot_cpu_request_intensity(cpus)
