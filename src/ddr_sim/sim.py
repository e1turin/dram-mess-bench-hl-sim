# %%

import statistics as stats
from multiprocessing.pool import ThreadPool

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from simpy import Environment, Resource
from simpy.core import SimTime
from tqdm import tqdm

SIM_TIME_UNIT: SimTime = 10  # ticks
NS = 1 * SIM_TIME_UNIT
US = 1000 * NS
MS = 1000 * US
SECOND = 1000 * MS

N_CPU = 4
RUN_TIME = SECOND // 1_000_000 

# %%

def use(g):
    for _ in g:
        ...


# %%


class Dram:
    BASE_LATENCY = (
        70 * NS # 65-75ns
        # 90 * NS # 65-75ns
    )
    CHANNELS = (
        1
        # 2
    )

    def __init__(self, env: Environment, gen_lat, channels: int = CHANNELS):
        self._env = env
        self._res = Resource(env, capacity=channels)
        self._lat = gen_lat
        self._latencies: list[tuple[int, float]] = []
        self._max_queue = 0
        self._latencies = []

    def read(self, cpu_id: int):
        with self._res.request() as req:
            read_begin = self._env.now
            
            self._max_queue = max(self._max_queue, len(self._res.queue))
            yield req
            yield self._env.timeout(self._lat())
            
            read_end = self._env.now
        
        lat = read_end - read_begin
        self._latencies.append((cpu_id, lat))

    @staticmethod
    def gen_lat():
        return Dram.BASE_LATENCY
        return np.random.poisson(lam=Dram.BASE_LATENCY) # like normal but limited by positive values

# %%


class Cpu:
    FREQ = (
        4
        # 3.3  # GHz
    )
    FREQ__HZ = FREQ * 1e9
    N_INST_LOOP = (
        1000
        # 500
        # 350
        # 200
        # 100
    )

    def __init__(
        self,
        dram: Dram,
        id,
        rng: np.random.Generator,
        freq: float = FREQ__HZ,
    ):
        self._dram = dram
        self._id = id
        self._time = SECOND / freq * self.N_INST_LOOP
        self._rng = rng

    def run(self, env: Environment):
        while True:
            env.process(self._dram.read(self._id))
            yield env.timeout(self._read_delay())

    def _read_delay(self):
        # return self._time
        return self._rng.poisson(lam=self._time) # like normal but limited by positive values


# %%


def run_simulation(
    n_cpu: int, run_time: float = 1 * MS
) -> dict:
    """Run a simulation. Returns input params + results as a flat dict."""
    env = Environment()
    dram = Dram(env, Dram.gen_lat)
    cpus = [
        Cpu(dram=dram, id=i, rng=np.random.default_rng(i))
        for i in range(n_cpu)
    ]
    for c in cpus:
        env.process(c.run(env))
        
    env.run(until=run_time)

    latencies = dram._latencies
    lat_avg = stats.mean(lat for _, lat in latencies) if latencies else 0.0
    tput = len(latencies) / run_time
    return {
        "n_cpu": n_cpu,
        "run_time": run_time,
        "dram_channels": dram._res.capacity,
        "lat_avg": lat_avg,
        "tput": tput,
        "max_queue": dram._max_queue,
    }


# %%


# cpu_counts = list(range(1, 17))
cpu_counts = [1, 2, 3, 4, 6, 8, 12, 16, 20]
with ThreadPool() as pool:
    rows = list(tqdm(pool.imap(run_simulation, cpu_counts), total=len(cpu_counts)))
df = pd.DataFrame(rows)
df

# %%

def plot_latency_vs_throughput(df: pd.DataFrame, yscale="linear"):
    """Latency vs throughput for varying CPU counts."""
    fig, ax = plt.subplots()
    ax.plot(df["tput"], df["lat_avg"], "o-")
    for _, row in df.iterrows():
        ax.annotate(
            str(int(row["n_cpu"])),
            (row["tput"], row["lat_avg"]),
            textcoords="offset points",
            xytext=(8, -8),
        )
    ax.set_title("Latency vs Throughput")
    ax.set_xlabel("Throughput (req/tick)")
    ax.set_ylabel("Avg Latency (sim ticks)")
    ax.set_yscale(yscale)
    fig.tight_layout()
    plt.show()


plot_latency_vs_throughput(df) #, "log")

# %%

def plot_max_queue_vs_cpus(df: pd.DataFrame):
    """Max queue length vs CPU count."""
    fig, ax = plt.subplots()
    ax.plot(df["n_cpu"], df["max_queue"], "o-")
    ax.set_title("Max DRAM Queue Length vs CPU Count")
    ax.set_xlabel("Number of CPUs")
    ax.set_ylabel("Max Queue Length")
    fig.tight_layout()
    plt.show()


plot_max_queue_vs_cpus(df)
