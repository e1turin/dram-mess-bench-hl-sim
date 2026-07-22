# %%

import statistics as stats
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from simpy import Environment, Resource
from simpy.core import SimTime

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
    BASE_LATENCY = 70 * NS # 65-75ns

    def __init__(self, env: Environment, gen_lat, channels: int = 1):
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
        return np.random.normal(loc=Dram.BASE_LATENCY)

# %%


class Cpu:
    FREQ = 3.3  # GHz
    FREQ__HZ = FREQ * 1e9
    N_INST_LOOP = 100000

    def __init__(self, dram: Dram, id, freq: float, rng: np.random.Generator):
        self._dram = dram
        self._id = id
        self._time = SECOND / freq * self.N_INST_LOOP
        self._latencies = []
        self._rng = rng

    def run(self, env: Environment):
        while True:
            env.process(self._dram.read(self._id))
            yield env.timeout(self._read_delay())

    def _read_delay(self):
        return self._rng.normal(loc=self._time)
        return self._rng.exponential(scale=self._time)


# %%



@dataclass
class SimResult:
    lat_avg: float
    tput: float
    max_queue: int


def run_simulation(
    n_cpu: int, run_time: float = SECOND // 1_000
) -> SimResult:
    """Run a simulation with *n_cpu* CPUs."""
    env = Environment()
    dram = Dram(env, Dram.gen_lat, channels=2)
    cpus = [
        Cpu(dram=dram, id=i, freq=Cpu.FREQ__HZ, rng=np.random.default_rng(i))
        for i in range(n_cpu)
    ]
    use(env.process(c.run(env)) for c in cpus)
    env.run(until=run_time)

    latencies = dram._latencies
    lat_avg = stats.mean(lat for _, lat in latencies)
    tput = len(latencies) / run_time
    return SimResult(
        lat_avg=lat_avg,
        tput=tput,
        max_queue=dram._max_queue,
    )


# %%

cpu_counts = list(range(1, 17))
results: dict[int, SimResult] = {n: run_simulation(n) for n in cpu_counts}

# %%

def plot_max_queue_vs_cpus(results: dict[int, SimResult]):
    """Max queue length vs CPU count."""
    cpu_ns = sorted(results)
    max_queues = [results[n].max_queue for n in cpu_ns]
    fig, ax = plt.subplots()
    ax.plot(cpu_ns, max_queues, "o-")
    ax.set_title("Max DRAM Queue Length vs CPU Count")
    ax.set_xlabel("Number of CPUs")
    ax.set_ylabel("Max Queue Length")
    fig.tight_layout()
    plt.show()


plot_max_queue_vs_cpus(results)



# %%

def plot_latency_vs_throughput(results: dict[int, SimResult]):
    """Latency vs throughput for varying CPU counts."""
    fig, ax = plt.subplots()
    sorted_items = sorted(results.items())
    tputs = [r.tput for _, r in sorted_items]
    lats = [r.lat_avg for _, r in sorted_items]
    ax.plot(tputs, lats, "o-")
    for n_cpu, tput, lat in zip([n for n, _ in sorted_items], tputs, lats):
        ax.annotate(str(n_cpu), (tput, lat), textcoords="offset points", xytext=(8, -2))
    ax.set_title("Latency vs Throughput")
    ax.set_xlabel("Throughput (req/tick)")
    ax.set_ylabel("Avg Latency (sim ticks)")
    fig.tight_layout()
    plt.show()


plot_latency_vs_throughput(results)
