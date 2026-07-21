# %%

import statistics as stats

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

    def read(self):
        with self._res.request() as read:
            yield read
            yield self._env.timeout(self._lat())


# %%


class Cpu:
    FREQ = 3.3  # GHz
    FREQ__HZ = FREQ * 1e9

    def __init__(self, dram: Dram, id, freq: float, rng: np.random.Generator):
        self._dram = dram
        self._id = id
        self._time = SECOND // freq
        self._latencies = []
        self._rng = rng

    def run(self, env: Environment):
        while True:
            start_read = env.now
            yield env.process(self._dram.read())
            end_read = env.now
            latency = end_read - start_read
            self._latencies.append((self._id, latency))
            yield env.timeout(self._rng.exponential(scale=self._time))

    @property
    def latencies(self):
        return self._latencies


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
tput_avg
