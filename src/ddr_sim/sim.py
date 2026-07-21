# %%

from simpy import Environment, Resource
from simpy.core import SimTime

SIM_TIME_UNIT: SimTime = 10  # ticks
NS = 1 * SIM_TIME_UNIT
US = 1000 * NS
MS = 1000 * US
SECOND = 1000 * MS

N_CPU = 4

# %%


class Dram:
    BASE_LATENCY = 10 * NS

    def __init__(self, env: Environment, gen, channels: int = 1):
        self._env = env
        self._res = Resource(env, capacity=channels)
        self._gen = gen

    def read(self):
        with self._res.request() as read:
            yield read
            yield self._env.timeout(self._gen())


# %%


class Cpu:
    FREQ = 3.3  # GHz
    FREQ__HZ = FREQ * 1e9

    def __init__(self, dram: Dram, id, freq: float):
        self._dram = dram
        self._id = id
        self._time = SECOND // freq

    def run(self, env):
        while True:
            yield env.process(self._dram.read())
            yield env.timeout(self._time)


# %%

env = Environment()
dram = Dram(env, lambda: Dram.BASE_LATENCY)
cpus = [Cpu(dram, i, Cpu.FREQ__HZ) for i in range(N_CPU)]

# %%

all(env.process(c.run(env)) for c in cpus)

# %%

env.run(until=SECOND // 1_000_000)

# %%
