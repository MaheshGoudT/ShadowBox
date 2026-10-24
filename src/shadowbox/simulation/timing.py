"""Advance simulation time independently of how often a frame is rendered."""


class SimulationClock:
    def __init__(self, timestep, max_catchup=1.0):
        self.timestep = timestep
        self.max_catchup = max_catchup
        self.pending = 0.0

    def steps(self, elapsed, active=True):
        if not active:
            self.pending = 0.0
            return 0
        self.pending += min(max(elapsed, 0.0), self.max_catchup)
        count = int((self.pending + 1e-10) / self.timestep)
        self.pending -= count * self.timestep
        return count
