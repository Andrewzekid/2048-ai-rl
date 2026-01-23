import math
class LinearDecay:
    def __init__(self,epsilon_start,epsilon_end,maxsteps):
        """Decays epsilon linearly"""
        self.epsilon_end = epsilon_end
        self.epsilon_start = epsilon_start
        self.maxsteps = maxsteps
        self.current_steps = 0
        self.decrement = (self.epsilon_end - self.epsilon_start) / maxsteps
    def decay(self,epsilon):
        if self.current_steps > self.maxsteps:
            if epsilon == self.epsilon_end:
                return epsilon
            else:
                return self.epsilon_end
        else:
            self.current_steps += 1
            return epsilon + self.decrement

class ExponentialDecay(LinearDecay):
    def __init__(self,epsilon_start,epsilon_end,maxsteps):
        """Decays epsilon exponentially. Epsilon reaches epsilon_end by maxsteps"""
        super().__init__(epsilon_start=epsilon_start,epsilon_end=epsilon_end,maxsteps=maxsteps)
        self.constant = math.log(epsilon_end / epsilon_start) / maxsteps
    def decay(self,epsilon):
        epsilon = self.epsilon_start * math.exp(self.constant * self.current_steps)
        self.current_steps += 1
        return epsilon