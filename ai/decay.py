import math
class LinearDecay:
    def __init__(self,epsilon_start,epsilon_end,maxsteps):
        """Decays epsilon linearly"""
        self.epsilon_end = epsilon_end
        self.epsilon = epsilon_start
        self.maxsteps = maxsteps
        self.current_steps = 0
        self.decrement = (epsilon_end - self.epsilon) / maxsteps
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
        """Decays epsilon exponentially"""
        super().__init__(epsilon_start=epsilon_start,epsilon_end=epsilon_end,maxsteps=maxsteps)
        self.decrement = math.log(self.epsilon / self.epsilon_end) / maxsteps
    def decay(self,epsilon):
        b = math.log(epsilon)
        if self.current_steps > self.maxsteps:
            if epsilon == self.epsilon_end:
                return epsilon
            else:
                return self.epsilon_end
        else:
            self.current_steps += 1
            nb = b - self.decrement
            return math.exp(nb)
