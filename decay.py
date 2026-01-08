import math
class LinearDecay:
    def __init__(self,epsilon_start,epsilon_end,maxsteps):
        """Decays epsilon linearly"""
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.maxsteps = maxsteps
        self.current_steps = 0
        self.decrement = (epsilon_end - epsilon_start) / maxsteps
    def decay(self,epsilon):
        if self.current_steps > self.maxsteps:
            if epsilon == self.epsilon_end:
                return epsilon
            else:
                return self.epsilon_end
        else:
            return epsilon + self.decrement

class ExponentialDecay(LinearDecay):
    def __init__(self,epsilon_start,epsilon_end,maxsteps):
        """Decays epsilon exponentially"""
        super().__init__()
        self.base = math.log(epsilon) #Find ln(epsilon)
        self.end_base = math.log(epsilon_end)
        self.decrement = (end_base - base) / maxsteps
    def decay(self,epsilon):
        b = math.log(epsilon)
        if self.current_steps > self.maxsteps:
            if epsilon == self.epsilon_end:
                return epsilon
            else:
                return self.epsilon_end
        else:
            nb = b + self.decrement
            return math.exp(nb)
