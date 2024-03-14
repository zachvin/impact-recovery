import numpy as np

class Memory():
    def __init__(self, state_shape, mem_size=50, num_motors=4):
        self.mem_size = mem_size

        self.state      = np.zeros((self.mem_size, state_shape), dtype=np.float32)
        self.new_state  = np.zeros((self.mem_size, state_shape), dtype=np.float32)
        self.action     = np.zeros((self.mem_size, num_motors), dtype=np.float32)
        self.reward     = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal   = np.zeros(self.mem_size, dtype=np.bool_)

