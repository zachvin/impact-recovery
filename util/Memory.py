import numpy as np
import torch as T

class Memory():
    def __init__(self, device, state_shape, mem_size=20000, num_motors=4):
        """
        Initialization of Memory buffer object. Includes simple methods for
        saving and retrieving memories.
        Params:
            state_shape : np.array
                Shape of the observation array saved to memory buffer.
            mem_size : int
                Number of memories to be stored in the buffer at any given time.
            num_motors : int
                Number of motors controlled.
        Returns:
            Memory object
        """

        self.device = device

        self.mem_size = mem_size
        self.i = 0

        self.state      = np.zeros((self.mem_size, state_shape), dtype=np.float32)
        self.state_     = np.zeros((self.mem_size, state_shape), dtype=np.float32)
        self.reward     = np.zeros(self.mem_size, dtype=np.float32)
        self.done       = np.zeros(self.mem_size, dtype=np.bool_)

    def append(self, obs, obs_, reward, done):
        """
        Add a single or batch memory to the Memory buffer.
        Params:
            obs : np.float32
                Current state
            obs_ : np.float32
                Next state
            reward : np.float32
                Reward for action
            done : np.bool_
                Whether simulation has terminated or truncated
        """

        # TODO memory interpolation
        index = self.i % self.mem_size

        self.state[index]       = obs
        self.state_[index]      = obs_
        self.reward[index]      = reward
        self.done[index]        = done

        self.i += 1

    def sample(self, batch_size=32):
        """
        Samples Memory buffer and returns batch_size number of memories.
        Params:
            batch_size : int
                Number of memories to be returned at once
        Returns:
            np.array of shape (batch_size, -1)
        """
        max_mem = min(self.i, self.mem_size)

        batch = np.random.choice(max_mem, batch_size, replace=False)

        obs     = self.state[batch]
        obs_    = self.state_[batch]
        rewards = self.reward[batch]
        dones   = self.done[batch]

        return obs, obs_, rewards, dones

