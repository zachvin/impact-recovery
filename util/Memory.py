import numpy as np
import torch as T

class Memory():
    def __init__(self, device, state_shape, mem_size=20000):
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

        self.minv = np.inf
        self.maxv = -np.inf

        self.mina = np.inf
        self.maxa = -np.inf

        self.state      = np.zeros((self.mem_size, state_shape), dtype=np.float32)
        self.state_     = np.zeros((self.mem_size, state_shape), dtype=np.float32)
        self.reward     = np.zeros(self.mem_size, dtype=np.float32)
        self.done       = np.zeros(self.mem_size, dtype=np.bool_)

    def append(self, obs, obs_, reward, done, clip=False):
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

        if clip:
            pos     = obs[0:3]
            rpy     = obs[3:6]
            vel     = obs[6:9]
            ang_v   = obs[9:12]

            self.minv = min(self.minv, vel[0])
            self.mina = min(self.mina, ang_v[0])

            self.maxv = max(self.maxv, vel[0])
            self.maxa = max(self.maxa, ang_v[0])

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

