import numpy as np
import torch as T

class Memory():
    def __init__(self, device, state_shape, mem_size=50, num_motors=4):
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
        self.new_state  = np.zeros((self.mem_size, state_shape), dtype=np.float32)
        self.action     = np.zeros((self.mem_size, num_motors), dtype=np.float32)
        self.reward     = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal   = np.zeros(self.mem_size, dtype=np.bool_)

    def append(self, obs, action, reward, obs_, term):
        """
        Add a single or batch memory to the Memory buffer.
        Params:
            obs : np.float32
                Current state
            action : np.float32
                Action taken
            reward : np.float32
                Reward for action
            obs_ : np.float32
                State following action
            term : np.bool_
                Whether simulation has terminated or truncated
        """
        num_mems = obs.shape[1]//12

        # add each memory in batch to the buffer
        for j in range(num_mems):
            index = self.i % self.mem_size

            self.state[index]       = np.reshape(obs, (-1, 12))[j]
            self.new_state[index]   = np.reshape(obs_, (-1, 12))[j]
            self.action[index]      = action
            self.reward[index]      = reward
            self.terminal[index]    = term

            self.i += 1

    def sample(self, batch_size=10):
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

        obs     = T.tensor(self.mem_state[batch]).to(self.device)
        obs_    = T.tensor(self.mem_new_state[batch]).to(self.device)
        actions = T.tensor(self.mem_action[batch]).to(self.device)
        rewards = T.tensor(self.mem_reward[batch]).to(self.device)
        terms   = T.tensor(self.mem_terminal[batch]).to(self.device)

        return obs, actions, rewards, obs_, terms

