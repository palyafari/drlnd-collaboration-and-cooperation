import numpy as np
import random

from collections import namedtuple, deque

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "all_states", "action", "other_action", "reward", "next_state", "all_next_states", "done"])
        np.random.seed(seed)
        random.seed(seed)
    
    def add(self, state, all_states, action, other_action, reward, next_state, all_next_states, done):
        """Add a new experience to memory."""
        e = self.experience(state, all_states, action, other_action, reward, next_state, all_next_states, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        return random.sample(self.memory, k=self.batch_size)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

class NaivePrioritizedReplayBuffer:
    """Fixed-size buffer to store experience tuples prioritized. This is a naive implementation using ring a ring buffer."""

    def __init__(self, action_size, buffer_size, batch_size, seed, alpha, epsilon):
        """Initialize a NaivePrioritizedReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.priorities = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "all_states", "action", "other_action", "reward", "next_state", "all_next_states", "done"])
        np.random.seed(seed)
        self.alpha = alpha
        self.epsilon = epsilon
        self.indices = []
    
    def add(self, state, all_states, action, other_action, reward, next_state, all_next_states, done):
        """Add a new experience to memory."""

        e = self.experience(state, all_states, action, other_action, reward, next_state, all_next_states, done)
        self.memory.append(e)
        if len(self.priorities) > 0:
            max_prio = max(self.priorities)
            self.priorities.append(max_prio)
        else:
            self.priorities.append(1.0)

        assert(len(self.memory)==len(self.priorities))
    
    def sample(self, beta):
        """Sample a batch of experiences from memory with the priorities"""

        N = len(self.priorities)
        # sample the indices
        probs = np.array(self.priorities) ** self.alpha
        probs /= sum(probs)
        self.indices = np.random.choice(range(len(self.priorities)), size=self.batch_size, p=probs)
        weights  = (N * probs[self.indices]) ** (-beta)
        weights /= max(weights)
        return ([self.memory[i] for i in self.indices], weights)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

    def update_priorities(self, priorities):
        """Update the priorities of the selected experiences"""
        for i,p in zip(self.indices, priorities):
            self.priorities[i] = abs(p) + self.epsilon