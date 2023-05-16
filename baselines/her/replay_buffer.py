import threading
import numpy as np
from baselines.common.segment_tree import SumSegmentTree, MinSegmentTree
import math
from scipy.stats import rankdata
import json

class ReplayBuffer:
    def __init__(self, buffer_shapes, size_in_transitions, T, sample_transitions):
        """Creates a replay buffer.

        Args:
            buffer_shapes (dict of ints): the shape for all buffers that are used in the replay
                buffer
            size_in_transitions (int): the size of the buffer, measured in transitions
            T (int): the time horizon for episodes
            sample_transitions (function): a function that samples from the replay buffer
        """
        self.buffer_shapes = buffer_shapes
        self.size = size_in_transitions // T
        self.T = T
        self.sample_transitions = sample_transitions
        # self.buffers is {key: array(size_in_episodes x T or T+1 x dim_key)}
        self.buffers = {key: np.empty([self.size, *shape])
                        for key, shape in buffer_shapes.items()}
        # memory management
        self.current_size = 0
        self.n_transitions_stored = 0
        self.lock = threading.Lock()

    @property
    def full(self):
        with self.lock:
            return self.current_size == self.size



    def sample(self, batch_size):
        """Returns a dict {key: array(batch_size x shapes[key])}
        """
        buffers = {}
        with self.lock:
            assert self.current_size > 0
            for key in self.buffers.keys():
                buffers[key] = self.buffers[key][:self.current_size]
        buffers['o_2'] = buffers['o'][:, 1:, :]
        buffers['ag_2'] = buffers['ag'][:, 1:, :]
        transitions = self.sample_transitions(buffers, batch_size)
        for key in (['r', 'o_2', 'ag_2'] + list(self.buffers.keys())):
            assert key in transitions, "key %s missing from transitions" % key
        return transitions

    def store_episode(self, episode_batch):
        """episode_batch: array(batch_size x (T or T+1) x dim_key)
        """
        batch_sizes = [len(episode_batch[key]) for key in episode_batch.keys()]
        assert np.all(np.array(batch_sizes) == batch_sizes[0])
        batch_size = batch_sizes[0]
        with self.lock:
            idxs = self._get_storage_idx(batch_size)
            # load inputs into buffers
            for key in self.buffers.keys():
                self.buffers[key][idxs] = episode_batch[key]
            self.n_transitions_stored += batch_size * self.T

    def get_current_episode_size(self):
        with self.lock:
            return self.current_size

    def get_current_size(self):
        with self.lock:
            return self.current_size * self.T

    def get_transitions_stored(self):
        with self.lock:
            return self.n_transitions_stored

    def clear_buffer(self):
        with self.lock:
            self.current_size = 0

    def _get_storage_idx(self, inc=None):
        inc = inc or 1   # size increment
        assert inc <= self.size, "Batch committed to replay is too large!"
        # go consecutively until you hit the end, and then go randomly.
        if self.current_size+inc <= self.size:
            idx = np.arange(self.current_size, self.current_size+inc)
        elif self.current_size < self.size:
            overflow = inc - (self.size - self.current_size)
            idx_a = np.arange(self.current_size, self.size)
            idx_b = np.random.randint(0, self.current_size, overflow)
            idx = np.concatenate([idx_a, idx_b])
        else:
            idx = np.random.randint(0, self.size, inc)
        # update replay size
        self.current_size = min(self.size, self.current_size+inc)
        if inc == 1:
            idx = idx[0]
        return idx

class ReplayBufferMotivation:
    def __init__(self, buffer_shapes, size_in_transitions, T, sample_transitions, prioritization, env_name, goal_type):
        """
        Creates a replay buffer for measuring the diversity
        Args:
            buffer_shapes (dict of ints): the shape for all buffers that are used in the replay
                buffer
            size_in_transitions (int): the size of the buffer, measured in transitions
            T (int): the time horizon for episodes
            sample_transitions (function): a function that samples from the replay buffer
        """
        self.buffer_shapes = buffer_shapes
        self.size = size_in_transitions // T
        self.T = T
        self.sample_transitions = sample_transitions
        self.buffers = {key: np.empty([self.size, *shape])
                        for key, shape in buffer_shapes.items()}
        self.buffers['mot'] = np.empty([self.size, 1])
        self.buffers['init_position'] = np.empty([self.size, 51,3]) 
        self.buffers['ag_last'] = np.empty([self.size, 51,3]) # last ag
        # the prioritization is dpp now
        self.prioritization = prioritization
        self.env_name = env_name
        # memory management
        self.current_size = 0
        self.n_transitions_stored = 0
        self.current_size_test = 0
        self.n_transitions_stored_test = 0
        self.goal_type = goal_type
        self.lock = threading.Lock()

    @property
    def full(self):
        with self.lock:
            return self.current_size == self.size

    def sample(self, batch_size):
        """Returns a dict {key: array(batch_size x shapes[key])}
        """
        buffers = {}
        with self.lock:
            assert self.current_size > 0
            for key in self.buffers.keys():
                buffers[key] = self.buffers[key][:self.current_size]
        buffers['o_2'] = buffers['o'][:, 1:, :]
        buffers['ag_2'] = buffers['ag'][:, 1:, :]
        transitions = self.sample_transitions(buffers, batch_size)
        return transitions

    def store_episode(self, episode_batch):
        """episode_batch: array(batch_size x (T or T+1) x dim_key)
        """
        def goal_distance(goal_a, goal_b):
                assert goal_a.shape == goal_b.shape
                return np.linalg.norm(goal_a - goal_b, axis=-1)

        batch_sizes = [len(episode_batch[key]) for key in episode_batch.keys()]
        assert np.all(np.array(batch_sizes) == batch_sizes[0])
        batch_size = batch_sizes[0]
        buffers = {}
        for key in episode_batch.keys():
            buffers[key] = episode_batch[key]


        # start to calculate the diversity
        if self.prioritization == 'motivation' or self.prioritization == 'none': 
            # we only consider the fetch environment now
            if self.goal_type == 'full':
                traj = buffers['ag'].copy().astype(np.float32)
                traj_g = buffers['g'].copy().astype(np.float32)
            elif self.goal_type == 'rotate':
                # if use the rotate...
                traj = buffers['ag'][:, :, 3:].copy().astype(np.float32)
                traj_g = buffers['g'].copy().astype(np.float32)
            else:
                raise NotImplementedError 

            traj_g = traj_g[:,0,:] 
            distance = []
            all_close = []
            for m in range(traj_g.shape[0]):
                for n in range(traj.shape[1]):
                    #print(traj[m, n, :].shape, traj_g[m,:].shape)
                    result = goal_distance(traj[m, n, :], traj_g[m,:])
                    distance.append(result)
                distance_b = distance[1:]
                distance_f = distance[:-1]
                distance_n = np.array(distance_f) - np.array(distance_b)
                distance_n[distance_n < 0] = 0
                distance = []
                all_close.append(sum(distance_n))
            episode_batch['mot'] = np.clip(np.array(all_close).reshape(-1, 1), 0, 2)


            ag_list= traj[:,:-1,:]

            # FetchPush
            # b = episode_batch['o'][:,0,3:6]

            # PandaPickAndPlaceJoints
            b = episode_batch['o'][:,0,7:10]
            
            # PandaPush
            # b = episode_batch['o'][:,0,6:9]

            ag_last = np.insert(ag_list, 0, values=b, axis=1)
            init_list = np.array(b)
            e = np.expand_dims(init_list,1).repeat(51,axis=1)
            episode_batch['init_position'] = np.array(e)
            episode_batch['ag_last'] = np.array(ag_last)

        with self.lock:
            idxs = self._get_storage_idx(batch_size)
            # load inputs into buffers
            for key in self.buffers.keys():
                self.buffers[key][idxs] = episode_batch[key]
            self.n_transitions_stored += batch_size * self.T

    def get_current_episode_size(self):
        with self.lock:
            return self.current_size

    def get_current_size(self):
        with self.lock:
            return self.current_size * self.T

    def get_transitions_stored(self):
        with self.lock:
            return self.n_transitions_stored

    def clear_buffer(self):
        with self.lock:
            self.current_size = 0

    def _get_storage_idx(self, inc=None):
        inc = inc or 1   # size increment
        assert inc <= self.size, "Batch committed to replay is too large!"
        # go consecutively until you hit the end, and then go randomly.
        if self.current_size+inc <= self.size:
            idx = np.arange(self.current_size, self.current_size+inc)
        elif self.current_size < self.size:
            overflow = inc - (self.size - self.current_size)
            idx_a = np.arange(self.current_size, self.size)
            idx_b = np.random.randint(0, self.current_size, overflow)
            idx = np.concatenate([idx_a, idx_b])
        else:
            # here we shouldn't randomly pick the trajectory to replace
            idx = np.random.randint(0, self.size, inc)
        # update replay size
        self.current_size = min(self.size, self.current_size+inc)
        if inc == 1:
            idx = idx[0]
        return idx
