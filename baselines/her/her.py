import numpy as np
from scipy.stats import rankdata
import random

def make_sample_her_transitions(replay_strategy, replay_k, reward_fun):
    """Creates a sample function that can be used for HER experience replay.

    Args:
        replay_strategy (in ['future', 'none']): the HER replay strategy; if set to 'none',
            regular DDPG experience replay is used
        replay_k (int): the ratio between HER replays and regular replays (e.g. k = 4 -> 4 times
            as many HER replays as regular replays are used)
        reward_fun (function): function to re-compute the reward with substituted goals
    """
    if (replay_strategy == 'future') or (replay_strategy == 'final'):
        future_p = 1 - (1. / (1 + replay_k))
    else:  # 'replay_strategy' == 'none'
        future_p = 0

    def goal_distance(goal_a, goal_b):
        assert goal_a.shape == goal_b.shape
        return np.linalg.norm(goal_a - goal_b, axis=-1)

    def _sample_her_transitions(episode_batch, batch_size_in_transitions):
        """episode_batch is {key: array(buffer_size x T x dim_key)}
        """
        T = episode_batch['u'].shape[1]
        rollout_batch_size = episode_batch['u'].shape[0]
        batch_size = batch_size_in_transitions
        # Select which episodes and time steps to use.
        episode_idxs = np.random.randint(0, rollout_batch_size, batch_size)
        t_samples = np.random.randint(T, size=batch_size)

        motivate_buffer = episode_batch['mot'][episode_idxs].flatten()

        # transitions = {key: episode_batch[key][episode_idxs, t_samples].copy()
        #                for key in episode_batch.keys()}

        transitions = {}
        for key in episode_batch.keys():
            if not key == 's' and not key == 'div'and not key == 'mot':
                transitions[key] = episode_batch[key][episode_idxs, t_samples].copy()
                
        # Select future time indexes proportional with probability future_p. These
        # will be used for HER replay by substituting in future goals.
        her_indexes = np.where(np.random.uniform(size=batch_size) < future_p)
        future_offset = np.random.uniform(size=batch_size) * (T - t_samples)
        future_offset = future_offset.astype(int)
        future_t = (t_samples + 1 + future_offset)[her_indexes]
        if replay_strategy == 'final':
            future_t[:] = T
        # Replace goal with achieved goal but only for the previously-selected
        # HER transitions (as defined by her_indexes). For the other transitions,
        # keep the original goal.
        future_ag = episode_batch['ag'][episode_idxs[her_indexes], future_t]
        transitions['g'][her_indexes] = future_ag
        # Reconstruct info dictionary for reward computation.
        info = {}
        for key, value in transitions.items():
            if key.startswith('info_'):
                info[key.replace('info_', '')] = value
        # Re-compute reward since we may have substituted the goal.
        reward_params = {k: transitions[k] for k in ['ag_2', 'g']}
        reward_params['info'] = info

        transitions['r'] = reward_fun(**reward_params)


        transitions = {k: transitions[k].reshape(batch_size, *transitions[k].shape[1:])
                       for k in transitions.keys()}
        assert(transitions['u'].shape[0] == batch_size_in_transitions)
        return transitions
    return _sample_her_transitions


def make_sample_her_transitions_motivation(replay_strategy, replay_k, reward_fun, ratio,ratio_o):
    if (replay_strategy == 'future') or (replay_strategy == 'final'):
        future_p = 1 - (1. / (1 + replay_k))
    else:
        future_p = 0

    def goal_distance(goal_a, goal_b):
        assert goal_a.shape == goal_b.shape
        return np.linalg.norm(goal_a - goal_b, axis=-1)
    
    def _sample_her_transitions(episode_batch, batch_size_in_transitions, update_stats=False):
        T = episode_batch['u'].shape[1]
        rollout_batch_size = episode_batch['u'].shape[0]
        batch_size = batch_size_in_transitions
        episode_idxs = np.random.randint(0, rollout_batch_size, batch_size)
        t_samples = np.random.randint(T, size=batch_size)
        

        
        if not update_stats:
            mot_trajectory = episode_batch['mot']
            # calculate the priority
            p_trajectory = mot_trajectory.copy()
            p_trajectory = p_trajectory / p_trajectory.sum()
            
            if(np.isnan(p_trajectory.flatten()).any()):
                p_f = np.random.randint(0,2,2)
                p_f[-1] = 1-p_f[0]
            else:
                p_f = p_trajectory.flatten()
            
            episode_idxs_mot = np.random.choice(rollout_batch_size, size=batch_size, replace=True, p=p_f)
            episode_idxs = episode_idxs_mot

        transitions = {}


        for key in episode_batch.keys():
            if not key == 's' and not key == 'mot':
                transitions[key] = episode_batch[key][episode_idxs, t_samples].copy()

        her_indexes = np.where(np.random.uniform(size=batch_size) < future_p)
        future_offset = np.random.uniform(size=batch_size) * (T - t_samples)
        future_offset = future_offset.astype(int)
        future_t = (t_samples + 1 + future_offset)[her_indexes]
        if replay_strategy == 'final':
            future_t[:] = T
        # Replace goal with achieved goal but only for the previously-selected
        # HER transitions (as defined by her_indexes). For the other transitions,
        # keep the original goal.
        future_ag = episode_batch['ag'][episode_idxs[her_indexes], future_t]
        transitions['g'][her_indexes] = future_ag
        # Reconstruct info dictionary for reward computation.
        info = {}
        for key, value in transitions.items():
            if key.startswith('info_'):
                info[key.replace('info_', '')] = value
        # Re-compute reward since we may have substituted the goal.
        reward_params = {k: transitions[k] for k in ['ag_2', 'g']}
        reward_params['info'] = info
        
        
        #------------------------------------PR reward-------------------------------------------
        progress_list = []
        n = transitions['ag_last'].shape[0]
        for f in range(n):
            last_dist = goal_distance(transitions['ag_last'][f,:], transitions['g'][f,:])
            now_dist = goal_distance(transitions['ag'][f,:], transitions['g'][f,:])
 
            progress =  last_dist - now_dist
            
            init_distance = goal_distance(transitions['init_position'][f,:],transitions['g'][f,:])
            
            if(progress > init_distance/50): 
            	progress_list.append(1)
            else:
            	progress_list.append(-1)

        progress_list=np.array(progress_list)
        transitions['r'] = ratio_o*reward_fun(**reward_params)+ratio*np.array(progress_list)
        #-------------------------------------------------------------------------------
        
        # transitions['r'] = reward_fun(**reward_params)

        transitions = {k: transitions[k].reshape(batch_size, *transitions[k].shape[1:])
                       for k in transitions.keys()}
        assert(transitions['u'].shape[0] == batch_size_in_transitions)
        return transitions
    return _sample_her_transitions
