"""A
Some miscellaneous utility functions

Functions to edit:
    1. sample_trajectory
"""

from collections import OrderedDict
import cv2
import numpy as np
import time

from cs285.infrastructure import pytorch_util as ptu


def sample_trajectory(env, policy, max_path_length, render=False):
    """Sample a rollout in the environment from a policy."""
    
    # initialize env for the beginning of a new rollout
    ob =  env.reset() # initial observation after resetting the env
    # ob = [np array[..., ..., ...], {}]
    ob = ob[0] # Get the numpy array from the list
    # init vars
    obs, acs, rewards, next_obs, terminals, image_obs = [], [], [], [], [], []
    steps = 0
    while True:
        # render image of the simulated env
        if render:
            if hasattr(env, 'sim'):
                img = env.sim.render(camera_name='track', height=500, width=500)[::-1]
            else:
                img = env.render(mode='single_rgb_array')
            image_obs.append(cv2.resize(img, dsize=(250, 250), interpolation=cv2.INTER_CUBIC))
    
        # TODO use the most recent ob to decide what to do
        # ac = TODO # HINT: this is a numpy array
        ac = policy.forward(ptu.from_numpy(ob))

        # if ac is 2D tensor, then we pick the first element
        if ac.dim() > 1:
            ac = ac[0]
        # if len(ac) > 1:
        # ac = ac[0]


        ac = ptu.to_numpy(ac)



        # TODO: take that action and get reward and next ob
        # in this version of gym. There is one extra vairable 
        '''
        (Pdb) env.step(ac)
        (array([ 6.91594367e-01,  9.95740407e-01,  3.81938381e-02,  3.22281657e-03,
            8.38563363e-02, -5.81835893e-01,  9.20813581e-01, -1.34227388e-01,
        -7.70821058e-01,  5.36470687e-01, -8.28939911e-01,  9.34773928e-02,
            1.03078561e+00, -3.40384894e-01, -8.00001930e-02, -5.37578046e-01,
        -4.44471996e-01, -5.94584779e-01, -4.95064264e-01, -3.45198646e+00,
            9.86393419e+00, -2.39126248e+00, -6.18738759e+00,  8.20065941e+00,
        -7.65501877e+00,  1.60309505e+00,  1.25425308e+01]), -0.11469752469479122, 
        False, 
        False, 
        {'reward_forward': -0.49894584098278555, 'reward_ctrl': -0.6157516837120056, 
        'reward_survive': 1.0, 'x_position': 0.05506420252152459, 'y_position': 0.011032858043066062, 
        'distance_from_origin': 0.056158617824247796, 'x_velocity': -0.49894584098278555, 
        'y_velocity': 0.11579151587997107, 'forward_reward': -0.49894584098278555})
        '''
        next_ob, rew, done, _, _= env.step(ac)
        

        # TODO rollout can end due to done, or due to max_path_length
        steps += 1
        # rollout_done = TODO # HINT: this is either 0 or 1
        rollout_done = 1 if done else 0
        
        # record result of taking that action
        obs.append(ob)
        acs.append(ac)
        rewards.append(rew)
        next_obs.append(next_ob)
        terminals.append(rollout_done)

        ob = next_ob # jump to next timestep

        # end the rollout if the rollout ended
        if rollout_done:
            break

    return {"observation" : np.array(obs, dtype=np.float32),
            "image_obs" : np.array(image_obs, dtype=np.uint8),
            "reward" : np.array(rewards, dtype=np.float32),
            "action" : np.array(acs, dtype=np.float32),
            "next_observation": np.array(next_obs, dtype=np.float32),
            "terminal": np.array(terminals, dtype=np.float32)}


def sample_trajectories(env, policy, min_timesteps_per_batch, max_path_length, render=False):
    """Collect rollouts until we have collected min_timesteps_per_batch steps."""

    timesteps_this_batch = 0
    paths = []
    while timesteps_this_batch < min_timesteps_per_batch:

        #collect rollout
        path = sample_trajectory(env, policy, max_path_length, render)
        paths.append(path)

        #count steps
        timesteps_this_batch += get_pathlength(path)

    return paths, timesteps_this_batch


def sample_n_trajectories(env, policy, ntraj, max_path_length, render=False):
    """Collect ntraj rollouts."""

    paths = []
    for i in range(ntraj):
        # collect rollout
        path = sample_trajectory(env, policy, max_path_length, render)
        paths.append(path)
    return paths


########################################
########################################


def convert_listofrollouts(paths, concat_rew=True):
    """
        Take a list of rollout dictionaries
        and return separate arrays,
        where each array is a concatenation of that array from across the rollouts
    """
    observations = np.concatenate([path["observation"] for path in paths])
    actions = np.concatenate([path["action"] for path in paths])
    if concat_rew:
        rewards = np.concatenate([path["reward"] for path in paths])
    else:
        rewards = [path["reward"] for path in paths]
    next_observations = np.concatenate([path["next_observation"] for path in paths])
    terminals = np.concatenate([path["terminal"] for path in paths])
    return observations, actions, rewards, next_observations, terminals


########################################
########################################
            

def compute_metrics(paths, eval_paths):
    """Compute metrics for logging."""

    # returns, for logging
    train_returns = [path["reward"].sum() for path in paths]
    eval_returns = [eval_path["reward"].sum() for eval_path in eval_paths]

    # episode lengths, for logging
    train_ep_lens = [len(path["reward"]) for path in paths]
    eval_ep_lens = [len(eval_path["reward"]) for eval_path in eval_paths]

    # decide what to log
    logs = OrderedDict()
    logs["Eval_AverageReturn"] = np.mean(eval_returns)
    logs["Eval_StdReturn"] = np.std(eval_returns)
    logs["Eval_MaxReturn"] = np.max(eval_returns)
    logs["Eval_MinReturn"] = np.min(eval_returns)
    logs["Eval_AverageEpLen"] = np.mean(eval_ep_lens)

    logs["Train_AverageReturn"] = np.mean(train_returns)
    logs["Train_StdReturn"] = np.std(train_returns)
    logs["Train_MaxReturn"] = np.max(train_returns)
    logs["Train_MinReturn"] = np.min(train_returns)
    logs["Train_AverageEpLen"] = np.mean(train_ep_lens)

    return logs


############################################
############################################


def get_pathlength(path):
    return len(path["reward"])
