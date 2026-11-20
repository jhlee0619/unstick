import numpy as np
from env.pointmaze.maze_model import MazeEnv
from utils import aggregate_dct

class PointMazeWrapper(MazeEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.action_dim = self.action_space.shape[0]
    
    def update_env(self, env_info):
        pass 
    
    def eval_state(self, goal_state, cur_state):
        success = np.linalg.norm(goal_state[:2] - cur_state[:2]) < 0.5
        state_dist = np.linalg.norm(goal_state - cur_state)
        return {
            'success': success,
            'state_dist': state_dist,
        }

    def prepare(self, seed, init_state):
        """Reset to init_state; returns obs (H W C) and state."""
        self.prepare_for_render()
        self.seed(seed)
        self.set_init_state(init_state)
        obs, state = self.reset()
        return obs, state

    def step_multiple(self, actions):
        """Steps through actions; obses and infos are dicts of (T, ...) arrays."""
        obses = []
        rewards = []
        dones = []
        infos = []
        for action in actions:
            o, r, d, info = self.step(action)
            obses.append(o)
            rewards.append(r)
            dones.append(d)
            infos.append(info)
        obses = aggregate_dct(obses)
        rewards = np.stack(rewards)
        dones = np.stack(dones)
        infos = aggregate_dct(infos)
        return obses, rewards, dones, infos

    def rollout(self, seed, init_state, actions):
        """actions (T, action_dim) from init_state -> obses dict of (T+1, ...) arrays, states (T+1, D)."""
        obs, state = self.prepare(seed, init_state)
        obses, rewards, dones, infos = self.step_multiple(actions)
        for k in obses.keys():
            obses[k] = np.vstack([np.expand_dims(obs[k], 0), obses[k]])
        states = np.vstack([np.expand_dims(state, 0), infos["state"]])
        states = np.stack(states)
        return obses, states
