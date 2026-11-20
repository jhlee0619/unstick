import numpy as np
from env.pusht.pusht_env import PushTEnv
from utils import aggregate_dct

class PushTWrapper(PushTEnv):
    def __init__(
            self,
            with_velocity=True,
            with_target=True,
            color="LightSlateGray",
            agent_color="RoyalBlue",
            goal_color=None,
        ):
        super().__init__(
            with_velocity=with_velocity,
            with_target=with_target,
            color=color,
            agent_color=agent_color,
            goal_color=goal_color,
        )
        self.action_dim = self.action_space.shape[0]
    
    def update_env(self, env_info):
        if 'shape' in env_info:
            self.shape = env_info['shape']
        if 'color' in env_info:
            self.color = env_info['color']
        if 'agent_color' in env_info:
            self.agent_color = env_info['agent_color']
        if 'goal_color' in env_info:
            self.goal_color_override = env_info['goal_color']
    
    def eval_state(self, goal_state, cur_state):
        """Success: agent and block positions within 20 (L2) and block angle within pi/9 of the goal."""
        pos_diff = np.linalg.norm(goal_state[:4] - cur_state[:4])
        angle_diff = np.abs((goal_state[4] - cur_state[4] + np.pi) % (2 * np.pi) - np.pi)
        success = pos_diff < 20 and angle_diff < np.pi / 9
        state_dist = np.linalg.norm(goal_state - cur_state)
        return {
            'success': success,
            'state_dist': state_dist,
        }

    def prepare(self, seed, init_state):
        """Reset to init_state; returns obs (H W C) and state."""
        self.seed(seed)
        self.reset_to_state = init_state
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
