"""Subprocess vector env (trimmed from tianshou's SubprocVectorEnv) with batched prepare/rollout/eval calls."""
import cloudpickle
import gym
import numpy as np

from multiprocessing import Pipe, connection
from multiprocessing.context import Process
from typing import Any, Callable, List

from utils import aggregate_dct


class CloudpickleWrapper(object):
    """A cloudpickle wrapper used in SubprocVectorEnv."""

    def __init__(self, data: Any) -> None:
        self.data = data

    def __getstate__(self) -> str:
        return cloudpickle.dumps(self.data)

    def __setstate__(self, data: str) -> None:
        self.data = cloudpickle.loads(data)


def _worker(
    parent: connection.Connection,
    p: connection.Connection,
    env_fn_wrapper: CloudpickleWrapper,
) -> None:
    parent.close()
    env = env_fn_wrapper.data()
    try:
        while True:
            try:
                cmd, data = p.recv()
            except EOFError:  # the pipe has been closed
                p.close()
                break
            if cmd == "rollout":
                p.send(env.rollout(data[0], data[1], data[2]))
            elif cmd == "prepare":
                p.send(env.prepare(data[0], data[1]))
            elif cmd == "eval_state":
                p.send(env.eval_state(data[0], data[1]))
            elif cmd == "update_env":
                p.send(env.update_env(data))
            else:
                p.close()
                raise NotImplementedError
    except KeyboardInterrupt:
        p.close()


class SubprocEnvWorker:
    def __init__(self, env_fn: Callable[[], gym.Env]) -> None:
        self.parent_remote, self.child_remote = Pipe()
        args = (
            self.parent_remote,
            self.child_remote,
            CloudpickleWrapper(env_fn),
        )
        self.process = Process(target=_worker, args=args, daemon=True)
        self.process.start()
        self.child_remote.close()

    def call(self, cmd, data):
        self.parent_remote.send([cmd, data])
        return self.parent_remote.recv()


class SubprocVectorEnv:
    """One subprocess per environment; methods take per-environment lists (seeds, states, actions)."""

    def __init__(self, env_fns: List[Callable[[], gym.Env]]) -> None:
        self.workers = [SubprocEnvWorker(fn) for fn in env_fns]
        self.env_num = len(env_fns)

    def rollout(self, seeds, init_states, action: np.ndarray):
        assert len(action) == self.env_num
        for i, w in enumerate(self.workers):
            w.parent_remote.send(["rollout", (seeds[i], init_states[i], action[i])])
        result = [w.parent_remote.recv() for w in self.workers]
        obses, states = tuple(zip(*result))
        obses = aggregate_dct(obses)
        states = np.stack(states)
        return obses, states

    def prepare(self, seeds, init_states):
        obs_list = []
        state = []
        for i in range(self.env_num):
            o, s = self.workers[i].call("prepare", (seeds[i], init_states[i]))
            obs_list.append(o)
            state.append(s)
        obs = aggregate_dct(obs_list)
        state = np.stack(state)
        return obs, state

    def eval_state(self, goal_state, cur_state):
        eval_result = []
        for i in range(self.env_num):
            s = self.workers[i].call("eval_state", (goal_state[i], cur_state[i]))
            eval_result.append(s)
        eval_result = aggregate_dct(eval_result)
        return eval_result

    def update_env(self, env_info):
        for i in range(self.env_num):
            self.workers[i].call("update_env", env_info[i])
