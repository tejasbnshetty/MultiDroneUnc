import time
import math
import random
import numpy as np
from collections import defaultdict
from typing import Dict, Tuple
from multi_drone import MultiDroneUnc

class UCTPlanner:
    """
    Online MCTS/UCT planner with progressive widening for MultiDroneUnc.
    - Uses the environment as a generative model via env.simulate(state, action)
    - Returns a single integer action in [0, env.num_actions - 1]
    - State nodes only; stochasticity is averaged in the Q estimates
    """

    def __init__(
        self,
        env: MultiDroneUnc,
        c_uct: float = 1.5,            # UCT exploration constant
        c_pw: float = 1.5,             # progressive widening coefficient
        alpha_pw: float = 0.5,         # progressive widening exponent (0<alpha<1)
        rollout_horizon: int = 60,     # max rollout steps
        epsilon_rollout: float = 0.2,  # ε for ε-greedy rollout
        k_rollout: int = 8,            # candidates evaluated per rollout step
        expand_to_rollout_depth: int = 2  # tree depth before switching to rollout
    ):
        self.env = env
        self.gamma = env.get_config().discount_factor
        self._num_actions = env.num_actions 

        self.c_uct = c_uct
        self.c_pw = c_pw
        self.alpha_pw = alpha_pw
        self.rollout_horizon = rollout_horizon
        self.epsilon_rollout = epsilon_rollout
        self.k_rollout = k_rollout
        self.expand_to_rollout_depth = expand_to_rollout_depth
        # state-key -> node
        self.nodes: Dict[Tuple, _Node] = {}

    '''public API expected by run_planner.py'''
    def plan(self, current_state: np.ndarray, planning_time_per_step: float) -> int:
        root_key = self._state_key(current_state)
        root = self._get_node(root_key)
        deadline = time.time() + max(0.0, float(planning_time_per_step))

        # run as many simulations as we can within the time budget
        while time.time() < deadline:
            self._simulation(current_state)

        # pick best root action by Q, tie break by N
        if not root.Q_a:
            return random.randrange(self._num_actions)
        return max(root.Q_a.keys(), key=lambda a: (root.Q_a[a], root.N_a[a]))

    '''one simulation (selection/expansion, rollout, backup)'''
    def _simulation(self, root_state: np.ndarray) -> None:
        path = []  # list of (state_key, action, reward)
        s = root_state
        depth = 0

        # Selection + Expansion with progressive widening
        while depth < self.expand_to_rollout_depth:
            skey = self._state_key(s)
            node = self._get_node(skey)

            if self._allow_widen(node):
                a = self._sample_new_action(node)
            else:
                if not node.Q_a:
                    a = self._sample_new_action(node)
                else:
                    a = self._uct_best_action(node)

            s_next, r, done, _ = self.env.simulate(s, a)
            path.append((skey, a, r))
            s = s_next
            depth += 1
            if done:
                break

        # Rollout (ε-greedy over k random actions, by one-step reward)
        while (not self._is_terminal(s)) and (depth < self.rollout_horizon):
            a = self._rollout_policy(s)
            s_next, r, done, _ = self.env.simulate(s, a)
            path.append((self._state_key(s), a, r))
            s = s_next
            depth += 1
            if done:
                break

        # Backup discounted return along the path
        G = 0.0
        for (_, _, r) in reversed(path):
            G = r + self.gamma * G

        G_running = G
        for (skey, a, r) in path:
            node = self._get_node(skey)
            node.N += 1
            node.N_a[a] += 1
            q_old = node.Q_a[a]
            n_sa = node.N_a[a]
            node.Q_a[a] = q_old + (G_running - q_old) / n_sa
            # shift one step forward (undo one discount)
            G_running = (G_running - r) / self.gamma if self.gamma > 0 else 0.0

    '''helpers: UCT, widening, rollout'''
    def _uct_best_action(self, node: "_Node") -> int:
        lnN = math.log(max(1, node.N))
        best_a, best_score = None, -float("inf")
        for a in node.Q_a.keys():
            q = node.Q_a[a]
            n_sa = node.N_a[a]
            u = self.c_uct * math.sqrt(lnN / (n_sa + 1e-9))
            score = q + u
            if score > best_score:
                best_score, best_a = score, a
        return best_a

    def _allow_widen(self, node: "_Node") -> bool:
        limit = self.c_pw * (node.N ** self.alpha_pw)
        # ensure we never exceed the action space
        return len(node.Q_a) < min(self._num_actions, int(limit) + 1)

    def _sample_new_action(self, node: "_Node") -> int:
        # uniform sample from untried actions
        while True:
            a = random.randrange(self._num_actions)
            if a not in node.N_a:
                node.N_a[a] = 0
                node.Q_a[a] = 0.0
                return a

    def _rollout_policy(self, s: np.ndarray) -> int:
        if random.random() < self.epsilon_rollout:
            return random.randrange(self._num_actions)
        # ε-greedy: by immediate reward, pick best
        best_a, best_r = None, -float("inf")
        for _ in range(self.k_rollout):
            a = random.randrange(self._num_actions)
            _, r, _, _ = self.env.simulate(s, a)
            if r > best_r:
                best_r, best_a = r, a
        return best_a if best_a is not None else random.randrange(self._num_actions)

    '''state bookkeeping'''
    def _state_key(self, s: np.ndarray) -> Tuple:
        arr = np.asarray(s)
        return tuple(tuple(int(v) for v in row) for row in arr)

    def _get_node(self, skey: Tuple) -> "_Node":
        node = self.nodes.get(skey)
        if node is None:
            node = _Node()
            self.nodes[skey] = node
        return node

    def _is_terminal(self, s: np.ndarray) -> bool:
        flags = (np.asarray(s)[:, 3] > 0.5)
        return bool(np.all(flags))


class _Node:
    __slots__ = ("N", "N_a", "Q_a")
    def __init__(self):
        self.N: int = 0
        self.N_a: Dict[int, int] = defaultdict(int)
        self.Q_a: Dict[int, float] = defaultdict(float)
