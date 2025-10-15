import math, json, random, tempfile, os
from pathlib import Path
from typing import List, Tuple
import numpy as np
import yaml

from multi_drone import MultiDroneUnc
from myplanner import UCTPlanner

'''experiment settinngs'''
NS = [1, 2, 3, 4]          # number of drones (action space grows as 8^N)
TRIALS_PER_N = 20          # repeat count per N
PLANNING_TIME_PER_STEP = 1.0

# XY-only (8-neighbour)
GRID_SIZE = [10, 10, 5]
FIXED_Z = 2
OBSTACLES = [[5, 3, 2], [5, 7, 2]]
CONFIG_BASE = dict(
    grid_size=GRID_SIZE,
    obstacle_cells=OBSTACLES,
    change_altitude=False,    
    step_cost=-1.0,
    collision_penalty=-50.0,
    goal_reward=100.0,
    discount_factor=0.98,
    max_num_steps=100,
    alpha=0.5                 
)

PLANNER_PARAMS = dict(
    c_uct=1.5,
    c_pw=1.5,
    alpha_pw=0.5,
    rollout_horizon=60,
    epsilon_rollout=0.2,
    k_rollout=8,
    expand_to_rollout_depth=2
)

'''helpers'''
def ci95(mean: float, xs: List[float]) -> Tuple[float, float]:
    n = len(xs)
    if n <= 1: return (mean, mean)
    sd = np.std(xs, ddof=1)
    half = 1.96 * sd / math.sqrt(n)
    return (mean - half, mean + half)

def per_scenario_positions(n: int) -> Tuple[List[List[int]], List[List[int]]]:
    #Spread starts/goals along y at fixed z to limit immediate collisions.
    gx, gy, _ = GRID_SIZE
    step_y = max(1, gy // (n + 1))
    ys = [(i + 1) * step_y for i in range(n)]
    starts = [[0, min(gy - 1, y), FIXED_Z] for y in ys]
    goals  = [[gx - 1, min(gy - 1, y), FIXED_Z] for y in ys]
    return starts, goals

def run_episode(env: MultiDroneUnc, planner: UCTPlanner, planning_time_per_step: float):
    state = env.reset()
    cfg = env.get_config()
    gamma = cfg.discount_factor
    total_disc, steps = 0.0, 0
    while True:
        a = planner.plan(state, planning_time_per_step)
        ns, r, done, info = env.simulate(state, a)
        total_disc += (gamma ** steps) * r
        steps += 1
        state = ns
        if done or steps >= cfg.max_num_steps:
            return total_disc, bool(info.get("success", False)), steps

def main():
    # reproducibility
    random.seed(42); np.random.seed(42)

    results = []
    for n in NS:
        starts, goals = per_scenario_positions(n)
        cfg = dict(CONFIG_BASE)
        cfg["start_positions"] = starts
        cfg["goal_positions"] = goals

        disc, succ, steps = [], [], []

        print(f"\n=== XY-only (8-neighbour) | N={n} drones | t/step={PLANNING_TIME_PER_STEP}s ===")

        for t in range(TRIALS_PER_N):
            seed = 10_000 + 97 * n + t
            random.seed(seed); np.random.seed(seed)
            with tempfile.TemporaryDirectory() as td:
                yml = os.path.join(td, f"env_n{n}.yaml")
                with open(yml, "w") as f:
                    yaml.safe_dump(cfg, f)
                env = MultiDroneUnc(yml)
                planner = UCTPlanner(env, **PLANNER_PARAMS)
                td_r, ok, nsteps = run_episode(env, planner, PLANNING_TIME_PER_STEP)
                disc.append(td_r); succ.append(1 if ok else 0); steps.append(nsteps)

        mean_r = float(np.mean(disc)); lo_r, hi_r = ci95(mean_r, disc)
        mean_s = float(np.mean(steps)); lo_s, hi_s = ci95(mean_s, steps)
        p = float(np.mean(succ))
        se = math.sqrt(max(1e-9, p*(1-p)/len(succ)))
        lo_p, hi_p = p - 1.96*se, p + 1.96*se

        print(f"Total discounted reward: mean={mean_r:.2f}, 95% CI=({lo_r:.2f}, {hi_r:.2f})")
        print(f"Steps to termination:   mean={mean_s:.1f}, 95% CI=({lo_s:.1f}, {hi_s:.1f})")
        print(f"Success rate:           mean={p*100:.1f}%, 95% CI=({max(0,lo_p)*100:.1f}%, {min(1,hi_p)*100:.1f}%)")

        results.append({
            "num_drones": n,
            "disc_reward_mean": mean_r, "disc_reward_ci": [lo_r, hi_r],
            "steps_mean": mean_s, "steps_ci": [lo_s, hi_s],
            "success_mean": p, "success_ci": [max(0,lo_p), min(1,hi_p)]
        })

    Path("q4_xyonly_results.json").write_text(json.dumps(results, indent=2))
    print("\nSaved q4_xyonly_results.json (drop straight into your report).")

if __name__ == "__main__":
    main()
