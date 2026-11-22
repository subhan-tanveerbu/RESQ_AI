import argparse, csv
from pathlib import Path
import numpy as np
from config import RescueConfig
from environment.rescue_env import RescueEnv
from agents.random_agent import RandomAgent
from agents.rule_based_agent import RuleBasedAgent
from agents.dqn_agent import DoubleDQNAgent

ROOT = Path(__file__).resolve().parents[1]

def run_evaluation(episodes=30, model_path=None, output=None, seed=1000):
    env = RescueEnv(RescueConfig())
    rl = DoubleDQNAgent(env.observation_size)
    model_path = Path(model_path or ROOT/"models/best_model/resq_double_dqn.pt")
    if model_path.exists(): rl.load(model_path)
    rl.epsilon = 0.0
    agents = {"Random": RandomAgent(), "Rule-Based": RuleBasedAgent(), "Double DQN": rl}
    rows = []
    for name, agent in agents.items():
        for ep in range(episodes):
            state, _ = env.reset(seed=seed+ep)
            done = False
            while not done:
                action = agent.act(state, env, explore=False) if name == "Double DQN" else agent.act(state, env)
                state, reward, done, info = env.step(action)
            rows.append({"agent":name,"episode":ep+1,"seed":seed+ep,"reward":info["reward"],
                         "rescued":info["rescued"],"critical_rescued":info["critical_rescued"],
                         "success":int(info["rescued"] == len(env.victims)),
                         "survived":int(info["health"] > 0),"steps":info["step"],
                         "energy_used":info["energy_used"],"hazard_encounters":info["hazard_encounters"]})
    output = Path(output or ROOT/"data/evaluation_results/comparison.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader(); writer.writerows(rows)
    print(f"Saved {len(rows)} evaluation rows to {output}")
    return rows

if __name__ == "__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--episodes",type=int,default=30)
    p.add_argument("--model",default=None)
    a=p.parse_args()
    run_evaluation(a.episodes,a.model)
