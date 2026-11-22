import argparse, csv, os, time
from pathlib import Path
import numpy as np
from config import RescueConfig
from environment.rescue_env import RescueEnv
from agents.dqn_agent import DoubleDQNAgent

ROOT = Path(__file__).resolve().parents[1]

def train(episodes=500, seed=7, log_path=None, model_path=None, resume=None):
    config = RescueConfig(seed=seed)
    env = RescueEnv(config)
    agent = DoubleDQNAgent(env.observation_size)
    if resume and Path(resume).exists():
        agent.load(resume)
    log_path = Path(log_path or ROOT/"data/training_logs/training.csv")
    model_path = Path(model_path or ROOT/"models/best_model/resq_double_dqn.pt")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not log_path.exists()
    best = -float("inf")
    with log_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["episode","reward","rescued","critical_rescued","steps","survived","epsilon","mean_loss"])
        if write_header: writer.writeheader()
        for ep in range(1, episodes+1):
            state, _ = env.reset(seed=seed+ep)
            total, losses = 0.0, []
            done = False
            while not done:
                action = agent.act(state, env, explore=True)
                next_state, reward, done, info = env.step(action)
                agent.remember(state, action, reward, next_state, done)
                loss = agent.learn()
                if loss is not None: losses.append(loss)
                state = next_state
                total += reward
            row = {"episode":ep, "reward":round(total,3), "rescued":info["rescued"],
                   "critical_rescued":info["critical_rescued"], "steps":info["step"],
                   "survived":int(info["health"]>0), "epsilon":round(agent.epsilon,4),
                   "mean_loss":round(float(np.mean(losses)),5) if losses else 0}
            writer.writerow(row); f.flush()
            if total > best:
                best = total
                agent.save(model_path)
            if ep % 10 == 0:
                print(f"Episode {ep}/{episodes} | reward={total:.1f} | rescued={info['rescued']} | epsilon={agent.epsilon:.3f}", flush=True)
    agent.save(ROOT/"models/checkpoints/latest.pt")
    print(f"Training complete. Best checkpoint: {model_path}", flush=True)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=500)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--log", default=None)
    p.add_argument("--model", default=None)
    p.add_argument("--resume", default=None)
    a = p.parse_args()
    train(a.episodes, a.seed, a.log, a.model, a.resume)
