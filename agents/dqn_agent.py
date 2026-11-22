from pathlib import Path
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from rl.network import QNetwork
from rl.replay_buffer import ReplayBuffer

class DoubleDQNAgent:
    def __init__(self, observation_size, action_size=7, learning_rate=5e-4,
                 gamma=0.99, epsilon=1.0, epsilon_min=0.05, epsilon_decay=0.9995,
                 buffer_size=20000, batch_size=64, target_update=250, device=None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.online = QNetwork(observation_size, action_size).to(self.device)
        self.target = QNetwork(observation_size, action_size).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.optimizer = Adam(self.online.parameters(), lr=learning_rate)
        self.memory = ReplayBuffer(buffer_size)
        self.gamma, self.epsilon = gamma, epsilon
        self.epsilon_min, self.epsilon_decay = epsilon_min, epsilon_decay
        self.batch_size, self.target_update = batch_size, target_update
        self.action_size, self.updates = action_size, 0

    def act(self, observation, env=None, explore=True):
        if explore and random.random() < self.epsilon:
            return random.randrange(self.action_size)
        with torch.no_grad():
            state = torch.tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
            return int(self.online(state).argmax(dim=1).item())

    def remember(self, *transition):
        self.memory.add(*transition)

    def learn(self):
        if len(self.memory) < self.batch_size:
            return None
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        states = torch.tensor(states, device=self.device)
        actions = torch.tensor(actions, device=self.device).unsqueeze(1)
        rewards = torch.tensor(rewards, device=self.device)
        next_states = torch.tensor(next_states, device=self.device)
        dones = torch.tensor(dones, device=self.device)
        q = self.online(states).gather(1, actions).squeeze(1)
        with torch.no_grad():
            next_actions = self.online(next_states).argmax(dim=1, keepdim=True)
            next_q = self.target(next_states).gather(1, next_actions).squeeze(1)
            target = rewards + self.gamma * next_q * (1-dones)
        loss = F.smooth_l1_loss(q, target)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online.parameters(), 10.0)
        self.optimizer.step()
        self.updates += 1
        if self.updates % self.target_update == 0:
            self.target.load_state_dict(self.online.state_dict())
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        return float(loss.item())

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"online": self.online.state_dict(), "target": self.target.state_dict(),
                    "epsilon": self.epsilon, "updates": self.updates}, path)

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.online.load_state_dict(checkpoint["online"])
        self.target.load_state_dict(checkpoint.get("target", checkpoint["online"]))
        self.epsilon = checkpoint.get("epsilon", self.epsilon)
        self.updates = checkpoint.get("updates", 0)
