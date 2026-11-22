from dataclasses import dataclass
from collections import deque
import numpy as np
from config import RescueConfig

# Actions: up, down, left, right, wait, interact, recharge/recover
UP, DOWN, LEFT, RIGHT, WAIT, INTERACT, RECHARGE = range(7)
ACTION_NAMES = ["Up", "Down", "Left", "Right", "Wait", "Interact / evacuate", "Recharge / recover"]
CRITICAL, INJURED, STABLE = 0, 1, 2
VICTIM_NAMES = ["Critical", "Injured", "Stable"]
VICTIM_REWARDS = [100.0, 60.0, 30.0]

@dataclass
class Victim:
    x: int
    y: int
    category: int
    survival: int
    status: str = "waiting"  # waiting, carried, rescued, lost

class RescueEnv:
    """Partially observable grid rescue environment with a Gym-like reset/step API."""
    def __init__(self, config=None):
        self.config = (config or RescueConfig()).validate()
        self.rng = np.random.default_rng(self.config.seed)
        self.reset()

    def reset(self, seed=None):
        if seed is not None:
            self.config.seed = int(seed)
        self.rng = np.random.default_rng(self.config.seed)
        n = self.config.map_size
        self.terrain = np.zeros((n, n), dtype=np.int8)  # 0 road/ground, 1 rubble, 2 blocked
        self.fire = np.zeros((n, n), dtype=np.int8)
        self.gas = np.zeros((n, n), dtype=np.int8)
        self.known = np.zeros((n, n), dtype=np.int8)
        self.agent = [1, 1]
        self.evac = (n - 2, n - 2)
        self.chargers = [(1, n - 2)]
        self.medical = [(n - 2, 1)]
        self.carried = []
        self.steps = 0
        self.energy = self.config.initial_energy
        self.health = self.config.initial_health
        self.rescued = 0
        self.critical_rescued = 0
        self.hazard_encounters = 0
        self.energy_used = 0
        self.total_reward = 0.0
        self.done = False
        self.reason = ""
        self._place_objects()
        self._reveal()
        return self.observe(), self.info()

    def _valid_empty(self, x, y, reserved):
        n = self.config.map_size
        return 0 <= x < n and 0 <= y < n and (x, y) not in reserved and self.terrain[y, x] != 2

    def _place_objects(self):
        n = self.config.map_size
        reserved = {tuple(self.agent), self.evac, *self.chargers, *self.medical}
        # Place rubble away from the essential start/exit/stations.
        candidates = [(x, y) for y in range(n) for x in range(n) if (x, y) not in reserved]
        self.rng.shuffle(candidates)
        for x, y in candidates[:self.config.rubble_count]:
            if self.rng.random() < 0.65:
                self.terrain[y, x] = 1
        # Keep a simple traversable corridor along the top and right edges.
        self.terrain[1, :] = 0
        self.terrain[:, n-2] = 0
        self.terrain[n-2, :] = 0
        self.victims = []
        available = [(x, y) for y in range(n) for x in range(n)
                     if self.terrain[y, x] != 2 and (x, y) not in reserved]
        self.rng.shuffle(available)
        cats = [CRITICAL, INJURED, STABLE]
        for i in range(self.config.victim_count):
            x, y = available[i % len(available)]
            cat = cats[i % 3]
            survival = {CRITICAL: 95, INJURED: 165, STABLE: 240}[cat]
            if self.config.difficulty == "easy":
                survival = int(survival * 1.25)
            elif self.config.difficulty == "hard":
                survival = int(survival * 0.75)
            self.victims.append(Victim(x, y, cat, survival))
        # Hazards are placed away from the starting tile and stations.
        hazard_candidates = [(x, y) for y in range(n) for x in range(n)
                             if (x, y) not in reserved and self.terrain[y, x] != 2]
        self.rng.shuffle(hazard_candidates)
        for x, y in hazard_candidates[:self.config.fire_count]:
            self.fire[y, x] = 1
        for x, y in hazard_candidates[self.config.fire_count:self.config.fire_count+self.config.gas_count]:
            self.gas[y, x] = 1

    def _reveal(self):
        x, y = self.agent
        for yy in range(max(0, y-2), min(self.config.map_size, y+3)):
            for xx in range(max(0, x-2), min(self.config.map_size, x+3)):
                self.known[yy, xx] = 1

    def observe(self):
        # 5x5 local view, four channels: known, rubble/blocked, hazard, visible waiting victim.
        x, y = self.agent
        n = self.config.map_size
        local = np.zeros((5, 5, 4), dtype=np.float32)
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                xx, yy = x+dx, y+dy
                ix, iy = dx+2, dy+2
                if 0 <= xx < n and 0 <= yy < n:
                    local[iy, ix, 0] = self.known[yy, xx]
                    if self.known[yy, xx]:
                        local[iy, ix, 1] = 1.0 if self.terrain[yy, xx] in (1, 2) else 0.0
                        local[iy, ix, 2] = max(self.fire[yy, xx], self.gas[yy, xx])
                        for v in self.victims:
                            if v.x == xx and v.y == yy and v.status == "waiting":
                                local[iy, ix, 3] = (3-v.category) / 3.0
        nrm = max(1, n-1)
        waiting = [v for v in self.victims if v.status == "waiting" and self.known[v.y, v.x]]
        # Four nearest known victims, relative positions and urgency.
        target_features = np.zeros(12, dtype=np.float32)
        waiting.sort(key=lambda v: abs(v.x-x)+abs(v.y-y))
        for i, v in enumerate(waiting[:4]):
            target_features[i*3:i*3+3] = [(v.x-x)/nrm, (v.y-y)/nrm, (3-v.category)/3.0]
        scalars = np.array([
            self.energy / max(1, self.config.initial_energy),
            self.health / max(1, self.config.initial_health),
            len(self.carried) / max(1, self.config.carrying_capacity),
            self.steps / max(1, self.config.max_steps),
            x/nrm, y/nrm,
            (self.evac[0]-x)/nrm, (self.evac[1]-y)/nrm,
            min(1.0, len(waiting)/4.0),
            len([v for v in self.victims if v.status == "waiting"]) / max(1, len(self.victims)),
            float(any((x, y) == p for p in self.chargers)),
            float(any((x, y) == p for p in self.medical)),
        ], dtype=np.float32)
        return np.concatenate([local.flatten(), target_features, scalars]).astype(np.float32)

    @property
    def observation_size(self):
        return 5*5*4 + 12 + 12

    def _walkable(self, x, y):
        n = self.config.map_size
        return 0 <= x < n and 0 <= y < n and self.terrain[y, x] != 2

    def _spread_hazards(self):
        if self.steps and self.steps % self.config.hazard_spread_interval == 0:
            old = np.argwhere(self.fire == 1)
            for y, x in old:
                for dx, dy in [(1,0),(-1,0),(0,1),(0,-1)]:
                    xx, yy = x+dx, y+dy
                    if self._walkable(xx, yy) and self.rng.random() < 0.12:
                        self.fire[yy, xx] = 1
        if self.steps and self.steps % self.config.aftershock_interval == 0:
            # An aftershock may turn one rubble/ground tile into blocked terrain.
            candidates = [(x,y) for y in range(self.config.map_size) for x in range(self.config.map_size)
                          if (x,y) not in [tuple(self.agent), self.evac] and self.terrain[y,x] == 1]
            if candidates:
                xx, yy = candidates[int(self.rng.integers(len(candidates)))]
                self.terrain[yy, xx] = 2

    def step(self, action):
        if self.done:
            return self.observe(), 0.0, True, self.info()
        action = int(action)
        reward = -0.2
        old_pos = tuple(self.agent)
        self.steps += 1
        if action in (UP, DOWN, LEFT, RIGHT):
            dxdy = {UP:(0,-1), DOWN:(0,1), LEFT:(-1,0), RIGHT:(1,0)}[action]
            nx, ny = self.agent[0]+dxdy[0], self.agent[1]+dxdy[1]
            if self._walkable(nx, ny):
                self.agent = [nx, ny]
                cost = 2 if self.terrain[ny,nx] == 1 else 1
                self.energy = max(0, self.energy-cost)
                self.energy_used += cost
            else:
                reward -= 1.0
        elif action == WAIT:
            self.energy = max(0, self.energy-1)
            self.energy_used += 1
        elif action == INTERACT:
            pos = tuple(self.agent)
            if pos == self.evac and self.carried:
                for cat in self.carried:
                    reward += VICTIM_REWARDS[cat]
                    self.rescued += 1
                    if cat == CRITICAL:
                        self.critical_rescued += 1
                self.carried.clear()
                reward += 10.0
            elif len(self.carried) < self.config.carrying_capacity:
                target = next((v for v in self.victims if v.status == "waiting" and (v.x,v.y) == pos), None)
                if target:
                    target.status = "carried"
                    self.carried.append(target.category)
                    reward += 1.0
                else:
                    reward -= 1.0
            else:
                reward -= 1.0
        elif action == RECHARGE:
            pos = tuple(self.agent)
            if pos in self.chargers:
                self.energy = min(self.config.initial_energy, self.energy+45)
                reward += 2.0
            elif pos in self.medical:
                self.health = min(self.config.initial_health, self.health+40)
                reward += 2.0
            else:
                reward -= 1.0
        else:
            reward -= 1.0

        self._reveal()
        self._spread_hazards()
        x, y = self.agent
        if self.fire[y,x] or self.gas[y,x]:
            self.health = max(0, self.health-(8 if self.fire[y,x] else 5))
            self.hazard_encounters += 1
            reward -= 8.0
        # Victim survival clocks progress. Carried victims are protected during transport.
        for v in self.victims:
            if v.status == "waiting":
                v.survival -= 1
                if v.survival <= 0:
                    v.status = "lost"
                    reward -= 20.0
        # Small first-discovery reward; only newly revealed victim tiles count.
        for v in self.victims:
            if v.status == "waiting" and tuple(self.agent) == (v.x,v.y):
                pass
        if self.health <= 0:
            self.done, self.reason = True, "Robot health depleted"
            reward -= 50.0
        elif self.steps >= self.config.max_steps:
            self.done, self.reason = True, "Mission time expired"
        elif self.rescued + sum(v.status == "lost" for v in self.victims) == len(self.victims):
            self.done, self.reason = True, "All victims accounted for"
            if self.rescued > 0:
                reward += 25.0
        elif self.energy <= 0 and not self.carried and tuple(self.agent) not in self.chargers:
            # Allow the robot to continue only if it can still reach a charger; simplified fail-safe.
            if self.steps > 1:
                self.done, self.reason = True, "Energy depleted"
        self.total_reward += reward
        return self.observe(), float(reward), self.done, self.info()

    def info(self):
        waiting = sum(v.status == "waiting" for v in self.victims)
        lost = sum(v.status == "lost" for v in self.victims)
        return {
            "step": self.steps, "energy": self.energy, "health": self.health,
            "rescued": self.rescued, "critical_rescued": self.critical_rescued,
            "waiting": waiting, "lost": lost, "carried": len(self.carried),
            "capacity": self.config.carrying_capacity, "hazard_encounters": self.hazard_encounters,
            "energy_used": self.energy_used, "reward": self.total_reward,
            "done": self.done, "reason": self.reason,
        }

    def render(self, cell=42):
        import pygame
        n = self.config.map_size
        surface = pygame.Surface((n*cell, n*cell))
        colors = {
            "ground": (224,222,212), "rubble": (151,143,130), "blocked": (65,67,65),
            "fire": (220,112,45), "gas": (194,174,104), "road": (193,207,205)
        }
        surface.fill(colors["ground"])
        for y in range(n):
            for x in range(n):
                rect = pygame.Rect(x*cell, y*cell, cell, cell)
                if self.terrain[y,x] == 1: color = colors["rubble"]
                elif self.terrain[y,x] == 2: color = colors["blocked"]
                else: color = colors["ground"]
                if self.known[y,x] == 0:
                    color = (52,57,55)
                elif self.fire[y,x]: color = colors["fire"]
                elif self.gas[y,x]: color = colors["gas"]
                pygame.draw.rect(surface, color, rect)
                pygame.draw.rect(surface, (185,184,174), rect, 1)
                if self.known[y,x]:
                    if (x,y) == self.evac:
                        pygame.draw.rect(surface, (112,169,104), rect.inflate(-8,-8), border_radius=5)
                        pygame.draw.line(surface, (255,255,255), (x*cell+cell//2,y*cell+8), (x*cell+cell//2,y*cell+cell-8), 3)
                        pygame.draw.line(surface, (255,255,255), (x*cell+8,y*cell+cell//2), (x*cell+cell-8,y*cell+cell//2), 3)
                    for p in self.chargers:
                        if (x,y) == p:
                            pygame.draw.circle(surface, (55,116,163), rect.center, 10)
                            pygame.draw.line(surface, (255,255,255), (rect.centerx,rect.centery-6), (rect.centerx,rect.centery+6), 2)
                    for v in self.victims:
                        if v.status == "waiting" and (v.x,v.y) == (x,y):
                            col = [(205,63,56),(223,157,56),(91,151,91)][v.category]
                            pygame.draw.circle(surface, col, rect.center, 10)
                            pygame.draw.line(surface, (255,255,255), (rect.centerx-5,rect.centery), (rect.centerx+5,rect.centery), 2)
                            pygame.draw.line(surface, (255,255,255), (rect.centerx,rect.centery-5), (rect.centerx,rect.centery+5), 2)
        ax, ay = self.agent
        pygame.draw.circle(surface, (40,103,73), (ax*cell+cell//2, ay*cell+cell//2), 13)
        pygame.draw.circle(surface, (220,245,207), (ax*cell+cell//2, ay*cell+cell//2), 5)
        return surface
