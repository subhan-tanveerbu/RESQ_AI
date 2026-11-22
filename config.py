from dataclasses import dataclass, asdict
import json
from pathlib import Path

@dataclass
class RescueConfig:
    map_size: int = 12
    victim_count: int = 6
    max_steps: int = 250
    initial_energy: int = 100
    initial_health: int = 100
    carrying_capacity: int = 2
    fire_count: int = 2
    gas_count: int = 2
    rubble_count: int = 12
    hazard_spread_interval: int = 18
    aftershock_interval: int = 55
    seed: int = 7
    difficulty: str = "normal"

    def validate(self):
        self.map_size = max(8, min(20, int(self.map_size)))
        self.victim_count = max(1, min(15, int(self.victim_count)))
        self.max_steps = max(50, int(self.max_steps))
        self.carrying_capacity = max(1, min(4, int(self.carrying_capacity)))
        self.fire_count = max(0, int(self.fire_count))
        self.gas_count = max(0, int(self.gas_count))
        self.rubble_count = max(0, int(self.rubble_count))
        if self.difficulty not in {"easy", "normal", "hard"}:
            self.difficulty = "normal"
        return self

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path):
        return cls(**json.loads(Path(path).read_text())).validate()
