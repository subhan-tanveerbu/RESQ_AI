from agents.pathfinding import shortest_path
from environment.rescue_env import INTERACT, RECHARGE, WAIT

class RuleBasedAgent:
    """Greedy informed baseline using the same discovered information as the observation."""
    def act(self, observation, env, explore=False):
        pos = tuple(env.agent)
        if pos == env.evac and env.carried:
            return INTERACT
        if len(env.carried) >= env.config.carrying_capacity:
            goals = [env.evac]
        elif env.energy < 24:
            goals = env.chargers
            if pos in goals:
                return RECHARGE
        elif env.health < 35 and pos in env.medical:
            return RECHARGE
        else:
            known_victims = [v for v in env.victims if v.status == "waiting" and env.known[v.y,v.x]]
            if known_victims:
                # Prioritize urgency, then distance.
                target = min(known_victims, key=lambda v: (v.category, abs(v.x-pos[0])+abs(v.y-pos[1])))
                goals = [(target.x, target.y)]
                if pos == (target.x, target.y):
                    return INTERACT
            else:
                # Explore toward the closest unknown tile.
                unknown = [(x,y) for y in range(env.config.map_size) for x in range(env.config.map_size)
                           if not env.known[y,x] and env.terrain[y,x] != 2]
                goals = unknown
        def walkable(x,y):
            return env._walkable(x,y) and not env.fire[y,x] and not env.gas[y,x]
        action = shortest_path(pos, goals, walkable)
        if action is not None:
            return action
        if pos in env.chargers and env.energy < env.config.initial_energy:
            return RECHARGE
        return WAIT
