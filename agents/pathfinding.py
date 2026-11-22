from collections import deque

def shortest_path(start, goals, walkable):
    """Return the first step toward the nearest reachable goal using BFS."""
    goals = set(goals)
    if not goals:
        return None
    q = deque([start])
    parent = {start: None}
    found = None
    while q:
        p = q.popleft()
        if p in goals:
            found = p
            break
        x, y = p
        for nxt in [(x,y-1),(x,y+1),(x-1,y),(x+1,y)]:
            if nxt not in parent and walkable(*nxt):
                parent[nxt] = p
                q.append(nxt)
    if found is None or found == start:
        return None
    cur = found
    while parent[cur] != start:
        cur = parent[cur]
    dx, dy = cur[0]-start[0], cur[1]-start[1]
    return {(0,-1):0,(0,1):1,(-1,0):2,(1,0):3}.get((dx,dy))
