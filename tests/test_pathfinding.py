from agents.pathfinding import shortest_path

def test_pathfinding_returns_first_action():
    def walkable(x,y):
        return 0 <= x < 4 and 0 <= y < 4
    action = shortest_path((0,0),[(2,0)],walkable)
    assert action in (0,1,2,3)
