from search4e import Problem

from config import GRID_N

class DisasterProblem(Problem):
    def __init__(self, initial, goal, hazards=None, blocked=None, direction_bias=None, use_risk=True, n=GRID_N):
        super().__init__(initial, goal)
        self.hazards = hazards or {}
        self.blocked = set(blocked or [])
        self.direction_bias = direction_bias
        self.use_risk = use_risk
        self.n = n

    def actions(self, state):
        x, y = state
        res = []
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.n and 0 <= ny < self.n:
                if (nx, ny) in self.blocked:
                    continue
                res.append((nx, ny))
        return res

    def result(self, state, action):
        return action

    def path_cost(self, c, state1, action, state2):
        base_cost = 1
        if not self.use_risk:
            return c + base_cost

        risk = self.hazards.get(state2, 0.0)

        # Riski görünür kılmak için büyük ceza
        if risk >= 0.9:
            risk_penalty = 5000
        else:
            risk_penalty = risk * 500

        dx, dy = state2[0] - state1[0], state2[1] - state1[1]
        dir_penalty = 0

        if self.direction_bias:
            if (self.direction_bias == "north" and dy > 0) or \
               (self.direction_bias == "south" and dy < 0) or \
               (self.direction_bias == "east" and dx > 0) or \
               (self.direction_bias == "west" and dx < 0):
                dir_penalty = 200

        return c + base_cost + risk_penalty + dir_penalty

    def h(self, node):
        return abs(node.state[0] - self.goal[0]) + abs(node.state[1] - self.goal[1])
