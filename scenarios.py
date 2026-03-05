from config import GRID_N


def load_scenario_1(app):
    """
    SCENARIO 1: THE CRUMBLING BRIDGE
    Concept: Two distinct paths separated by a wall.
    1. The 'Bridge': Short, direct, but unstable (Risk 0.8).
    2. The 'Detour': Long, winding, but perfectly clear (Risk 0.0).

    Expected Result:
    - BLIND (UCS): Takes the Bridge (shorter distance).
    - SMART (A*): Takes the Detour (avoids risk).
    """
    blocked = []
    hazards = {}

    mid = GRID_N // 2

    # Dividing wall (blocked) with a gap at the middle
    for y in range(2, GRID_N - 2):
        blocked.append((mid, y))

    blocked.remove((mid, mid))
    for x in range(mid - 2, mid + 3):
        hazards[(x, mid)] = 0.9

    survivor = (GRID_N - 1, mid)
    app._apply_scenario(
        "SCEN 1: Crumbling Bridge vs Long Road",
        blocked,
        hazards,
        survivor,
        start=(0, mid),
        bias=None,
    )


def load_scenario_2(app):
    """
    SCENARIO 2: AFTERSHOCK MULTI-HAZARD MAP
    Mixed hazards with blocked islands; A* should route around risk clusters.
    """
    blocked = []
    for x in range(4, 8):
        for y in range(5, 11):
            blocked.append((x, y))
    for x in range(10, 13):
        for y in range(2, 6):
            blocked.append((x, y))

    hazards = {
        (2, 4): 0.7,    # debris
        (3, 4): 0.7,
        (4, 4): 0.7,
        (5, 4): 0.6,
        (2, 9): 0.6,
        (3, 9): 0.6,
        (9, 7): 0.8,
        (10, 7): 0.8,
        (11, 7): 0.7,
        (9, 8): 1.0,    # fire
        (10, 8): 1.0,   # fire
        (11, 8): 0.7,   # smoke
        (9, 9): 0.7,
        (10, 9): 0.6,
    }

    survivor = (12, 8)
    app._apply_scenario("SCEN 2: Aftershock multi-hazard map", blocked, hazards, survivor, bias="east")


def load_scenario_3(app):
    # Scenario 3: Aftershock multi-hazard islands + direction bias east
    blocked = []
    for x in range(4, 8):
        for y in range(5, 11):
            blocked.append((x, y))
    for x in range(10, 13):
        for y in range(2, 6):
            blocked.append((x, y))

    hazards = {
        (2, 4): 0.7,    # debris
        (3, 4): 0.7,
        (4, 4): 0.7,
        (5, 4): 0.6,
        (2, 9): 0.6,
        (3, 9): 0.6,
        (9, 7): 0.8,
        (10, 7): 0.8,
        (11, 7): 0.7,
        (9, 8): 1.0,    # fire
        (10, 8): 1.0,   # fire
        (11, 8): 0.7,   # smoke
        (9, 9): 0.7,
        (10, 9): 0.6,
    }

    survivor = (12, 8)
    app._apply_scenario("Scenario 3: Aftershock multi-hazard map", blocked, hazards, survivor, bias="east")
