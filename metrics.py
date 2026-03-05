def calculate_damage(app, path):
    return sum(1 for (x, y) in path if app.persistent_hazards.get((x, y), 0.0) > 0.5)


def hazard_breakdown(app, path):
    fire = 0
    debris = 0
    smoke = 0
    for (x, y) in path:
        risk = float(app.persistent_hazards.get((x, y), 0.0))
        if risk >= 0.9:
            fire += 1
        elif risk >= 0.5:
            debris += 1
        elif risk >= 0.2:
            smoke += 1
    return fire, debris, smoke


def path_distance(path):
    return max(len(path) - 1, 0)


def path_total_risk(app, path):
    total = 0.0
    for (x, y) in path[1:]:
        total += float(app.persistent_hazards.get((x, y), 0.0)) * 2.0
    return total


def path_safety_score(app, path):
    dist = path_distance(path)
    if dist == 0:
        return 100.0
    avg_risk = path_total_risk(app, path) / dist
    return 100.0 * (1.0 - min(max(avg_risk, 0.0), 1.0))
