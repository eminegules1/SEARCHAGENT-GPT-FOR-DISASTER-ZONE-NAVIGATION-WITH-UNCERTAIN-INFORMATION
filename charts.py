import matplotlib.animation as animation
import numpy as np

from config import GRID_N
from metrics import calculate_damage


def draw_static_elements(app, ax):
    grid = np.zeros((GRID_N, GRID_N))
    for (x, y), r in app.persistent_hazards.items():
        grid[int(y), int(x)] = r

    ax.imshow(
        grid,
        cmap="plasma",
        origin="lower",
        vmin=0,
        vmax=1.2,
        interpolation="nearest",
        alpha=0.7,
    )

    for (x, y), risk in app.persistent_hazards.items():
        if (x, y) == app.goal_pos:
            continue
        if risk >= 0.9:
            ax.scatter(x, y, s=150, c="red", marker="X", edgecolors="black", zorder=50)
        elif risk >= 0.5:
            ax.scatter(x, y, s=150, c="gray", marker="^", edgecolors="white", zorder=50)
        elif risk >= 0.2:
            ax.scatter(x, y, s=100, c="blue", marker="o", alpha=0.5, zorder=50)

    for (x, y) in app.persistent_blocked:
        if (x, y) == app.goal_pos or (x, y) == app.start_pos:
            continue
        ax.scatter(x, y, s=260, c="black", marker="s", edgecolors="white", linewidth=1, zorder=80)
        ax.text(x, y, "B", color="white", fontsize=8, ha="center", va="center", zorder=81)

    gx, gy = app.goal_pos
    ax.scatter(gx, gy, s=300, c="#00ff00", marker="s", edgecolors="white", linewidth=2, zorder=100)
    ax.text(gx, gy, "T", color="black", fontsize=9, ha="center", va="center", fontweight="bold", zorder=101)

    sx, sy = app.start_pos
    ax.scatter(sx, sy, s=200, c="white", marker="D", edgecolors="black", zorder=100)


def draw_comparison_graph(app):
    app.ax3.clear()
    app.ax3.set_facecolor("#ffffff")

    for ax in list(app.fig.axes):
        if ax not in (app.ax1, app.ax2, app.ax3):
            ax.remove()

    col_dist = "#00FFFF"
    col_safe = "#FF00FF"
    bar_width = 0.34

    for spine in app.ax3.spines.values():
        spine.set_edgecolor("#111111")
    app.ax3.grid(True, color="#dddddd", linestyle="--", linewidth=0.6, axis="y", alpha=0.8)
    app.ax3.set_axisbelow(True)

    app.ax3.set_title(
        f"METRICS ANALYSIS (Run {app.run_count})",
        color="#111111",
        fontsize=12,
        pad=12,
        fontname="Segoe UI",
        fontweight="bold",
    )

    if not app.last_metrics:
        app.ax3.text(
            0.5,
            0.5,
            "WAITING FOR DATA...",
            transform=app.ax3.transAxes,
            ha="center",
            va="center",
            color="#777777",
            fontname="Segoe UI",
        )
        app.ax3.set_xticks([])
        app.ax3.set_yticks([])
        return

    if app.metrics_history:
        ucs, astar = app.metrics_history[-1]
    else:
        ucs, astar = app.last_metrics
    labels = ["UCS (Blind)", "A* (Smart)"]
    x = np.arange(len(labels))

    dist_vals = [ucs["distance"], astar["distance"]]
    safe_vals = [ucs["safety"], astar["safety"]]

    bars1 = app.ax3.bar(
        x - bar_width / 2,
        dist_vals,
        width=bar_width,
        color=col_dist,
        alpha=0.95,
        edgecolor="white",
        linewidth=1,
        label="Distance (steps)",
    )

    ax_safe = app.ax3.twinx()
    ax_safe.set_facecolor("none")
    for spine in ax_safe.spines.values():
        spine.set_edgecolor("#666666")

    bars2 = ax_safe.bar(
        x + bar_width / 2,
        safe_vals,
        width=bar_width,
        color=col_safe,
        alpha=0.95,
        edgecolor="white",
        linewidth=1,
        label="Safety (0-100)",
    )

    left_max = max(dist_vals if dist_vals else [10])
    app.ax3.set_ylabel("Distance (steps)", color="#111111", fontname="Segoe UI", fontweight="bold")
    app.ax3.tick_params(axis="y", colors="#111111", labelsize=9)
    app.ax3.set_ylim(0, left_max * 1.25)

    ax_safe.set_ylabel("Safety (0-100)", color="#111111", fontname="Segoe UI", fontweight="bold")
    ax_safe.tick_params(axis="y", colors="#111111", labelsize=9)
    ax_safe.set_yticks([0, 25, 50, 75, 100])
    ax_safe.set_ylim(0, 110)

    app.ax3.set_xticks(x)
    app.ax3.set_xticklabels(labels, color="#111111", fontname="Segoe UI", fontweight="bold", fontsize=10)
    app.ax3.tick_params(axis="x", pad=6)

    def add_value_labels(bars, axis):
        y_offset = max(1, left_max * 0.04)
        for b in bars:
            h = b.get_height()
            axis.text(
                b.get_x() + b.get_width() / 2,
                h + y_offset,
                f"{int(h)}",
                ha="center",
                va="bottom",
                color="#111111",
                fontsize=9,
                fontweight="bold",
            )

    add_value_labels(bars1, app.ax3)
    for b in bars2:
        h = b.get_height()
        ax_safe.text(
            b.get_x() + b.get_width() / 2,
            h + 2,
            f"{h:.1f}",
            ha="center",
            va="bottom",
            color="#111111",
            fontsize=9,
            fontweight="bold",
        )

    h1, l1 = app.ax3.get_legend_handles_labels()
    h2, l2 = ax_safe.get_legend_handles_labels()
    app.ax3.legend(
        h1 + h2,
        l1 + l2,
        loc="upper center",
        ncol=2,
        facecolor="#ffffff",
        edgecolor="#111111",
        labelcolor="#111111",
        fontsize=8,
        framealpha=0.9,
        borderpad=0.6,
        handlelength=1.4,
        columnspacing=1.2,
    )

    u_fire, u_debris, u_smoke = ucs["hazard_breakdown"]
    a_fire, a_debris, a_smoke = astar["hazard_breakdown"]
    footer = (
        f"[UCS] Hazards: {ucs['hazards_hit']} | Risk: {ucs['total_risk']:.1f} "
        f"| Fire={u_fire} Debris={u_debris} Smoke={u_smoke}\n"
        f"[A* ] Hazards: {astar['hazards_hit']} | Risk: {astar['total_risk']:.1f} "
        f"| Fire={a_fire} Debris={a_debris} Smoke={a_smoke}"
    )
    app.ax3.text(
        0.5,
        -0.22,
        footer,
        transform=app.ax3.transAxes,
        ha="center",
        va="top",
        color="#444444",
        fontname="Segoe UI",
        fontsize=9,
    )


def animate_all(app):
    app.ax1.clear()
    app.ax2.clear()
    app.ax1.set_facecolor("#ffffff")
    app.ax2.set_facecolor("#ffffff")

    draw_comparison_graph(app)
    draw_static_elements(app, app.ax1)
    draw_static_elements(app, app.ax2)

    app.ax1.set_title(
        f"BLIND AGENT\nHit {calculate_damage(app, app.path_ucs)} Hazards",
        color="#111111",
        fontsize=11,
        fontname="Segoe UI",
    )
    app.ax2.set_title(
        f"SMART AGENT\nHit {calculate_damage(app, app.path_astar)} Hazards",
        color="#111111",
        fontsize=11,
        fontname="Segoe UI",
    )

    for ax in [app.ax1, app.ax2]:
        ax.set_xlim(-0.5, GRID_N - 0.5)
        ax.set_ylim(-0.5, GRID_N - 0.5)
        ax.grid(True, color="#dddddd", linestyle="-", linewidth=0.3, alpha=0.8)
        ax.set_xticks(range(GRID_N))
        ax.set_yticks(range(GRID_N))
        ax.tick_params(colors="#111111", labelsize=7)

    if app.path_ucs:
        ux, uy = zip(*app.path_ucs)
        app.ax1.plot(ux, uy, c="magenta", linestyle="--", linewidth=1, alpha=0.5)

    app.blind_dot, = app.ax1.plot([], [], "o", c="magenta", markersize=10, markeredgecolor="white")
    app.blind_line, = app.ax1.plot([], [], "-", c="magenta", linewidth=3, alpha=0.8)

    app.smart_dot, = app.ax2.plot([], [], "o", c="cyan", markersize=10, markeredgecolor="white")
    app.smart_line, = app.ax2.plot([], [], "-", c="cyan", linewidth=3, alpha=0.8)

    def update(frame):
        if app.path_ucs:
            idx_ucs = min(frame, len(app.path_ucs) - 1)
            path_b = app.path_ucs[: idx_ucs + 1]
            app.blind_line.set_data([p[0] for p in path_b], [p[1] for p in path_b])
            app.blind_dot.set_data([app.path_ucs[idx_ucs][0]], [app.path_ucs[idx_ucs][1]])

        if app.path_astar:
            idx_astar = min(frame, len(app.path_astar) - 1)
            path_s = app.path_astar[: idx_astar + 1]
            app.smart_line.set_data([p[0] for p in path_s], [p[1] for p in path_s])
            app.smart_dot.set_data([app.path_astar[idx_astar][0]], [app.path_astar[idx_astar][1]])

        return app.blind_dot, app.blind_line, app.smart_dot, app.smart_line

    max_frames = max(len(app.path_ucs), len(app.path_astar)) + 5
    app.anim = animation.FuncAnimation(
        app.fig, update, frames=max_frames, interval=220, repeat=False, blit=False
    )
    app.canvas.draw()
