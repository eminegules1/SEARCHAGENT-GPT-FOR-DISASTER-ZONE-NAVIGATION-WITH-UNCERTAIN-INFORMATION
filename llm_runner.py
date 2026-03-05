import threading
from tkinter import messagebox

from config import GRID_N
from llm import get_llama_analysis, sanitize_llm_data


def run_simulation(app):
    report = app.input_entry.get().strip()
    if not report:
        return
    if app._busy:
        return

    app._busy = True
    app.run_btn.config(state="disabled")
    app.info_lbl.config(text="UPDATING TACTICAL MAP (LLM)...", fg="#555555")
    app.root.update_idletasks()

    threading.Thread(target=run_simulation_worker, args=(app, report), daemon=True).start()


def run_simulation_worker(app, report: str):
    try:
        data = get_llama_analysis(report)
        data = sanitize_llm_data(data, n=GRID_N)
        app.root.after(0, lambda: apply_llm_result(app, data))
    except Exception as e:
        app.root.after(0, lambda: finish_with_error(app, e))


def apply_llm_result(app, data: dict):
    try:
        new_hazards = data.get("hazards", [])
        new_bias = data.get("direction_bias")
        new_survivor = data.get("survivor")
        summary = data.get("summary", "")

        new_blocked = data.get("blocked", [])
        for b in new_blocked:
            if isinstance(b, list) and len(b) >= 2:
                app.persistent_blocked.add((int(b[0]), int(b[1])))

        for item in new_hazards:
            raw_coord = item.get("coord", [])
            if not isinstance(raw_coord, list) or len(raw_coord) < 2:
                continue
            cx, cy = int(raw_coord[0]), int(raw_coord[1])
            risk = float(item.get("risk", 0.0))
            radius = int(item.get("radius", 0))

            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    nx, ny = cx + dx, cy + dy
                    if 0 <= nx < GRID_N and 0 <= ny < GRID_N:
                        if (nx, ny) in app.persistent_blocked:
                            continue
                        old_risk = app.persistent_hazards.get((nx, ny), 0.0)
                        app.persistent_hazards[(nx, ny)] = max(old_risk, risk)

        if new_survivor and isinstance(new_survivor, list) and len(new_survivor) >= 2:
            app.persistent_survivor = tuple(map(int, new_survivor))

        if new_bias:
            app.persistent_bias = new_bias

        app.start_pos = (0, 0)
        app.goal_pos = app.persistent_survivor if app.persistent_survivor else (GRID_N - 1, GRID_N - 1)

        app.persistent_blocked.discard(app.start_pos)
        app.persistent_blocked.discard(app.goal_pos)

        app.info_lbl.config(
            text=f"{summary} | TARGET: {app.goal_pos} | hazards={len(app.persistent_hazards)} | blocked={len(app.persistent_blocked)}",
            fg="#333333",
        )

        app._replan_and_draw()
        finish_ok(app)
    except Exception as e:
        finish_with_error(app, e)


def finish_ok(app):
    app._busy = False
    app.run_btn.config(state="normal")


def finish_with_error(app, e: Exception):
    messagebox.showerror("Error", str(e))
    app._busy = False
    app.run_btn.config(state="normal")
