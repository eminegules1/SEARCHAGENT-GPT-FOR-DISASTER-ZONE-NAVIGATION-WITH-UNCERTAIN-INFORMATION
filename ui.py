import tkinter as tk
from tkinter import messagebox

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.gridspec import GridSpec

from charts import animate_all
from config import GRID_N
from llm_runner import run_simulation
from metrics import calculate_damage, hazard_breakdown, path_distance, path_safety_score, path_total_risk
from problem import DisasterProblem
from scenarios import load_scenario_1, load_scenario_2, load_scenario_3
from search4e import astar_search_metrics, uniform_cost_search_metrics

plt.style.use("default")

class DisasterApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Tactical Radar (15x15) - Scenarios + LLM")
        self.root.geometry("1400x900")
        self.root.configure(bg="#ffffff")

        self._busy = False

        # MEMORY
        self.persistent_hazards = {}
        self.persistent_survivor = None
        self.persistent_bias = None
        self.persistent_blocked = set()

        # defaults
        self.start_pos = (0, 0)
        self.goal_pos = (GRID_N - 1, GRID_N - 1)
        self.path_ucs = []
        self.path_astar = []

        # metrics
        self.run_count = 0
        self.last_metrics = None
        self.metrics_history = []
        self.history_window = None
        self.history_text = None

        # --- Control Panel ---
        frame = tk.Frame(root, pady=14, bg="#f2f2f2")
        frame.pack(side=tk.TOP, fill=tk.X)

        tk.Label(frame, text="REPORT:", bg="#f2f2f2", fg="#111111",
                 font=("Segoe UI", 14, "bold")).pack(side=tk.LEFT, padx=15)

        self.input_entry = tk.Entry(frame, width=55, font=("Segoe UI", 12),
                                    bg="#ffffff", fg="#111111", insertbackground="#111111")
        self.input_entry.pack(side=tk.LEFT, padx=5)
        self.input_entry.insert(0, "Fire at 7,7. Survivor at top right.")

        self.run_btn = tk.Button(frame, text="ADD INFO & RUN (LLM)", command=self.run_simulation,
                                 bg="#a7e3a7", fg="#0a0a0a", font=("Segoe UI", 11, "bold"),
                                 activebackground="#8fd48f")
        self.run_btn.pack(side=tk.LEFT, padx=10)

        self.reset_btn = tk.Button(frame, text="RESET MAP", command=self.clear_memory,
                                   bg="#f0a0a0", fg="#0a0a0a", font=("Segoe UI", 11, "bold"),
                                   activebackground="#e38c8c")
        self.reset_btn.pack(side=tk.LEFT, padx=10)

        # --- Scenario Buttons ---
        scen_frame = tk.Frame(root, pady=8, bg="#f2f2f2")
        scen_frame.pack(side=tk.TOP, fill=tk.X)

        tk.Label(scen_frame, text="SCENARIOS:", bg="#f2f2f2", fg="#111111",
                 font=("Segoe UI", 12, "bold")).pack(side=tk.LEFT, padx=15)

        tk.Button(scen_frame, text="Scenario 1", command=self.load_scenario_1,
                  bg="#cfe2f3", fg="#0a0a0a", font=("Segoe UI", 11, "bold")).pack(side=tk.LEFT, padx=8)

        tk.Button(scen_frame, text="Scenario 2", command=self.load_scenario_2,
                  bg="#f6d5b5", fg="#0a0a0a", font=("Segoe UI", 11, "bold")).pack(side=tk.LEFT, padx=8)

        tk.Button(scen_frame, text="History", command=self.show_history,
                  bg="#cdebd6", fg="#0a0a0a", font=("Segoe UI", 11, "bold")).pack(side=tk.LEFT, padx=8)

        legend_frame = tk.Frame(root, pady=4, bg="#f8f8f8")
        legend_frame.pack(side=tk.TOP, fill=tk.X)

        tk.Label(legend_frame, text="LEGEND:", bg="#f8f8f8", fg="#333333",
                 font=("Segoe UI", 10, "bold")).pack(side=tk.LEFT, padx=12)

        def add_icon(label, draw_fn):
            canvas = tk.Canvas(
                legend_frame,
                width=18,
                height=18,
                bg="#ffffff",
                highlightthickness=1,
                highlightbackground="#cccccc",
            )
            draw_fn(canvas)
            canvas.pack(side=tk.LEFT, padx=(6, 2))
            tk.Label(legend_frame, text=label, bg="#f8f8f8", fg="#333333",
                     font=("Segoe UI", 10)).pack(side=tk.LEFT, padx=6)

        add_icon("Fire", lambda c: (
            c.create_line(4, 4, 14, 14, fill="#cc0000", width=2),
            c.create_line(14, 4, 4, 14, fill="#cc0000", width=2),
        ))
        add_icon("Debris", lambda c: c.create_polygon(9, 3, 15, 14, 3, 14, outline="#666666", fill="#aaaaaa"))
        add_icon("Smoke/Water", lambda c: c.create_oval(4, 4, 14, 14, outline="#3b6ea5", fill="#7fb5ff"))
        add_icon("Blocked", lambda c: (
            c.create_rectangle(3, 3, 15, 15, outline="#000000", fill="#000000"),
            c.create_text(9, 9, text="B", fill="#ffffff", font=("Segoe UI", 8, "bold")),
        ))
        add_icon("Start", lambda c: (
            c.create_polygon(9, 2, 16, 9, 9, 16, 2, 9, outline="#000000", fill="#ffffff"),
            c.create_text(9, 9, text="D", fill="#000000", font=("Segoe UI", 8, "bold")),
        ))
        add_icon("Target", lambda c: (
            c.create_rectangle(3, 3, 15, 15, outline="#008800", fill="#9aff9a"),
            c.create_text(9, 9, text="T", fill="#000000", font=("Segoe UI", 8, "bold")),
        ))

        self.info_lbl = tk.Label(root, text="SYSTEM ONLINE. WAITING...",
                                 bg="#ffffff", fg="#333333", font=("Segoe UI", 11))
        self.info_lbl.pack(side=tk.TOP, pady=5)

        # --- Figure layout ---
        self.fig = plt.figure(figsize=(14, 8.0), facecolor="#ffffff")
        gs = GridSpec(2, 2, height_ratios=[3.2, 1.2], figure=self.fig)
        self.ax1 = self.fig.add_subplot(gs[0, 0])
        self.ax2 = self.fig.add_subplot(gs[0, 1])
        self.ax3 = self.fig.add_subplot(gs[1, :])

        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.reset_visuals()

    # ----------------------------
    # RESET
    # ----------------------------
    def clear_memory(self):
        self.persistent_hazards = {}
        self.persistent_survivor = None
        self.persistent_bias = None
        self.persistent_blocked = set()

        self.info_lbl.config(text="MEMORY CLEARED.", fg="#333333")

        self.start_pos = (0, 0)
        self.goal_pos = (GRID_N - 1, GRID_N - 1)
        self.path_ucs = []
        self.path_astar = []
        self.last_metrics = None
        self.reset_visuals()

    def reset_visuals(self):
        for ax in [self.ax1, self.ax2]:
            ax.clear()
            ax.set_facecolor("#ffffff")
            ax.set_xlim(-0.5, GRID_N - 0.5)
            ax.set_ylim(-0.5, GRID_N - 0.5)
            ax.grid(True, color="#dddddd", linestyle="-", linewidth=0.5, alpha=0.6)
            ax.set_xticks(range(GRID_N))
            ax.set_yticks(range(GRID_N))
            ax.tick_params(colors="#111111", labelsize=7)
            for spine in ax.spines.values():
                spine.set_edgecolor("#111111")

        self.ax1.set_title("MONITOR 1: BLIND AGENT (UCS)", color="#111111", fontsize=12, pad=10, fontname="Segoe UI")
        self.ax2.set_title("MONITOR 2: SMART AGENT (A*)", color="#111111", fontsize=12, pad=10, fontname="Segoe UI")

        self.ax3.clear()
        self.ax3.set_facecolor("#ffffff")
        for spine in self.ax3.spines.values():
            spine.set_edgecolor("#111111")
        self.ax3.tick_params(colors="#111111", labelsize=8)
        self.ax3.grid(True, color="#dddddd", linestyle="-", linewidth=0.5, alpha=0.6)
        self.ax3.set_title("PERFORMANCE COMPARISON (UCS vs A*)", color="#111111", fontsize=11, pad=8, fontname="Segoe UI")
        self.ax3.text(0.5, 0.5, "Run a scenario to view metrics",
                      transform=self.ax3.transAxes, ha="center", va="center",
                      color="#444444", fontsize=10, fontname="Segoe UI")
        self.ax3.set_xticks([])
        self.ax3.set_yticks([])
        self.canvas.draw()

    def show_history(self):
        if self.history_window and self.history_window.winfo_exists():
            self._update_history_view()
            self.history_window.lift()
            return

        if not self.metrics_history:
            messagebox.showinfo("History", "No runs recorded yet.")
            return

        win = tk.Toplevel(self.root)
        win.title("Run History")
        win.configure(bg="#050505")
        win.geometry("780x420")

        text = tk.Text(win, wrap="none", bg="#0b0b0b", fg="#e0e0e0",
                       font=("Segoe UI", 10))
        text.pack(fill=tk.BOTH, expand=True)

        self.history_window = win
        self.history_text = text
        self._update_history_view()

    def _update_history_view(self):
        if not self.history_text:
            return
        lines = []
        for idx, (ucs, astar) in enumerate(self.metrics_history, start=1):
            lines.append(f"Run {idx}")
            lines.append(
                "  UCS  | dist={d} | safety={s:.1f} | hazards={h} | nodes={n} | time={t:.1f}ms".format(
                    d=ucs["distance"], s=ucs["safety"], h=ucs["hazards_hit"],
                    n=ucs["nodes_expanded"], t=ucs["runtime_ms"]
                )
            )
            lines.append(
                "  A*   | dist={d} | safety={s:.1f} | hazards={h} | nodes={n} | time={t:.1f}ms".format(
                    d=astar["distance"], s=astar["safety"], h=astar["hazards_hit"],
                    n=astar["nodes_expanded"], t=astar["runtime_ms"]
                )
            )
            lines.append("")

        self.history_text.config(state="normal")
        self.history_text.delete("1.0", tk.END)
        self.history_text.insert("1.0", "\n".join(lines))
        self.history_text.config(state="disabled")

    # ----------------------------
    # THREADED LLM RUN
    # ----------------------------
    def run_simulation(self):
        run_simulation(self)

    # ----------------------------
    # SCENARIOS (Instructor request)
    # ----------------------------
    def _apply_scenario(self, name: str, blocked, hazards, survivor, start=None, bias=None):
        # clear current map
        self.persistent_hazards = {}
        self.persistent_blocked = set()
        self.persistent_survivor = None
        self.persistent_bias = None
        self.last_metrics = None
        self.metrics_history = []
        self.run_count = 0

        for (x, y) in blocked:
            if 0 <= x < GRID_N and 0 <= y < GRID_N:
                self.persistent_blocked.add((x, y))

        for (x, y), risk in hazards.items():
            if 0 <= x < GRID_N and 0 <= y < GRID_N:
                if (x, y) not in self.persistent_blocked:
                    self.persistent_hazards[(x, y)] = max(0.0, min(1.0, float(risk)))

        if survivor is not None:
            sx, sy = survivor
            if 0 <= sx < GRID_N and 0 <= sy < GRID_N:
                self.persistent_survivor = (sx, sy)

        self.persistent_bias = bias

        self.start_pos = (0, 0)
        self.goal_pos = self.persistent_survivor if self.persistent_survivor else (GRID_N - 1, GRID_N - 1)

        self.persistent_blocked.discard(self.start_pos)
        self.persistent_blocked.discard(self.goal_pos)

        self.info_lbl.config(
            text=f"✅ {name} loaded | TARGET: {self.goal_pos} | hazards={len(self.persistent_hazards)} | blocked={len(self.persistent_blocked)}",
            fg="#333333",
        )

        self._replan_and_draw()



    def load_scenario_1(self):
        load_scenario_1(self)

    def load_scenario_2(self):
        load_scenario_2(self)

    def load_scenario_3(self):
        load_scenario_3(self)

    # ----------------------------
    # REPLAN + DRAW
    # ----------------------------
    def _replan_and_draw(self):
        # UCS
        prob_ucs = DisasterProblem(
            self.start_pos,
            self.goal_pos,
            hazards=self.persistent_hazards,
            blocked=self.persistent_blocked,
            direction_bias=None,
            use_risk=False,
            n=GRID_N,
        )
        node_ucs, m_ucs = uniform_cost_search_metrics(prob_ucs)
        self.path_ucs = [n.state for n in node_ucs.path()] if node_ucs else []

        # A*
        prob_astar = DisasterProblem(
            self.start_pos,
            self.goal_pos,
            hazards=self.persistent_hazards,
            blocked=self.persistent_blocked,
            direction_bias=self.persistent_bias,
            use_risk=True,
            n=GRID_N,
        )
        node_astar, m_astar = astar_search_metrics(prob_astar, prob_astar.h)
        self.path_astar = [n.state for n in node_astar.path()] if node_astar else []

        # metrics
        self.run_count += 1
        ucs_info = {
            "algo": "UCS",
            "distance": path_distance(self.path_ucs),
            "safety": path_safety_score(self, self.path_ucs),
            "total_risk": path_total_risk(self, self.path_ucs),
            "hazards_hit": calculate_damage(self, self.path_ucs),
            "hazard_breakdown": hazard_breakdown(self, self.path_ucs),
            "nodes_expanded": int(m_ucs.nodes_expanded),
            "runtime_ms": float(m_ucs.elapsed_ms),
        }
        astar_info = {
            "algo": "A*",
            "distance": path_distance(self.path_astar),
            "safety": path_safety_score(self, self.path_astar),
            "total_risk": path_total_risk(self, self.path_astar),
            "hazards_hit": calculate_damage(self, self.path_astar),
            "hazard_breakdown": hazard_breakdown(self, self.path_astar),
            "nodes_expanded": int(m_astar.nodes_expanded),
            "runtime_ms": float(m_astar.elapsed_ms),
        }
        self.last_metrics = (ucs_info, astar_info)
        self.metrics_history.append(self.last_metrics)
        if self.history_window and self.history_window.winfo_exists():
            self._update_history_view()

        if not self.path_astar and not self.path_ucs:
            messagebox.showerror("FAILURE", "Target Unreachable!")
            self.reset_visuals()
            return

        animate_all(self)

