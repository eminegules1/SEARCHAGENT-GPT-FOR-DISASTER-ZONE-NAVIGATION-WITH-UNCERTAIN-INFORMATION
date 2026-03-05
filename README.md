# SearchAgent-GPT

**Hybrid LLM-Enhanced Navigation Agent for Disaster-Zone Environments**

> A research project by Emine Güleş & Sümeyye Yıldız — Istanbul Atlas University, Software Engineering Department

---

## Overview

SearchAgent-GPT is a hybrid intelligent navigation agent designed for disaster-zone environments where hazard information is incomplete, ambiguous, or expressed in natural language. It combines **LLM-based semantic reasoning** with the classical **A\* search algorithm** to enable safety-aware path planning under uncertainty.

The key insight: traditional search algorithms like A\* and UCS cannot interpret qualitative human reports (e.g., *"there's smoke in the north corridor"*). SearchAgent-GPT bridges this gap by using a local LLM to convert such descriptions into bounded numerical risk estimates, which are then incorporated into A\*'s path cost function — biasing the planner away from hazardous regions without modifying the underlying heuristic.

**Uniform Cost Search (UCS)** is included as a non-LLM baseline for comparative evaluation.

---

## Features

- **Hybrid symbolic–semantic planning** — LLM acts as an advisory risk module, not a planner
- **Natural-language hazard interpretation** — transforms unstructured text into risk cost penalties
- **Safety-aware A\* search** — biases paths away from inferred hazards while preserving optimality guarantees
- **GridWorld simulation** — fully simulated 15×15 disaster-zone environment
- **Side-by-side comparison** — A\* (risk-aware) vs. UCS (blind baseline)
- **Interactive Tkinter UI** — load scenarios, run planners, view results
- **Performance metrics** — path length, hazards encountered, safety score, nodes expanded, runtime

---

## Architecture

The system follows a modular **Perception → Reasoning → Action** (PRA) architecture:

| Layer | Responsibility | Module(s) |
|---|---|---|
| **Perception** | Grid state, cell types, start/goal positions | `config.py`, `scenarios.py` |
| **Reasoning** | LLM interprets hazard text → bounded risk values | `llm.py`, `llm_runner.py` |
| **Action** | A\* and UCS planners execute over the grid | `problem.py`, `search4e.py` |
| **Evaluation** | Metrics collection and comparison | `metrics.py` |
| **Visualization & UI** | Interactive interface and path/chart rendering | `ui.py`, `charts.py` |

The LLM **only** produces risk annotations that affect traversal cost — it does not choose actions, modify heuristics, or control the planner directly.

---

## Tech Stack

| Category | Tool |
|---|---|
| Language | Python 3.11 |
| Planning Algorithms | A\* (risk-aware), Uniform Cost Search |
| Search Framework | AIMA-based (`search4e`) |
| AI Model | LLaMA 3.1 |
| LLM Runtime | Ollama (local inference) |
| LLM Communication | HTTP + JSON (`requests`) |
| Simulation | GridWorld (2D, 15×15) |
| UI | Tkinter |
| Visualization | Matplotlib |
| Concurrency | Python `threading` |
| Numerics | NumPy |

---

## How It Works

1. **Environment setup** — a 15×15 GridWorld is loaded with blocked cells, a start position (0,0), and a goal position.
2. **Hazard input** — natural-language descriptions (e.g., *"fire in the center"*, *"blocked path near the goal"*) are provided as input.
3. **LLM risk analysis** — LLaMA 3.1 interprets the descriptions and returns bounded numerical risk estimates for approximate grid regions.
4. **Cost augmentation** — risk values are validated and added as penalties to the A\* path cost function `g(n)`.
5. **Planning** — A\* searches for the lowest-cost path using `f(n) = g(n) + h(n)`, where `h(n)` is Manhattan distance (unchanged). UCS runs in parallel on the same grid without any risk information.
6. **Comparison** — metrics are collected and visualized for both planners.

---

## Evaluation Metrics

| Metric | Description |
|---|---|
| `distance` | Total movement steps to goal |
| `hazards_hit` | Grid positions along the path with elevated risk penalties |
| `safety_score` | Normalized score reflecting risk exposure (higher = safer) |
| `nodes_expanded` | Nodes expanded during search (computational effort) |
| `runtime (ms)` | Wall-clock planning time |

---

## Results Summary

Experimental results in the simulated GridWorld show:

- **SearchAgent-GPT (A\*)** avoids regions inferred as hazardous even when they are not explicitly blocked, leading to higher safety scores than UCS.
- **UCS** finds shorter geometric paths but exhibits no behavioral response to semantic hazard descriptions.
- A\* generally expands fewer nodes than UCS due to heuristic guidance, with minor overhead from risk-augmented cost evaluation.
- Runtime for both planners is comparable; the LLM overhead is bounded and acceptable for the simulation scale.

The trade-off: risk-aware paths may be slightly longer in step count but significantly safer — a desirable property in disaster-response contexts.

---

## References

1. S. Russell and P. Norvig, *Artificial Intelligence: A Modern Approach*, 4th ed., Pearson, 2020.
2. S. Meng et al., "LLM-A\*: Large Language Model Enhanced Incremental Heuristic Search on Path Planning," EMNLP 2024.
3. M. Zajac, "Heuristic, Hybrid, and LLM-Assisted Heuristics for Container Yard Strategies Under Incomplete Information," *Applied Sciences*, 2025.
4. D. Shah et al., "Navigation with Large Language Models: Semantic Guesswork as a Heuristic for Planning," *arXiv*, 2023.

---

*Istanbul Atlas University — Software Engineering Department*
