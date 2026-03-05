"""Minimal AIMA-style search utilities for this project.

`main.py` expects:
  - Problem
  - astar_search
  - uniform_cost_search

The original repo also had `search4e.py`, but it includes notebook demo cells
that execute on import. This file is a clean, import-safe implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
import time
import heapq
from typing import Any, Callable, Dict, Generic, Iterable, Iterator, List, Optional, Sequence, Set, Tuple, TypeVar

TState = TypeVar("TState")
TAction = TypeVar("TAction")


class Problem(Generic[TState, TAction]):
    """Abstract problem class."""

    def __init__(self, initial: TState, goal: Optional[TState] = None):
        self.initial = initial
        self.goal = goal

    # --- override these in subclasses ---
    def actions(self, state: TState) -> Iterable[TAction]:
        raise NotImplementedError

    def result(self, state: TState, action: TAction) -> TState:
        raise NotImplementedError

    def goal_test(self, state: TState) -> bool:
        return state == self.goal

    def path_cost(self, c: float, state1: TState, action: TAction, state2: TState) -> float:
        return c + 1

    # Heuristic: subclasses may override as `h(self, node)`
    def h(self, node: "Node[TState, TAction]") -> float:
        return 0.0


@dataclass(order=False)
class Node(Generic[TState, TAction]):
    state: TState
    parent: Optional["Node[TState, TAction]"] = None
    action: Optional[TAction] = None
    path_cost: float = 0.0
    depth: int = 0

    def __post_init__(self) -> None:
        if self.parent is not None:
            self.depth = self.parent.depth + 1

    def expand(self, problem: Problem[TState, TAction]) -> List["Node[TState, TAction]"]:
        children: List[Node[TState, TAction]] = []
        for action in problem.actions(self.state):
            next_state = problem.result(self.state, action)
            next_cost = problem.path_cost(self.path_cost, self.state, action, next_state)
            children.append(Node(next_state, self, action, next_cost))
        return children

    def solution(self) -> List[TAction]:
        return [node.action for node in self.path()[1:]]  # type: ignore[misc]

    def path(self) -> List["Node[TState, TAction]"]:
        node: Optional[Node[TState, TAction]] = self
        back: List[Node[TState, TAction]] = []
        while node is not None:
            back.append(node)
            node = node.parent
        return list(reversed(back))


class PriorityQueue(Generic[TState, TAction]):
    """A min-priority queue of Nodes with update-by-state support."""

    def __init__(self, f: Callable[[Node[TState, TAction]], float]):
        self.f = f
        self._heap: List[Tuple[float, int, Node[TState, TAction]]] = []
        self._best: Dict[Any, float] = {}   # state -> best priority
        self._counter = 0

    def push(self, node: Node[TState, TAction]) -> None:
        p = float(self.f(node))
        state_key = node.state
        # Only keep if this is the best seen for this state
        if state_key in self._best and p >= self._best[state_key]:
            return
        self._best[state_key] = p
        self._counter += 1
        heapq.heappush(self._heap, (p, self._counter, node))

    def pop(self) -> Node[TState, TAction]:
        while self._heap:
            p, _, node = heapq.heappop(self._heap)
            # Skip stale entries
            if self._best.get(node.state, float('inf')) == p:
                return node
        raise IndexError("pop from empty PriorityQueue")

    def __len__(self) -> int:
        return len(self._heap)


@dataclass
class SearchMetrics:
    """Lightweight instrumentation for comparing algorithms."""

    nodes_popped: int = 0          # number of nodes removed from the frontier
    nodes_expanded: int = 0        # number of nodes for which we generated successors
    nodes_generated: int = 0       # number of child nodes generated
    max_frontier_size: int = 0     # peak frontier size (heap size)
    elapsed_ms: float = 0.0        # wall time in milliseconds


def best_first_graph_search(
    problem: Problem[TState, TAction],
    f: Callable[[Node[TState, TAction]], float],
) -> Optional[Node[TState, TAction]]:
    """Best-first graph search (with explored-set) returning a goal node or None.

    Note: This is the minimal, uninstrumented version.
    Use `best_first_graph_search_metrics` for instrumentation.
    """
    node = Node(problem.initial)
    if problem.goal_test(node.state):
        return node

    frontier = PriorityQueue(f)
    frontier.push(node)
    explored: Set[Any] = set()

    while len(frontier):
        node = frontier.pop()
        if problem.goal_test(node.state):
            return node

        explored.add(node.state)

        for child in node.expand(problem):
            if child.state not in explored:
                frontier.push(child)

    return None


def best_first_graph_search_metrics(
    problem: Problem[TState, TAction],
    f: Callable[[Node[TState, TAction]], float],
) -> Tuple[Optional[Node[TState, TAction]], SearchMetrics]:
    """Instrumented best-first graph search.

    Returns (goal_node_or_None, metrics).
    """

    metrics = SearchMetrics()
    t0 = time.perf_counter()

    node = Node(problem.initial)
    if problem.goal_test(node.state):
        metrics.elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return node, metrics

    frontier = PriorityQueue(f)
    frontier.push(node)
    explored: Set[Any] = set()
    metrics.max_frontier_size = max(metrics.max_frontier_size, len(frontier))

    while len(frontier):
        metrics.max_frontier_size = max(metrics.max_frontier_size, len(frontier))
        node = frontier.pop()
        metrics.nodes_popped += 1

        if problem.goal_test(node.state):
            metrics.elapsed_ms = (time.perf_counter() - t0) * 1000.0
            return node, metrics

        explored.add(node.state)
        metrics.nodes_expanded += 1

        children = node.expand(problem)
        metrics.nodes_generated += len(children)

        for child in children:
            if child.state not in explored:
                frontier.push(child)

    metrics.elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return None, metrics


def uniform_cost_search(problem: Problem[TState, TAction]) -> Optional[Node[TState, TAction]]:
    """Uniform-cost search: f(n) = g(n)."""
    return best_first_graph_search(problem, f=lambda n: n.path_cost)


def uniform_cost_search_metrics(problem: Problem[TState, TAction]) -> Tuple[Optional[Node[TState, TAction]], SearchMetrics]:
    """Instrumented uniform-cost search."""
    return best_first_graph_search_metrics(problem, f=lambda n: n.path_cost)


def astar_search(
    problem: Problem[TState, TAction],
    h: Optional[Callable[[Node[TState, TAction]], float]] = None,
) -> Optional[Node[TState, TAction]]:
    """A* search: f(n) = g(n) + h(n)."""
    heuristic = h if h is not None else problem.h
    return best_first_graph_search(problem, f=lambda n: n.path_cost + float(heuristic(n)))


def astar_search_metrics(
    problem: Problem[TState, TAction],
    h: Optional[Callable[[Node[TState, TAction]], float]] = None,
) -> Tuple[Optional[Node[TState, TAction]], SearchMetrics]:
    """Instrumented A* search."""
    heuristic = h if h is not None else problem.h
    return best_first_graph_search_metrics(problem, f=lambda n: n.path_cost + float(heuristic(n)))
