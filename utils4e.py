# utils4e.py (MINIMAL) — only what UCS / A* usually needs

from __future__ import annotations
import heapq
from functools import lru_cache
from typing import Any, Callable, Iterable, Optional

# AIMA uses this constant a lot
infinity = float("inf")


def memoize(fn: Callable, slot: Optional[str] = None, maxsize: int = 2048) -> Callable:
    """
    Cache the return value of `fn`.

    - If slot is provided, store result on the first argument object as `obj.<slot>`.
      (AIMA sometimes does this for heuristic caching per Node.)
    - Otherwise use an LRU cache.
    """
    if slot:
        def memoized(obj, *args):
            if hasattr(obj, slot):
                return getattr(obj, slot)
            val = fn(obj, *args)
            setattr(obj, slot, val)
            return val
        return memoized

    return lru_cache(maxsize=maxsize)(fn)


def is_in(x: Any, seq: Iterable[Any]) -> bool:
    """Identity-based membership (AIMA sometimes prefers this over `in`)."""
    return any(x is y for y in seq)


def argmin(seq: Iterable[Any], key: Callable[[Any], Any]) -> Any:
    """Return element with minimum key(seq[i])."""
    return min(seq, key=key)


def argmax(seq: Iterable[Any], key: Callable[[Any], Any]) -> Any:
    """Return element with maximum key(seq[i])."""
    return max(seq, key=key)


class PriorityQueue:
    """
    A simple priority queue (min-heap by default) used by UCS / A*.

    Supports:
    - append(item)
    - pop()
    - item in pq
    - pq[item] -> priority
    - del pq[item]
    """
    def __init__(self, order=min, f: Callable[[Any], Any] = lambda x: x):
        self.order = order
        self.f = f
        self.heap: list[tuple[Any, int, Any]] = []
        self.entry_finder: dict[Any, tuple[Any, int, Any]] = {}
        self.counter = 0

    def append(self, item: Any) -> None:
        # For max-queue, invert priority by negating when possible
        priority = self.f(item)
        if self.order is max:
            try:
                priority = -priority
            except Exception:
                # If it can't be negated, fallback to min behavior
                pass

        entry = (priority, self.counter, item)
        self.counter += 1
        self.entry_finder[item] = entry
        heapq.heappush(self.heap, entry)

    def pop(self) -> Any:
        while self.heap:
            priority, _, item = heapq.heappop(self.heap)
            # Skip stale entries
            if self.entry_finder.get(item) == (priority, _, item):
                del self.entry_finder[item]
                return item
        raise KeyError("pop from empty PriorityQueue")

    def __len__(self) -> int:
        return len(self.entry_finder)

    def __contains__(self, item: Any) -> bool:
        return item in self.entry_finder

    def __getitem__(self, item: Any) -> Any:
        return self.entry_finder[item][0]

    def __delitem__(self, item: Any) -> None:
        # Lazy deletion: remove from dict; heap entry becomes stale
        if item in self.entry_finder:
            del self.entry_finder[item]
        else:
            raise KeyError(item)
