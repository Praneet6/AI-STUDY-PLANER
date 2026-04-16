"""
astar.py — Core A* Search Algorithm for Study Path Generation
=============================================================

This module implements the A* Search Algorithm to find the optimal
ordering of study topics.

KEY CONCEPTS:
  f(n) = g(n) + h(n)

  g(n)  → Actual cumulative cost: total adjusted study time + cognitive penalties
           for topics already completed in this path.

  h(n)  → Heuristic estimate: sum of adjusted times for remaining (unvisited) topics.
           This is ADMISSIBLE because it never overestimates (ignores penalties).

  f(n)  → Total estimated cost from start → current → goal.

STATE SPACE:
  - Each state = frozenset of remaining topic indices (not yet studied)
  - Start state  = all topics remaining
  - Goal state   = empty frozenset (all topics studied)
  - Transition   = picking the next topic to study from the remaining set

OPTIMALITY:
  Because h(n) is admissible and consistent, A* guarantees an optimal solution.
"""

import heapq
from dataclasses import dataclass, field
from typing import List, FrozenSet, Optional, Tuple, Dict, Any


# ─── Data Classes ─────────────────────────────────────────────────────────────

@dataclass
class Topic:
    """
    Represents a single study topic (node in the graph).

    Attributes:
        name         : Display name of the topic
        difficulty   : 'Easy' | 'Medium' | 'Hard'
        base_time    : Estimated hours without any adjustments
        familiarity  : 0–100 (0 = never seen it, 100 = already mastered)
        is_priority  : If True, treated as high-importance → small cost discount
    """
    name: str
    difficulty: str       # 'Easy', 'Medium', 'Hard'
    base_time: float      # raw hours
    familiarity: int      # 0–100
    is_priority: bool = False

    # ── Derived properties ──────────────────────────────────────────────────

    @property
    def difficulty_weight(self) -> float:
        """
        Difficulty multiplier:
          Easy   → 1.0  (takes normal time)
          Medium → 1.5  (50% more effort)
          Hard   → 2.0  (double effort)
        """
        return {'Easy': 1.0, 'Medium': 1.5, 'Hard': 2.0}[self.difficulty]

    @property
    def familiarity_factor(self) -> float:
        """
        Converts familiarity (0–100) into a reduction factor (0.0–1.0).
        High familiarity → less time needed.
        """
        return self.familiarity / 100.0

    @property
    def adjusted_time(self) -> float:
        """
        Adjusted Time = base_time × difficulty_weight × (1 − familiarity_factor)

        Examples:
          base=4h, Hard, familiarity=0   → 4 × 2.0 × 1.0 = 8.0h
          base=4h, Hard, familiarity=75  → 4 × 2.0 × 0.25 = 2.0h
          base=2h, Easy, familiarity=50  → 2 × 1.0 × 0.50 = 1.0h
        """
        raw = self.base_time * self.difficulty_weight * (1.0 - self.familiarity_factor)
        return max(raw, 0.1)  # floor at 0.1h to avoid zero-cost topics

    @property
    def difficulty_num(self) -> int:
        """Numeric difficulty (1/2/3) used for penalty calculations."""
        return {'Easy': 1, 'Medium': 2, 'Hard': 3}[self.difficulty]


# ─── Search Node ──────────────────────────────────────────────────────────────

@dataclass(order=True)
class SearchNode:
    """
    A single node in the A* search tree.

    The @dataclass(order=True) makes Python compare nodes by field order,
    so heapq uses f as the primary sort key — the hallmark of A*.

    Fields:
        f         → f(n) = g(n) + h(n), drives priority queue ordering
        g         → cumulative actual cost so far
        h         → heuristic remaining estimate
        path      → list of topic indices in study order so far
        last_idx  → index of the last topic studied (for penalty calculation)
        remaining → frozenset of topic indices NOT yet studied
    """
    f: float
    g: float = field(compare=False)
    h: float = field(compare=False)
    path: List[int] = field(compare=False)
    last_idx: Optional[int] = field(compare=False)
    remaining: FrozenSet[int] = field(compare=False)


# ─── A* Planner ───────────────────────────────────────────────────────────────

class AStarStudyPlanner:
    """
    Implements the A* Search Algorithm to produce an optimal study sequence.

    Why A* and not greedy / simple sorting?
    ----------------------------------------
    Greedy chooses the LOCALLY best next topic (lowest immediate cost) but can
    get stuck in poor orderings. A* keeps track of ALL paths and selects the one
    with the best f(n) = actual_cost + estimated_remaining, guaranteeing the
    globally optimal sequence.

    The cognitive_penalty makes topic ORDER matter:
    studying Hard→Hard is penalised, so A* will naturally interleave difficulties.
    """

    # Maximum topics before switching to beam search (avoids exponential blow-up)
    EXACT_LIMIT = 14

    def __init__(self, topics: List[Topic]):
        self.topics = topics
        self.n = len(topics)

    # ── Private Helpers ────────────────────────────────────────────────────

    def _cognitive_penalty(self, last_idx: Optional[int], next_idx: int) -> float:
        """
        Transition penalty between topics — models cognitive load.

        Rules:
          Hard → Hard          : +20% of next topic's adjusted time
          Easy/Med → Hard skip : +10% (jumping two difficulty levels)
          Everything else      : 0%

        This is NOT included in h(n), keeping it admissible.
        """
        if last_idx is None:
            return 0.0

        last_d = self.topics[last_idx].difficulty_num
        next_d = self.topics[next_idx].difficulty_num
        next_t = self.topics[next_idx].adjusted_time

        if last_d == 3 and next_d == 3:
            return 0.20 * next_t   # consecutive hard topics
        elif next_d - last_d >= 2:
            return 0.10 * next_t   # big difficulty jump
        return 0.0

    def heuristic(self, remaining: FrozenSet[int]) -> float:
        """
        h(n): Admissible heuristic — sum of adjusted times for all remaining topics.

        Admissibility proof:
          The real remaining cost ≥ sum(adjusted_times) because cognitive penalties
          are always ≥ 0. So h(n) ≤ true remaining cost. ✓

        Consistency / monotonicity:
          h(state) ≤ edge_cost(state→next) + h(next_state)
          Since edge_cost includes the topic's own adjusted_time, and
          h(next) = h(current) − adjusted_time(chosen), the triangle inequality holds. ✓
        """
        return sum(self.topics[i].adjusted_time for i in remaining)

    # ── Main Solve ─────────────────────────────────────────────────────────

    def solve(self) -> Tuple[List[int], Dict[str, Any]]:
        """
        Execute A* search and return the optimal study order.

        Returns:
            path     : List of topic indices in optimal study order
            metadata : Stats about the search (cost, nodes expanded, etc.)
        """
        if self.n == 0:
            return [], {}

        # For large inputs, use beam search to keep runtime reasonable
        if self.n > self.EXACT_LIMIT:
            return self._beam_search(beam_width=50)

        return self._exact_astar()

    def _exact_astar(self) -> Tuple[List[int], Dict[str, Any]]:
        """Full A* search with closed-list deduplication."""
        all_indices = frozenset(range(self.n))
        h0 = self.heuristic(all_indices)

        start_node = SearchNode(
            f=h0, g=0.0, h=h0,
            path=[], last_idx=None,
            remaining=all_indices
        )

        heap: List[SearchNode] = [start_node]
        # closed list: (remaining, last_idx) → best g seen
        closed: Dict[Tuple, float] = {}

        nodes_expanded = 0
        nodes_generated = 1

        while heap:
            node = heapq.heappop(heap)
            nodes_expanded += 1

            state_key = (node.remaining, node.last_idx)
            if state_key in closed and closed[state_key] <= node.g:
                continue   # already found a better path to this state
            closed[state_key] = node.g

            # ── Goal Check ────────────────────────────────────────────────
            if not node.remaining:
                return node.path, {
                    'total_adjusted_time': node.g,
                    'nodes_expanded': nodes_expanded,
                    'nodes_generated': nodes_generated,
                    'algorithm': 'A* (exact)',
                }

            # ── Expand Node ───────────────────────────────────────────────
            for idx in node.remaining:
                topic = self.topics[idx]

                # g(n) update: adjusted time + cognitive penalty
                penalty = self._cognitive_penalty(node.last_idx, idx)

                # Priority topics get a 10% cost discount to bubble them forward
                priority_disc = 0.10 * topic.adjusted_time if topic.is_priority else 0.0

                child_g = node.g + topic.adjusted_time + penalty - priority_disc
                child_remaining = node.remaining - {idx}
                child_h = self.heuristic(child_remaining)
                child_f = child_g + child_h   # ← THE A* EVALUATION FUNCTION

                child_key = (child_remaining, idx)
                if child_key not in closed or closed[child_key] > child_g:
                    nodes_generated += 1
                    heapq.heappush(heap, SearchNode(
                        f=child_f,
                        g=child_g,
                        h=child_h,
                        path=node.path + [idx],
                        last_idx=idx,
                        remaining=child_remaining
                    ))

        # Fallback (should never reach here with correct input)
        return list(range(self.n)), {'algorithm': 'fallback', 'total_adjusted_time': 0}

    def _beam_search(self, beam_width: int = 50) -> Tuple[List[int], Dict[str, Any]]:
        """
        Beam-search approximation for > 14 topics.
        Keeps only the `beam_width` best partial paths at each depth level.
        Still uses f(n) = g(n) + h(n) as the ranking criterion.
        """
        all_indices = frozenset(range(self.n))
        h0 = self.heuristic(all_indices)

        beam = [SearchNode(f=h0, g=0.0, h=h0, path=[], last_idx=None, remaining=all_indices)]
        nodes_expanded = 0

        for _ in range(self.n):  # exactly n steps to assign all topics
            candidates = []
            for node in beam:
                nodes_expanded += 1
                for idx in node.remaining:
                    topic = self.topics[idx]
                    penalty = self._cognitive_penalty(node.last_idx, idx)
                    priority_disc = 0.10 * topic.adjusted_time if topic.is_priority else 0.0
                    child_g = node.g + topic.adjusted_time + penalty - priority_disc
                    child_remaining = node.remaining - {idx}
                    child_h = self.heuristic(child_remaining)
                    child_f = child_g + child_h
                    candidates.append(SearchNode(
                        f=child_f, g=child_g, h=child_h,
                        path=node.path + [idx],
                        last_idx=idx,
                        remaining=child_remaining
                    ))

            if not candidates:
                break

            # Keep only the best beam_width candidates
            candidates.sort()
            beam = candidates[:beam_width]

        best = beam[0] if beam else None
        if best:
            return best.path, {
                'total_adjusted_time': best.g,
                'nodes_expanded': nodes_expanded,
                'algorithm': f'Beam Search (width={beam_width}, A* heuristic)',
            }
        return list(range(self.n)), {}
