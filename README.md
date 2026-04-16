# 🧠 Smart Study Path Generator — A* Algorithm

> **AI-powered study planning using the A\* Search Algorithm.**  
> Finds the optimal order to study your topics, minimising total effort and cognitive load.

---

## 📐 How It Works

This project models study planning as an **AI search problem**:

| Concept | Mapping |
|---|---|
| **Node** | A study topic |
| **State** | Set of completed topics |
| **Initial state** | No topics studied |
| **Goal state** | All topics studied |
| **Transition** | Completing one topic |

The **A\* algorithm** is applied to find the ordering that minimises total adjusted cost:

```
f(n) = g(n) + h(n)
```

### g(n) — Actual Cost
Cumulative adjusted study time for topics completed so far:

```
Adjusted Time = base_time × difficulty_weight × (1 - familiarity_factor)
```

| Difficulty | Weight |
|---|---|
| Easy | 1.0 |
| Medium | 1.5 |
| Hard | 2.0 |

Plus **cognitive load penalties** for topic transitions:
- Hard → Hard consecutive: **+20%**
- Easy/Med → Hard (big jump): **+10%**

### h(n) — Heuristic
Sum of adjusted times for all **remaining** (unvisited) topics.

**Admissibility:** h(n) never overestimates because it ignores penalties, which are always ≥ 0.  
**Consistency:** Satisfies the triangle inequality, guaranteeing optimality.

---

## 🚀 Quick Start

### Requirements
```bash
pip install streamlit plotly pandas
```

### Run
```bash
cd study_planner
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

## 📁 Project Structure

```
study_planner/
├── astar.py      # A* algorithm, Topic and SearchNode dataclasses
├── utils.py      # Day scheduling, stats, export helpers
├── app.py        # Streamlit UI
└── README.md     # This file
```

---

## 🧪 Sample Test Input

| Topic | Difficulty | Hours | Familiarity |
|---|---|---|---|
| Python Basics | Easy | 2h | 70% |
| Data Structures | Medium | 4h | 40% |
| Graph Theory | Hard | 5h | 10% ⭐ |
| Dynamic Programming | Hard | 6h | 5% |
| Machine Learning 101 | Medium | 4h | 20% |

**Constraints:** 4h/day, 14-day deadline

### Expected Output (order may vary)
A* will prefer easy topics first to build familiarity, then interleave medium topics before tackling back-to-back hard ones — avoiding the 20% consecutive-hard penalty.

---

## 🔑 Key Design Decisions

1. **Why A\* not sorting?**  
   Simple sorting (e.g. by difficulty) is greedy and ignores transition costs. A\* considers the full path cost, avoiding, e.g., putting two Hard topics back-to-back unnecessarily.

2. **Admissible heuristic proof:**  
   True remaining cost = Σ(adjusted_times) + penalties. Since penalties ≥ 0, h = Σ(adjusted_times) ≤ true cost.

3. **Scalability:**  
   Exact A\* is used for ≤ 14 topics (state space ≤ 2^14 = 16,384).  
   For larger inputs, beam search (width=50) with the same A\* heuristic is used.

---

## ✅ Extra Features

- ⭐ Priority topics (10% cost discount to bubble them forward)
- 📅 Day-wise schedule with configurable hours/day
- 📊 Interactive charts (effort breakdown, cumulative progress, distribution)
- 💾 Export plan as `.txt` or `.csv`
- 🔄 Re-generate plan after editing topics
- 🗂 Sample data loader for instant demo
