# Parallel Agent Execution

## Goal

Speed up the iterative feature engineering agent by running multiple workers in parallel, each independently generating and evaluating features. Update the dashboard to show all parallel workers' progress.

---

## Current Performance Bottleneck

Each iteration takes ~10-15 seconds:
- LLM call to Gemini: ~2-5 seconds
- Code execution: <1 second
- 5-fold CV evaluation: ~5-10 seconds (main bottleneck)
- SHAP computation: ~2-3 seconds (if feedback mode)

With 10 iterations = ~2+ minutes. Parallelism can dramatically reduce this.

---

## Architecture

### Worker-Based Parallelism

Run N independent workers, each:
1. Generates a feature (LLM call)
2. Executes the code
3. Evaluates via CV
4. Reports result to coordinator

### Shared State (Thread-Safe)
- `tried_features: Set[str]` - Prevent duplicate generation
- `accumulated_df: pd.DataFrame` - Current best feature set
- `best_rmsle: float` - Current best score
- `lock: threading.Lock` - Protect shared state updates

### Coordination
- Workers compete to improve RMSLE
- First worker to find improvement updates shared state
- Other workers continue with latest state
- Dashboard shows all workers' activity

---

## Implementation Plan

### 1. Create `src/agent/parallel_agent.py`

```python
# Key components:
class ParallelFeatureAgent:
    def __init__(self, n_workers: int = 3, ...):
        self.n_workers = n_workers
        self.shared_state = SharedState()

    def run(self, n_iterations: int):
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = [
                executor.submit(self._worker_loop, worker_id, iterations_per_worker)
                for worker_id in range(n_workers)
            ]
            # Wait for all workers

class SharedState:
    def __init__(self):
        self.lock = threading.RLock()
        self.tried_features: Set[str] = set()
        self.best_rmsle: float = None
        self.accumulated_df: pd.DataFrame = None

    def try_accept_feature(self, code, new_df, new_rmsle) -> bool:
        with self.lock:
            if new_rmsle < self.best_rmsle:
                self.best_rmsle = new_rmsle
                self.accumulated_df = new_df
                self.tried_features.add(code)
                return True
            self.tried_features.add(code)
            return False
```

### 2. Update Event Emitter

Add worker_id to events:
```python
emit('iteration_start', {
    'worker_id': worker_id,
    'iteration': iteration
})
emit('feature_accepted', {
    'worker_id': worker_id,
    ...
})
```

### 3. Update Dashboard UI

Show multiple workers:
```html
<!-- Worker status cards -->
<div class="workers-grid">
    <div class="worker-card" data-worker="0">
        <div class="worker-status">Evaluating...</div>
        <div class="worker-code">df['Area_ratio']...</div>
    </div>
    <!-- Repeat for each worker -->
</div>

<!-- Chart: show all workers' attempts with different markers -->
```

### 4. Add CLI Flag

```
python -m src.agent.parallel_agent -n 10 --workers 2 --dashboard
```

---

## Configuration

- **Default workers**: 2 (conservative for RAM usage)
- **Iteration strategy**: Shared work queue
  - Workers pull from an atomic counter
  - Load-balances automatically
  - Fastest completion time

---

## Files to Modify/Create

1. **Create**: `src/agent/parallel_agent.py` - Main parallel agent
2. **Create**: `src/agent/shared_state.py` - Thread-safe shared state
3. **Modify**: `src/agent/event_emitter.py` - Add worker_id support
4. **Modify**: `static/dashboard.html` - Multi-worker UI
5. **Modify**: `src/agent/dashboard_server.py` - Handle worker events

---

## Dashboard Changes

### Layout
- Add "Workers" section showing each worker's current status
- Color-code workers (Worker 1: orange, Worker 2: blue, Worker 3: purple)
- Chart shows all attempts with worker-specific markers

### Status Bar Updates
- "Active Workers: 3/3"
- "Total Iterations: 15" (across all workers)

### Worker Cards
```
┌─────────────────────────────────┐
│ Worker 1 ● Evaluating           │
│ df['TotalSF'] = df['1stFlrSF']  │
│ + df['2ndFlrSF'] + df['Total... │
└─────────────────────────────────┘
```
