# Iteration Progress Visualization Plan

## Goal

Create a visualization system that shows how the iterative feature engineering agent improves over time:
1. **Plot**: X-axis = iterations, Y-axis = RMSLE metric
2. **Table**: Iteration | Feature Summary | Metric

---

## Current Data Structure Gap

The current `iteration_history` in memory.json stores:
```json
{
  "iteration": 1,
  "rmsle": 0.10353,
  "success": true,
  "timestamp": "2024-..."
}
```

**Missing**: Feature summary/description for each iteration. Need to add this.

---

## Implementation Plan

### Step 1: Enhance Memory System

Update `memory.py` to store feature summary in each iteration:

```python
def log_iteration(self, iteration, rmsle, feature_code, success, feature_summary=None):
    self.memory['iteration_history'].append({
        'iteration': iteration,
        'rmsle': rmsle,
        'success': success,
        'feature_summary': feature_summary,  # NEW: e.g., "QualSF (OverallQual × GrLivArea)"
        'timestamp': datetime.now().isoformat()
    })
```

Also add helper to extract feature summary from code:
```python
def _extract_feature_summary(self, code: str, columns: list[str]) -> str:
    # Try to extract "# Feature: XXX" comment from code
    # Fall back to column names if no comment found
```

### Step 2: Create Visualizer Module

New file: `src/agent/visualizer.py`

```python
class ProgressVisualizer:
    def __init__(self, memory_path='outputs/logs/agent_memory.json'):
        self.memory = self._load_memory(memory_path)

    def plot_progress(self, save_path='outputs/logs/progress_plot.png'):
        """
        Create matplotlib plot showing:
        - X-axis: Iteration number (0 = baseline)
        - Y-axis: RMSLE
        - Horizontal dashed line: baseline RMSLE
        - Line plot: RMSLE over iterations
        - Green markers: successful iterations
        - Red markers: failed iterations
        """

    def generate_table(self) -> pd.DataFrame:
        """
        Create table with columns:
        - Iteration (0 = baseline)
        - Feature Summary
        - RMSLE
        - Delta (change from previous)
        - Cumulative Improvement %
        """

    def generate_report(self, output_path='outputs/logs/progress_report.html'):
        """
        Generate combined HTML report with:
        - Summary statistics
        - Progress plot (embedded)
        - Feature table
        """

    def show(self):
        """Display plot interactively (for Jupyter/terminal)"""
```

### Step 3: Update Iterative Agent

Add `--visualize` flag to `iterative_agent.py`:
```python
parser.add_argument('--visualize', action='store_true',
                    help='Generate visualization after completion')
```

### Step 4: Standalone Visualization Script

New file: `scripts/visualize_progress.py`

```bash
# Usage
python scripts/visualize_progress.py                    # Default: show plot
python scripts/visualize_progress.py --save             # Save to file
python scripts/visualize_progress.py --html             # Generate HTML report
python scripts/visualize_progress.py --table            # Print table to terminal
```

---

## Output Formats

1. **PNG image**: `outputs/logs/progress_plot.png`
2. **Terminal table**: Printed to console when running
3. **HTML report**: `outputs/logs/progress_report.html`

---

## Plot Design

```
RMSLE
  ^
  |  -------- baseline (0.10353) --------
  |     ○
  |        ●
  |           ○     ○
  |              ●
  |                    ●  ← best
  +---------------------------------> Iteration
      0   1   2   3   4   5

Legend:
  ● = Successful (improved)
  ○ = Failed (no improvement)
  --- = Baseline
```

**Visual elements:**
- Baseline: horizontal dashed gray line
- Actual RMSLE per iteration: connected line (shows real ups and downs)
- Success markers: green filled circles
- Failure markers: red hollow circles
- Best RMSLE annotation
- Title with summary stats

---

## Table Design

| Iter | Feature Summary | RMSLE | Delta | Cumul. % |
|------|-----------------|-------|-------|----------|
| 0 | (baseline) | 0.10353 | - | - |
| 1 | QualSF (Quality × Area) | 0.10320 | -0.00033 | -0.32% |
| 2 | GarageInteraction | 0.10350 | +0.00030 | -0.03% |
| 3 | BasementRatio | 0.10290 | -0.00060 | -0.61% |

---

## Files to Create/Modify

| File | Action | Purpose |
|------|--------|---------|
| `src/agent/memory.py` | Modify | Add feature_summary to log_iteration |
| `src/agent/visualizer.py` | Create | Plot + table generation |
| `src/agent/iterative_agent.py` | Modify | Add --visualize flag |
| `scripts/visualize_progress.py` | Create | Standalone visualization CLI |

---

## Success Criteria

- [ ] Plot shows iterations on X-axis, RMSLE on Y-axis
- [ ] Baseline shown as reference line
- [ ] Success/failure iterations visually distinguished
- [ ] Table shows iteration, feature summary, and metric
- [ ] Can run standalone or integrated with iterative_agent
- [ ] Output saved to outputs/logs/
