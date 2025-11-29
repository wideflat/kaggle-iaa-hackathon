"""
Progress visualization for the feature engineering agent
"""

import json
import os
from datetime import datetime
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd


class ProgressVisualizer:
    """
    Visualize the progress of iterative feature engineering

    Generates:
    - Progress plot (iterations vs RMSLE)
    - Summary table
    - HTML report
    """

    def __init__(self, memory_path: str = 'outputs/logs/agent_memory.json'):
        """
        Initialize visualizer

        Args:
            memory_path: Path to agent memory JSON file
        """
        self.memory_path = memory_path
        self.memory = self._load_memory()

    def _load_memory(self) -> dict:
        """Load memory from disk"""
        if not os.path.exists(self.memory_path):
            raise FileNotFoundError(f"Memory file not found: {self.memory_path}")

        with open(self.memory_path, 'r') as f:
            return json.load(f)

    def plot_progress(
        self,
        save_path: Optional[str] = 'outputs/logs/progress_plot.png',
        show: bool = False,
        figsize: tuple = (12, 6)
    ) -> plt.Figure:
        """
        Create progress plot showing RMSLE over iterations

        Args:
            save_path: Path to save PNG (None to skip saving)
            show: Whether to display the plot interactively
            figsize: Figure size (width, height)

        Returns:
            matplotlib Figure object
        """
        history = self.memory.get('iteration_history', [])
        baseline = self.memory.get('baseline_rmsle')
        best = self.memory.get('best_rmsle')

        if not history:
            print("No iteration history to plot")
            return None

        # Extract data
        iterations = [0] + [h['iteration'] for h in history]
        rmsles = [baseline] + [h['rmsle'] for h in history]
        successes = [True] + [h['success'] for h in history]

        # Create figure
        fig, ax = plt.subplots(figsize=figsize)

        # Plot baseline as horizontal line
        ax.axhline(y=baseline, color='gray', linestyle='--', linewidth=1.5,
                   label=f'Baseline ({baseline:.5f})')

        # Plot RMSLE line
        ax.plot(iterations, rmsles, 'b-', linewidth=1.5, alpha=0.7)

        # Plot markers for each iteration
        for i, (it, rmsle, success) in enumerate(zip(iterations, rmsles, successes)):
            if i == 0:
                # Baseline marker
                ax.scatter(it, rmsle, c='blue', s=100, marker='s', zorder=5,
                          label='Baseline')
            elif success:
                ax.scatter(it, rmsle, c='green', s=80, marker='o', zorder=5)
            else:
                ax.scatter(it, rmsle, c='red', s=80, marker='o', facecolors='none',
                          linewidths=2, zorder=5)

        # Add legend markers for success/failure
        ax.scatter([], [], c='green', s=80, marker='o', label='Improved')
        ax.scatter([], [], c='red', s=80, marker='o', facecolors='none',
                  linewidths=2, label='No improvement')

        # Annotate best RMSLE
        if best and best < baseline:
            best_idx = rmsles.index(min(rmsles))
            ax.annotate(f'Best: {best:.5f}',
                       xy=(iterations[best_idx], rmsles[best_idx]),
                       xytext=(10, -20), textcoords='offset points',
                       fontsize=10, color='green',
                       arrowprops=dict(arrowstyle='->', color='green'))

        # Labels and title
        ax.set_xlabel('Iteration', fontsize=12)
        ax.set_ylabel('RMSLE', fontsize=12)

        improvement = baseline - best if best else 0
        improvement_pct = (improvement / baseline * 100) if baseline else 0
        ax.set_title(
            f'Feature Engineering Progress\n'
            f'Baseline: {baseline:.5f} → Best: {best:.5f} '
            f'(Improvement: {improvement_pct:.2f}%)',
            fontsize=14
        )

        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

        # Set x-axis to show integer iterations
        ax.set_xticks(iterations)

        plt.tight_layout()

        # Save if path provided
        if save_path:
            # Ensure directory exists
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Plot saved to: {save_path}")

        if show:
            plt.show()

        return fig

    def generate_table(self) -> pd.DataFrame:
        """
        Generate summary table of all iterations

        Returns:
            DataFrame with columns: Iteration, Feature Summary, RMSLE, Delta, Cumul. %
        """
        history = self.memory.get('iteration_history', [])
        baseline = self.memory.get('baseline_rmsle')

        rows = []

        # Add baseline row
        rows.append({
            'Iteration': 0,
            'Feature Summary': '(baseline)',
            'RMSLE': baseline,
            'Delta': None,
            'Cumul. %': None,
            'Status': '-'
        })

        # Add iteration rows
        prev_rmsle = baseline
        for h in history:
            delta = prev_rmsle - h['rmsle'] if prev_rmsle else None
            cumul = ((baseline - h['rmsle']) / baseline * 100) if baseline else None

            rows.append({
                'Iteration': h['iteration'],
                'Feature Summary': h.get('feature_summary', '(unknown)'),
                'RMSLE': h['rmsle'],
                'Delta': delta,
                'Cumul. %': cumul,
                'Status': 'Improved' if h['success'] else 'Failed'
            })

            # Only update prev_rmsle if it was an improvement
            if h['success']:
                prev_rmsle = h['rmsle']

        df = pd.DataFrame(rows)
        return df

    def print_table(self):
        """Print the summary table to console"""
        df = self.generate_table()

        print("\n" + "=" * 90)
        print("ITERATION PROGRESS TABLE")
        print("=" * 90)

        # Format for display
        pd.set_option('display.max_colwidth', 40)
        pd.set_option('display.width', 120)

        # Format numeric columns
        df_display = df.copy()
        df_display['RMSLE'] = df_display['RMSLE'].apply(lambda x: f'{x:.5f}' if x else '-')
        df_display['Delta'] = df_display['Delta'].apply(
            lambda x: f'{x:+.5f}' if x is not None else '-'
        )
        df_display['Cumul. %'] = df_display['Cumul. %'].apply(
            lambda x: f'{x:+.2f}%' if x is not None else '-'
        )

        print(df_display.to_string(index=False))
        print("=" * 90)

    def generate_report(
        self,
        output_path: str = 'outputs/logs/progress_report.html',
        plot_path: str = 'progress_plot.png'
    ):
        """
        Generate HTML report with plot and table

        Args:
            output_path: Path to save HTML report
            plot_path: Relative path to plot image (from report location)
        """
        # Generate plot first
        plot_dir = os.path.dirname(output_path)
        plot_full_path = os.path.join(plot_dir, plot_path)
        self.plot_progress(save_path=plot_full_path, show=False)

        # Get table data
        df = self.generate_table()
        stats = self._get_stats()

        # Generate HTML
        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Feature Engineering Progress Report</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }}
        h1 {{
            color: #333;
            border-bottom: 2px solid #4CAF50;
            padding-bottom: 10px;
        }}
        .stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }}
        .stat-card {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .stat-card h3 {{
            margin: 0 0 10px 0;
            color: #666;
            font-size: 14px;
        }}
        .stat-card .value {{
            font-size: 28px;
            font-weight: bold;
            color: #333;
        }}
        .stat-card .value.improved {{
            color: #4CAF50;
        }}
        .plot-container {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin: 20px 0;
        }}
        .plot-container img {{
            width: 100%;
            height: auto;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            background: white;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        th, td {{
            padding: 12px 15px;
            text-align: left;
            border-bottom: 1px solid #eee;
        }}
        th {{
            background: #4CAF50;
            color: white;
            font-weight: 600;
        }}
        tr:hover {{
            background: #f9f9f9;
        }}
        .status-improved {{
            color: #4CAF50;
            font-weight: bold;
        }}
        .status-failed {{
            color: #f44336;
        }}
        .timestamp {{
            color: #999;
            font-size: 12px;
            margin-top: 20px;
        }}
    </style>
</head>
<body>
    <h1>Feature Engineering Progress Report</h1>

    <div class="stats">
        <div class="stat-card">
            <h3>Baseline RMSLE</h3>
            <div class="value">{stats['baseline']:.5f}</div>
        </div>
        <div class="stat-card">
            <h3>Best RMSLE</h3>
            <div class="value improved">{stats['best']:.5f}</div>
        </div>
        <div class="stat-card">
            <h3>Improvement</h3>
            <div class="value improved">{stats['improvement_pct']:.2f}%</div>
        </div>
        <div class="stat-card">
            <h3>Success Rate</h3>
            <div class="value">{stats['success_rate']:.1f}%</div>
        </div>
    </div>

    <div class="plot-container">
        <h2>Progress Over Iterations</h2>
        <img src="{plot_path}" alt="Progress Plot">
    </div>

    <h2>Iteration Details</h2>
    <table>
        <thead>
            <tr>
                <th>Iter</th>
                <th>Feature Summary</th>
                <th>RMSLE</th>
                <th>Delta</th>
                <th>Cumul. %</th>
                <th>Status</th>
            </tr>
        </thead>
        <tbody>
"""
        # Add table rows
        for _, row in df.iterrows():
            status_class = 'status-improved' if row['Status'] == 'Improved' else 'status-failed' if row['Status'] == 'Failed' else ''
            delta_str = f"{row['Delta']:+.5f}" if row['Delta'] is not None else '-'
            cumul_str = f"{row['Cumul. %']:+.2f}%" if row['Cumul. %'] is not None else '-'

            html += f"""            <tr>
                <td>{row['Iteration']}</td>
                <td>{row['Feature Summary'][:50]}</td>
                <td>{row['RMSLE']:.5f}</td>
                <td>{delta_str}</td>
                <td>{cumul_str}</td>
                <td class="{status_class}">{row['Status']}</td>
            </tr>
"""

        html += f"""        </tbody>
    </table>

    <p class="timestamp">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
</body>
</html>
"""

        # Save HTML
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(html)

        print(f"HTML report saved to: {output_path}")

    def _get_stats(self) -> dict:
        """Get summary statistics"""
        history = self.memory.get('iteration_history', [])
        baseline = self.memory.get('baseline_rmsle', 0)
        best = self.memory.get('best_rmsle', baseline)

        successes = sum(1 for h in history if h['success'])
        total = len(history)

        return {
            'baseline': baseline,
            'best': best,
            'improvement': baseline - best,
            'improvement_pct': ((baseline - best) / baseline * 100) if baseline else 0,
            'total_iterations': total,
            'successes': successes,
            'failures': total - successes,
            'success_rate': (successes / total * 100) if total else 0
        }
