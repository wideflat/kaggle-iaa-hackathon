#!/usr/bin/env python
"""
Standalone visualization script for feature engineering progress

Usage:
    python scripts/visualize_progress.py                    # Show plot + print table
    python scripts/visualize_progress.py --save             # Save plot to file
    python scripts/visualize_progress.py --html             # Generate HTML report
    python scripts/visualize_progress.py --table            # Print table only
    python scripts/visualize_progress.py --all              # All outputs
"""

import argparse
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.agent.visualizer import ProgressVisualizer


def main():
    parser = argparse.ArgumentParser(
        description='Visualize feature engineering progress',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/visualize_progress.py              # Show plot interactively
  python scripts/visualize_progress.py --save       # Save plot to PNG
  python scripts/visualize_progress.py --html       # Generate HTML report
  python scripts/visualize_progress.py --all        # All outputs
        """
    )
    parser.add_argument('--memory', '-m', type=str,
                        default='outputs/logs/agent_memory.json',
                        help='Path to memory JSON file')
    parser.add_argument('--save', '-s', action='store_true',
                        help='Save plot to PNG file')
    parser.add_argument('--html', action='store_true',
                        help='Generate HTML report')
    parser.add_argument('--table', '-t', action='store_true',
                        help='Print table to console')
    parser.add_argument('--all', '-a', action='store_true',
                        help='Generate all outputs (plot, table, HTML)')
    parser.add_argument('--no-show', action='store_true',
                        help='Do not show plot interactively')
    parser.add_argument('--output-dir', '-o', type=str,
                        default='outputs/logs',
                        help='Output directory for files')

    args = parser.parse_args()

    # Check if memory file exists
    if not os.path.exists(args.memory):
        print(f"Error: Memory file not found: {args.memory}")
        print("Run the iterative agent first to generate data:")
        print("  python -m src.agent.iterative_agent --iterations 5")
        sys.exit(1)

    # Initialize visualizer
    try:
        viz = ProgressVisualizer(memory_path=args.memory)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)

    # Handle --all flag
    if args.all:
        args.save = True
        args.table = True
        args.html = True

    # Default behavior: show plot and table if no specific flags
    if not (args.save or args.table or args.html):
        args.table = True
        if not args.no_show:
            # Show plot interactively
            viz.plot_progress(save_path=None, show=True)

    # Generate outputs based on flags
    if args.save:
        save_path = os.path.join(args.output_dir, 'progress_plot.png')
        viz.plot_progress(save_path=save_path, show=not args.no_show)

    if args.table:
        viz.print_table()

    if args.html:
        html_path = os.path.join(args.output_dir, 'progress_report.html')
        viz.generate_report(output_path=html_path)

    print("\nDone!")


if __name__ == '__main__':
    main()
