"""
Logging configuration for the Feature Engineering Agent

Provides:
- Console output with colors
- File logging with timestamps
- Configurable log levels
"""

import logging
import os
import sys
from datetime import datetime
from typing import Optional


class ColorFormatter(logging.Formatter):
    """Custom formatter with colors for console output"""

    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[35m',  # Magenta
        'RESET': '\033[0m',      # Reset
    }

    def format(self, record):
        color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        reset = self.COLORS['RESET']

        # Add color to level name
        record.levelname = f"{color}{record.levelname}{reset}"

        return super().format(record)


def setup_logger(
    name: str = 'agent',
    log_level: str = 'INFO',
    log_file: Optional[str] = None,
    console: bool = True
) -> logging.Logger:
    """
    Set up logger with console and file handlers

    Args:
        name: Logger name
        log_level: Log level (DEBUG, INFO, WARNING, ERROR)
        log_file: Path to log file (None to disable file logging)
        console: Whether to log to console

    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    # Remove existing handlers
    logger.handlers = []

    # Console handler with colors
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.DEBUG)
        console_format = ColorFormatter(
            '%(levelname)s %(message)s'
        )
        console_handler.setFormatter(console_format)
        logger.addHandler(console_handler)

    # File handler (no colors)
    if log_file:
        # Ensure directory exists
        os.makedirs(os.path.dirname(log_file), exist_ok=True)

        file_handler = logging.FileHandler(log_file, mode='a')
        file_handler.setLevel(logging.DEBUG)
        file_format = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_format)

        # Add filter to strip ANSI color codes from file output
        class StripColorFilter(logging.Filter):
            import re
            ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
            def filter(self, record):
                if hasattr(record, 'msg'):
                    record.msg = self.ansi_escape.sub('', str(record.msg))
                return True

        file_handler.addFilter(StripColorFilter())
        logger.addHandler(file_handler)

        # Log session start
        logger.info(f"{'='*60}")
        logger.info(f"New session started at {datetime.now().isoformat()}")
        logger.info(f"{'='*60}")

    return logger


def get_logger(name: str = 'agent') -> logging.Logger:
    """Get existing logger by name"""
    return logging.getLogger(name)


class AgentLogger:
    """
    High-level logger wrapper for the agent

    Provides semantic logging methods for agent operations
    """

    def __init__(self, config: Optional[dict] = None):
        """
        Initialize agent logger

        Args:
            config: Logging configuration dict with keys:
                - level: Log level (default: INFO)
                - file: Log file path (default: None)
                - console: Log to console (default: True)
        """
        config = config or {}
        self.logger = setup_logger(
            name='agent',
            log_level=config.get('level', 'INFO'),
            log_file=config.get('log_path') if config.get('file', True) else None,
            console=config.get('console', True)
        )

    def iteration_start(self, iteration: int, total: int, use_feedback: bool = False):
        """Log start of an iteration"""
        feedback_str = " [SHAP Feedback]" if use_feedback else ""
        self.logger.info(f"")
        self.logger.info(f"{'='*60}")
        self.logger.info(f"ITERATION {iteration}/{total}{feedback_str}")
        self.logger.info(f"{'='*60}")

    def iteration_result(
        self,
        iteration: int,
        feature_name: str,
        old_rmsle: float,
        new_rmsle: float,
        success: bool
    ):
        """Log result of an iteration"""
        delta = old_rmsle - new_rmsle
        if success:
            self.logger.info(f"✓ IMPROVED: {feature_name}")
            self.logger.info(f"  RMSLE: {old_rmsle:.5f} → {new_rmsle:.5f} (Δ {delta:+.5f})")
        else:
            self.logger.warning(f"✗ No improvement: {feature_name}")
            self.logger.warning(f"  RMSLE: {old_rmsle:.5f} → {new_rmsle:.5f} (Δ {delta:+.5f})")

    def shap_summary(self, top_features: list, top_feature: str):
        """Log SHAP analysis summary"""
        self.logger.debug(f"SHAP top feature: {top_feature}")
        self.logger.debug(f"SHAP top 5: {', '.join(top_features[:5])}")

    def generated_code(self, code: str, max_lines: int = 5):
        """Log generated code"""
        lines = code.strip().split('\n')
        self.logger.info(f"Generated code:")
        for line in lines[:max_lines]:
            self.logger.info(f"  {line}")
        if len(lines) > max_lines:
            self.logger.info(f"  ...")

    def execution_error(self, error: str):
        """Log code execution error"""
        self.logger.error(f"Execution failed: {error[:100]}...")

    def gemini_error(self, error: str):
        """Log Gemini API error"""
        self.logger.error(f"Gemini API error: {error}")

    def session_summary(
        self,
        iterations: int,
        successes: int,
        baseline_rmsle: float,
        best_rmsle: float
    ):
        """Log end of session summary"""
        improvement = baseline_rmsle - best_rmsle
        improvement_pct = (improvement / baseline_rmsle) * 100 if baseline_rmsle else 0

        self.logger.info(f"")
        self.logger.info(f"{'='*60}")
        self.logger.info(f"SESSION COMPLETE")
        self.logger.info(f"{'='*60}")
        self.logger.info(f"Iterations:    {iterations}")
        self.logger.info(f"Successful:    {successes} ({successes/iterations*100:.1f}%)")
        self.logger.info(f"Baseline:      {baseline_rmsle:.5f}")
        self.logger.info(f"Best:          {best_rmsle:.5f}")
        self.logger.info(f"Improvement:   {improvement:.5f} ({improvement_pct:.2f}%)")
        self.logger.info(f"{'='*60}")

    def info(self, msg: str):
        """Log info message"""
        self.logger.info(msg)

    def debug(self, msg: str):
        """Log debug message"""
        self.logger.debug(msg)

    def warning(self, msg: str):
        """Log warning message"""
        self.logger.warning(msg)

    def error(self, msg: str):
        """Log error message"""
        self.logger.error(msg)
