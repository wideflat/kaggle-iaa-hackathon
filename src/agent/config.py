"""
Configuration management for the Feature Engineering Agent

Supports:
- YAML config files
- Environment variable overrides
- CLI argument overrides
"""

import os
from typing import Any, Optional

import yaml


# Default configuration
DEFAULT_CONFIG = {
    'agent': {
        'iterations': 10,
        'use_feedback': True,
    },
    'gemini': {
        'model': 'gemini-2.5-flash',
        'strategy': None,
    },
    'evaluation': {
        'n_folds': 5,
        'params_path': 'outputs/models/best_lgbm_params.json',
    },
    'output': {
        'memory_path': 'outputs/logs/agent_memory.json',
        'log_path': 'outputs/logs/agent.log',
        'plot_path': 'outputs/logs/progress_plot.png',
        'report_path': 'outputs/logs/progress_report.html',
    },
    'logging': {
        'level': 'INFO',
        'console': True,
        'file': True,
    },
    'visualization': {
        'auto_generate': False,
        'show_plot': False,
    },
}


class Config:
    """
    Configuration manager for the agent

    Loads config from:
    1. Default values
    2. YAML config file (if specified)
    3. Environment variables (AGENT_*)
    4. CLI arguments (highest priority)
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration

        Args:
            config_path: Path to YAML config file (optional)
        """
        self._config = self._deep_copy(DEFAULT_CONFIG)

        # Load from file if specified
        if config_path:
            self._load_from_file(config_path)
        else:
            # Try default locations
            for path in ['config/local.yaml', 'config/default.yaml']:
                if os.path.exists(path):
                    self._load_from_file(path)
                    break

        # Load from environment variables
        self._load_from_env()

    def _deep_copy(self, d: dict) -> dict:
        """Deep copy a dictionary"""
        result = {}
        for k, v in d.items():
            if isinstance(v, dict):
                result[k] = self._deep_copy(v)
            else:
                result[k] = v
        return result

    def _load_from_file(self, path: str):
        """Load configuration from YAML file"""
        if not os.path.exists(path):
            print(f"   Config file not found: {path}")
            return

        try:
            with open(path, 'r') as f:
                file_config = yaml.safe_load(f) or {}
            self._merge(self._config, file_config)
            print(f"   Loaded config from: {path}")
        except yaml.YAMLError as e:
            print(f"   Error parsing config file: {e}")

    def _load_from_env(self):
        """Load configuration from environment variables"""
        env_mappings = {
            'AGENT_ITERATIONS': ('agent', 'iterations', int),
            'AGENT_FEEDBACK': ('agent', 'use_feedback', lambda x: x.lower() == 'true'),
            'GEMINI_MODEL': ('gemini', 'model', str),
            'AGENT_LOG_LEVEL': ('logging', 'level', str),
        }

        for env_var, (section, key, converter) in env_mappings.items():
            value = os.getenv(env_var)
            if value is not None:
                try:
                    self._config[section][key] = converter(value)
                except (ValueError, KeyError):
                    pass

    def _merge(self, base: dict, override: dict):
        """Recursively merge override into base"""
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._merge(base[key], value)
            else:
                base[key] = value

    def get(self, *keys: str, default: Any = None) -> Any:
        """
        Get a config value by key path

        Args:
            *keys: Key path (e.g., 'agent', 'iterations')
            default: Default value if not found

        Returns:
            Config value
        """
        value = self._config
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        return value

    def set(self, *keys_and_value):
        """
        Set a config value by key path

        Args:
            *keys_and_value: Key path followed by value
                e.g., set('agent', 'iterations', 20)
        """
        keys = keys_and_value[:-1]
        value = keys_and_value[-1]

        target = self._config
        for key in keys[:-1]:
            if key not in target:
                target[key] = {}
            target = target[key]
        target[keys[-1]] = value

    def override_from_args(self, args):
        """
        Override config from argparse args

        Args:
            args: argparse Namespace object
        """
        # Map CLI args to config paths
        if hasattr(args, 'iterations') and args.iterations is not None:
            self.set('agent', 'iterations', args.iterations)
        if hasattr(args, 'feedback') and args.feedback:
            self.set('agent', 'use_feedback', True)
        if hasattr(args, 'no_feedback') and args.no_feedback:
            self.set('agent', 'use_feedback', False)
        if hasattr(args, 'model') and args.model:
            self.set('gemini', 'model', args.model)
        if hasattr(args, 'log_level') and args.log_level:
            self.set('logging', 'level', args.log_level.upper())
        if hasattr(args, 'output_dir') and args.output_dir:
            self.set('output', 'memory_path', f"{args.output_dir}/agent_memory.json")
            self.set('output', 'log_path', f"{args.output_dir}/agent.log")
            self.set('output', 'plot_path', f"{args.output_dir}/progress_plot.png")
            self.set('output', 'report_path', f"{args.output_dir}/progress_report.html")
        if hasattr(args, 'visualize') and args.visualize:
            self.set('visualization', 'auto_generate', True)

    @property
    def agent(self) -> dict:
        """Agent configuration"""
        return self._config['agent']

    @property
    def gemini(self) -> dict:
        """Gemini configuration"""
        return self._config['gemini']

    @property
    def evaluation(self) -> dict:
        """Evaluation configuration"""
        return self._config['evaluation']

    @property
    def output(self) -> dict:
        """Output paths configuration"""
        return self._config['output']

    @property
    def logging(self) -> dict:
        """Logging configuration"""
        return self._config['logging']

    @property
    def visualization(self) -> dict:
        """Visualization configuration"""
        return self._config['visualization']

    def to_dict(self) -> dict:
        """Return full config as dictionary"""
        return self._deep_copy(self._config)

    def print_config(self):
        """Print current configuration"""
        print("\n" + "=" * 50)
        print("CONFIGURATION")
        print("=" * 50)
        print(yaml.dump(self._config, default_flow_style=False))
        print("=" * 50)
