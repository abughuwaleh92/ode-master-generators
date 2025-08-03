# utils/config.py
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import logging
import threading

logger = logging.getLogger(__name__)


class ConfigManager:
    """Centralised configuration management with freeze support"""

    # ───────────────────────────────────────────────────────────
    #  Constructor & freeze helpers
    # ───────────────────────────────────────────────────────────
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or "config.yaml"
        self.config = self._load_config()         # <- needs the method below
        self._frozen = False
        self._lock = threading.Lock()

    def freeze(self):
        """Prevent further mutations to the config object"""
        with self._lock:
            self._frozen = True
            logger.info("Configuration frozen")

    def unfreeze(self):
        """Allow mutations again (use with caution)"""
        with self._lock:
            self._frozen = False
            logger.warning("Configuration unfrozen")

    def _check_frozen(self):
        if self._frozen:
            raise RuntimeError("Cannot modify frozen configuration")

    def update(self, updates: Dict[str, Any]):
        """Merge a dict into the existing config (unless frozen)"""
        self._check_frozen()
        self._deep_merge(self.config, updates)

    # ───────────────────────────────────────────────────────────
    #  Internal helpers (restored)
    # ───────────────────────────────────────────────────────────
    def _get_default_config(self) -> Dict[str, Any]:
        """Return a full default configuration in case no file exists"""
        return {
            "generation": {
                "samples_per_combo": 5,
                "parameter_ranges": {
                    "alpha": [0, 0.5, 1, 1.5, 2],
                    "beta":  [0.5, 1, 1.5, 2],
                    "M":     [0, 0.5, 1],
                    "q":     [2, 3],
                    "v":     [2, 3, 4],
                    "a":     [2, 3, 4],
                },
            },
            "verification": {
                "numeric_test_points": [0.1, 0.3, 0.7, 1.0],
                "residual_tolerance": 1e-8,
                "max_derivative_order": 5,
                "verification_timeout": 30,
            },
            "performance": {
                "cache_size": 256,
                "n_workers": None,
                "streaming_enabled": True,
                "batch_size": 100,
                "enable_progress_bar": True,
            },
            "output": {
                "streaming_file": "ode_dataset.jsonl",
                "features_file":  "ode_features.parquet",
                "report_file":    "generation_report.json",
                "save_intermediate": True,
                "checkpoint_interval": 500,
            },
            "logging": {
                "level": "INFO",
                "file":  "ode_generation.log",
            },
        }

    def _load_config(self) -> Dict[str, Any]:
        """Load YAML config if it exists, otherwise fall back to defaults"""
        defaults = self._get_default_config()
        cfg_path = Path(self.config_path)

        if cfg_path.exists():
            try:
                with cfg_path.open("r", encoding="utf-8") as fh:
                    user_cfg = yaml.safe_load(fh) or {}
                merged = self._deep_merge(defaults, user_cfg)
                logger.info(f"Loaded configuration from {cfg_path}")
                return merged
            except Exception as err:
                logger.warning(
                    f"Error reading {cfg_path}: {err}. Falling back to defaults."
                )
                return defaults
        else:
            logger.info("No config file found – using built‑in defaults")
            return defaults

    def _deep_merge(self, base: Dict[str, Any], upd: Dict[str, Any]) -> Dict[str, Any]:
        """Recursive dict merge: modifies *base* in‑place and returns it"""
        for k, v in upd.items():
            if k in base and isinstance(base[k], dict) and isinstance(v, dict):
                self._deep_merge(base[k], v)
            else:
                base[k] = v
        return base

    # ───────────────────────────────────────────────────────────
    #  Public convenience helpers
    # ───────────────────────────────────────────────────────────
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Dot‑notation lookup, e.g.  cfg.get("generation.samples_per_combo")
        """
        node: Any = self.config
        for part in key_path.split("."):
            if isinstance(node, dict) and part in node:
                node = node[part]
            else:
                return default
        return node

    def save_config(self, path: Optional[str] = None):
        """Write current config back to disk"""
        self._check_frozen()
        save_to = Path(path or self.config_path)
        with save_to.open("w", encoding="utf-8") as fh:
            yaml.dump(self.config, fh, default_flow_style=False, sort_keys=False)
        logger.info(f"Configuration written to {save_to}")
