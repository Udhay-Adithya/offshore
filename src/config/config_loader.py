"""
Configuration loader for Offshore.

Handles loading, merging, and validating YAML configuration files.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional, Union
import copy

import yaml


class ConfigLoader:
    """
    Configuration loader and manager.

    Handles loading YAML configs, merging multiple configs,
    and providing easy access to nested configuration values.

    Example:
        >>> loader = ConfigLoader()
        >>> config = loader.load("configs/data.yaml")
        >>> lookback = config.get("features.lookback", default=60)
    """

    def __init__(self, base_path: Optional[Union[str, Path]] = None):
        """
        Initialize the configuration loader.

        Args:
            base_path: Base path for resolving relative config paths.
                      Defaults to current working directory.
        """
        self.base_path = Path(base_path) if base_path else Path.cwd()
        self._configs: dict[str, dict[str, Any]] = {}

    def load(self, config_path: Union[str, Path]) -> dict[str, Any]:
        """
        Load a YAML configuration file.

        Args:
            config_path: Path to the YAML config file.

        Returns:
            Dictionary containing the configuration.

        Raises:
            FileNotFoundError: If config file doesn't exist.
            yaml.YAMLError: If config file is invalid YAML.
        """
        path = self._resolve_path(config_path)

        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path, "r") as f:
            config = yaml.safe_load(f)

        if config is None:
            config = {}

        # Cache the loaded config
        self._configs[str(path)] = config

        return config

    def load_all(self, *config_paths: Union[str, Path]) -> dict[str, Any]:
        """
        Load and merge multiple configuration files.

        Later configs override earlier ones (deep merge).

        Args:
            *config_paths: Paths to YAML config files.

        Returns:
            Merged configuration dictionary.
        """
        merged: dict[str, Any] = {}

        for config_path in config_paths:
            config = self.load(config_path)
            merged = merge_configs(merged, config)

        return merged

    def _resolve_path(self, config_path: Union[str, Path]) -> Path:
        """Resolve a config path relative to base path if not absolute."""
        path = Path(config_path)
        if not path.is_absolute():
            path = self.base_path / path
        return path

    @staticmethod
    def get_nested(
        config: dict[str, Any], key: str, default: Any = None, separator: str = "."
    ) -> Any:
        """
        Get a nested configuration value using dot notation.

        Args:
            config: Configuration dictionary.
            key: Dot-separated key path (e.g., "features.lookback").
            default: Default value if key not found.
            separator: Key path separator (default ".").

        Returns:
            The configuration value or default.

        Example:
            >>> config = {"features": {"lookback": 60}}
            >>> ConfigLoader.get_nested(config, "features.lookback")
            60
        """
        keys = key.split(separator)
        value = config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value


def load_config(config_path: Union[str, Path]) -> dict[str, Any]:
    """
    Load a single YAML configuration file.

    Convenience function that creates a ConfigLoader and loads a file.

    Args:
        config_path: Path to the YAML config file.

    Returns:
        Dictionary containing the configuration.
    """
    loader = ConfigLoader()
    return loader.load(config_path)


def merge_configs(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """
    Deep merge two configuration dictionaries.

    Values in `override` take precedence over values in `base`.
    Nested dictionaries are merged recursively.

    Args:
        base: Base configuration dictionary.
        override: Override configuration dictionary.

    Returns:
        New merged configuration dictionary.

    Example:
        >>> base = {"a": 1, "b": {"c": 2, "d": 3}}
        >>> override = {"b": {"c": 10}, "e": 5}
        >>> merge_configs(base, override)
        {'a': 1, 'b': {'c': 10, 'd': 3}, 'e': 5}
    """
    result = copy.deepcopy(base)

    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = copy.deepcopy(value)

    return result


def get_config_value(config: dict[str, Any], key: str, default: Any = None) -> Any:
    """
    Get a nested configuration value using dot notation.

    Args:
        config: Configuration dictionary.
        key: Dot-separated key path.
        default: Default value if not found.

    Returns:
        The configuration value or default.
    """
    return ConfigLoader.get_nested(config, key, default)


def save_config(config: dict[str, Any], path: Union[str, Path]) -> None:
    """
    Save a configuration dictionary to a YAML file.

    Args:
        config: Configuration dictionary to save.
        path: Path to save the YAML file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def validate_config(config: dict[str, Any], schema: dict[str, Any]) -> list[str]:
    """
    Validate a configuration against a schema.

    Simple validation checking for required keys and types.

    Args:
        config: Configuration to validate.
        schema: Schema defining required keys and types.

    Returns:
        List of validation error messages (empty if valid).
    """
    errors: list[str] = []

    def _validate(cfg: dict[str, Any], sch: dict[str, Any], path: str = "") -> None:
        for key, requirements in sch.items():
            current_path = f"{path}.{key}" if path else key

            if isinstance(requirements, dict):
                if "required" in requirements and requirements["required"]:
                    if key not in cfg:
                        errors.append(f"Missing required key: {current_path}")
                        continue

                if "type" in requirements:
                    expected_type = requirements["type"]
                    if key in cfg and not isinstance(cfg[key], expected_type):
                        errors.append(
                            f"Invalid type for {current_path}: "
                            f"expected {expected_type.__name__}, "
                            f"got {type(cfg[key]).__name__}"
                        )

                if "nested" in requirements and key in cfg:
                    _validate(cfg[key], requirements["nested"], current_path)
            elif isinstance(requirements, type):
                if key in cfg and not isinstance(cfg[key], requirements):
                    errors.append(
                        f"Invalid type for {current_path}: "
                        f"expected {requirements.__name__}, "
                        f"got {type(cfg[key]).__name__}"
                    )

    _validate(config, schema)
    return errors
