"""Configuration management for the traffic detection system."""

import json
import os
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class LaneConfig:
    """Configuration for a single detection lane."""
    name: str
    line_a: List[float]  # [x1, y1, x2, y2] normalized 0-1
    line_b: List[float]  # [x1, y1, x2, y2] normalized 0-1
    distance: float = 5.0  # meters between lines
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "line_a": self.line_a,
            "line_b": self.line_b,
            "distance": self.distance
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LaneConfig":
        return cls(
            name=data.get("name", "Lane"),
            line_a=data.get("line_a", [0.3, 0.4, 0.7, 0.4]),
            line_b=data.get("line_b", [0.3, 0.6, 0.7, 0.6]),
            distance=data.get("distance", 5.0)
        )


@dataclass
class AppConfig:
    """Application configuration."""
    lanes: List[LaneConfig] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "lanes": [lane.to_dict() for lane in self.lanes]
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AppConfig":
        lanes = [LaneConfig.from_dict(lane) for lane in data.get("lanes", [])]
        return cls(lanes=lanes)


class ConfigManager:
    """Manages application configuration with persistence."""
    
    def __init__(self, config_path: str):
        self.config_path = config_path
        self._config: Optional[AppConfig] = None
    
    @property
    def config(self) -> AppConfig:
        """Get current configuration, loading from disk if needed."""
        if self._config is None:
            self._config = self._load()
        return self._config
    
    def _load(self) -> AppConfig:
        """Load configuration from disk."""
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, "r") as f:
                    data = json.load(f)
                logger.info(f"Loaded config from {self.config_path}")
                return AppConfig.from_dict(data)
            except Exception as e:
                logger.error(f"Failed to load config: {e}")
        return AppConfig()
    
    def save(self) -> None:
        """Save current configuration to disk."""
        try:
            with open(self.config_path, "w") as f:
                json.dump(self.config.to_dict(), f, indent=2)
            logger.info(f"Saved config to {self.config_path}")
        except Exception as e:
            logger.error(f"Failed to save config: {e}")
    
    def get_lanes(self) -> List[Dict[str, Any]]:
        """Get lane configurations as list of dicts."""
        return [lane.to_dict() for lane in self.config.lanes]
    
    def set_lanes(self, lanes_data: List[Dict[str, Any]]) -> None:
        """Set lane configurations from list of dicts."""
        self.config.lanes = [LaneConfig.from_dict(lane) for lane in lanes_data]
        self.save()
    
    def reload(self) -> None:
        """Force reload configuration from disk."""
        self._config = self._load()
