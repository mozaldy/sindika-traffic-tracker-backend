"""Configuration management for the traffic detection system."""

import json
import os
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# Default module configuration
DEFAULT_MODULES = {
    "speed": True,
    "turn": False,  # Turn detection (requires zone config)
    "plate": False  # Off by default (expensive)
}


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
class ZoneConfig:
    """Configuration for a detection zone (direction or plate)."""
    id: str
    name: str
    type: str  # "direction" | "plate"
    polygon: List[float]  # [x1,y1,x2,y2,...] flat normalized 0-1
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "type": self.type,
            "polygon": self.polygon
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ZoneConfig":
        # Support both 'polygon' (flat array) and 'points' (nested array) formats
        polygon = data.get("polygon", [])
        if not polygon and "points" in data:
            # Convert [[x1,y1], [x2,y2]] to [x1,y1,x2,y2]
            points = data.get("points", [])
            polygon = [coord for point in points for coord in point]
        
        return cls(
            id=data.get("id", ""),
            name=data.get("name", "Zone"),
            type=data.get("type", "direction"),
            polygon=polygon
        )


@dataclass
class AppConfig:
    """Application configuration."""
    lanes: List[LaneConfig] = field(default_factory=list)
    
    # Module toggles
    modules: Dict[str, bool] = field(default_factory=lambda: DEFAULT_MODULES.copy())
    
    # Plate capture settings
    plate_trigger: str = "on_line"  # "on_line", "on_speed_exceed", "always"
    speed_threshold: float = 80.0  # km/h threshold for "on_speed_exceed" trigger
    
    # Detection zones (direction, plate)
    zones: List[ZoneConfig] = field(default_factory=list)
    
    # Plate capture line [x1, y1, x2, y2] normalized 0-1
    plate_line: Optional[List[float]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "lanes": [lane.to_dict() for lane in self.lanes],
            "zones": [zone.to_dict() for zone in self.zones],
            "modules": self.modules,
            "plate_trigger": self.plate_trigger,
            "speed_threshold": self.speed_threshold,
            "plate_line": self.plate_line
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AppConfig":
        lanes = [LaneConfig.from_dict(lane) for lane in data.get("lanes", [])]
        # Support both 'zones' (new) and 'turn_zones' (old) keys
        zones_data = data.get("zones", data.get("turn_zones", []))
        zones = [ZoneConfig.from_dict(zone) for zone in zones_data]
        modules = {**DEFAULT_MODULES, **data.get("modules", {})}
        return cls(
            lanes=lanes,
            zones=zones,
            modules=modules,
            plate_trigger=data.get("plate_trigger", "on_line"),
            speed_threshold=data.get("speed_threshold", 80.0),
            plate_line=data.get("plate_line")
        )


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
    
    def get_modules(self) -> Dict[str, bool]:
        """Get current module configuration."""
        return self.config.modules.copy()
    
    def set_modules(self, modules: Dict[str, bool]) -> None:
        """Set module configuration."""
        self.config.modules.update(modules)
        self.save()
        logger.info(f"Updated modules: {self.config.modules}")
    
    def is_module_enabled(self, module_name: str) -> bool:
        """Check if a specific module is enabled."""
        return self.config.modules.get(module_name, False)
    
    def get_plate_trigger(self) -> str:
        """Get the plate capture trigger mode."""
        return self.config.plate_trigger
    
    def set_plate_trigger(self, trigger: str) -> None:
        """Set the plate capture trigger mode."""
        if trigger in ("on_exit", "on_speed_exceed", "always", "on_line"):
            self.config.plate_trigger = trigger
            self.save()
    
    def get_speed_threshold(self) -> float:
        """Get the speed threshold for plate capture trigger."""
        return self.config.speed_threshold
    
    def set_speed_threshold(self, threshold: float) -> None:
        """Set the speed threshold for plate capture trigger."""
        self.config.speed_threshold = threshold
        self.save()
    
    def get_zones(self) -> List[Dict[str, Any]]:
        """Get zone configurations as list of dicts."""
        return [zone.to_dict() for zone in self.config.zones]

    def set_zones(self, zones_data: List[Dict[str, Any]]) -> None:
        """Set zone configurations from list of dicts."""
        self.config.zones = [ZoneConfig.from_dict(zone) for zone in zones_data]
        self.save()

    def get_plate_line(self) -> Optional[List[float]]:
        """Get the plate capture line coordinates."""
        return self.config.plate_line

    def set_plate_line(self, line: Optional[List[float]]) -> None:
        """Set the plate capture line coordinates."""
        self.config.plate_line = line
        self.save()
        logger.info(f"Updated plate line: {line}")

    def reload(self) -> None:
        """Force reload configuration from disk."""
        self._config = self._load()

