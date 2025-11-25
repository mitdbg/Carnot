"""Progress event logging for web session tracking."""
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import json
from pathlib import Path
from typing import Literal
import fcntl
import logging


logger = logging.getLogger(__name__)


class ProgressLevel(str, Enum):
    """Execution level for progress tracking."""
    OUTER = "outer"  # Main execution
    INNER = "inner"  # Nested PZ program


class ProgressStatus(str, Enum):
    """Status of a progress event."""
    STARTED = "started"
    PROGRESS = "progress"
    COMPLETED = "completed"
    ERROR = "error"


@dataclass
class ProgressEvent:
    """A single progress event to be logged."""
    timestamp: str
    session_id: str
    level: ProgressLevel | str
    operator_id: str
    operator_name: str
    status: ProgressStatus | str
    current: int = 0
    total: int = 0
    percentage: float = 0.0
    cost: float = 0.0
    message: str = ""
    
    def to_dict(self) -> dict:
        """Convert event to dictionary."""
        return {
            "timestamp": self.timestamp,
            "session_id": self.session_id,
            "level": self.level.value if isinstance(self.level, ProgressLevel) else self.level,
            "operator_id": self.operator_id,
            "operator_name": self.operator_name,
            "status": self.status.value if isinstance(self.status, ProgressStatus) else self.status,
            "progress": {
                "current": self.current,
                "total": self.total,
                "percentage": self.percentage,
            },
            "cost": self.cost,
            "message": self.message,
        }
    
    def to_json(self) -> str:
        """Convert event to JSON string."""
        return json.dumps(self.to_dict())


class ProgressLogger:
    """Thread-safe logger for progress events."""
    
    def __init__(self, log_file_path: str | None):
        """Initialize progress logger with optional file path."""
        self.log_file_path = Path(log_file_path) if log_file_path else None
        if self.log_file_path:
            # Ensure parent directory exists
            self.log_file_path.parent.mkdir(parents=True, exist_ok=True)
    
    def log_event(self, event: ProgressEvent):
        """Write a progress event to the log file (thread-safe)."""
        print(f"DEBUG [ProgressLogger.log_event]: log_file_path={self.log_file_path}")
        if not self.log_file_path:
            print(f"DEBUG [ProgressLogger.log_event]: No log file path, returning")
            return
        
        try:
            print(f"DEBUG [ProgressLogger.log_event]: Writing to {self.log_file_path}")
            with open(self.log_file_path, 'a') as f:
                # Use file locking for thread safety
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                try:
                    f.write(event.to_json() + '\n')
                    f.flush()
                    print(f"DEBUG [ProgressLogger.log_event]: Successfully wrote event")
                finally:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        except Exception as e:
            # Don't let logging errors crash the execution
            print(f"DEBUG [ProgressLogger.log_event]: ERROR: {e}")
            logger.warning(f"Failed to write progress event: {e}")

