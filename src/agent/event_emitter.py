"""
Event emitter for real-time dashboard updates

Publishes agent events to WebSocket server for live visualization.
"""

import json
import asyncio
from typing import Optional, Callable, List, Dict, Any
from dataclasses import dataclass, asdict
from datetime import datetime


@dataclass
class AgentEvent:
    """Event data structure"""
    event_type: str
    timestamp: str
    data: Dict[str, Any]

    def to_json(self) -> str:
        return json.dumps(asdict(self))


class EventEmitter:
    """
    Emit events from agent to dashboard.

    Supports both sync and async listeners.
    """

    def __init__(self):
        self._listeners: List[Callable] = []
        self._async_listeners: List[Callable] = []
        self._event_queue: asyncio.Queue = None
        self._history: List[AgentEvent] = []

    def add_listener(self, callback: Callable):
        """Add synchronous listener"""
        self._listeners.append(callback)

    def add_async_listener(self, callback: Callable):
        """Add async listener"""
        self._async_listeners.append(callback)

    def set_queue(self, queue: asyncio.Queue):
        """Set async queue for WebSocket broadcasting"""
        self._event_queue = queue

    def emit(self, event_type: str, data: Dict[str, Any] = None):
        """
        Emit an event to all listeners.

        Event types:
        - agent_start: Agent started
        - iteration_start: Starting iteration N
        - feature_generated: LLM generated code
        - evaluation_start: Starting CV evaluation
        - evaluation_complete: CV evaluation done
        - feature_accepted: Feature improved RMSLE
        - feature_rejected: Feature did not improve
        - agent_complete: Agent finished all iterations
        """
        event = AgentEvent(
            event_type=event_type,
            timestamp=datetime.now().isoformat(),
            data=data or {}
        )

        self._history.append(event)

        # Notify sync listeners
        for listener in self._listeners:
            try:
                listener(event)
            except Exception as e:
                print(f"Event listener error: {e}")

        # Put in queue for async consumers
        if self._event_queue is not None:
            try:
                self._event_queue.put_nowait(event)
            except asyncio.QueueFull:
                pass  # Skip if queue is full

    def get_history(self) -> List[AgentEvent]:
        """Get all emitted events"""
        return self._history

    def clear_history(self):
        """Clear event history"""
        self._history = []


# Global emitter instance
_emitter: Optional[EventEmitter] = None


def get_emitter() -> EventEmitter:
    """Get or create global event emitter"""
    global _emitter
    if _emitter is None:
        _emitter = EventEmitter()
    return _emitter


def emit(event_type: str, data: Dict[str, Any] = None):
    """Convenience function to emit event"""
    get_emitter().emit(event_type, data)
