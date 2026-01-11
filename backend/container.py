"""Dependency injection container for Coarch."""

from typing import TypeVar
import threading


T = TypeVar("T")


class Container:
    """Simple dependency injection container."""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if Container._instance is None:
            with Container._lock:
                if Container._instance is None:
                    Container._instance = object.__new__(cls)
                    Container._instance._init()
        return Container._instance

    def _init(self):
        """Initialize container state."""
        self._dependencies = {}
        self._singletons = {}

    def register(self, interface, implementation, singleton=True):
        """Register a dependency."""
        key = interface.__name__
        self._dependencies[key] = (implementation, singleton)

    def resolve(self, interface):
        """Resolve a dependency."""
        key = interface.__name__

        if key not in self._dependencies:
            raise ValueError(f"No dependency registered for {key}")

        implementation, is_singleton = self._dependencies[key]

        if is_singleton:
            if key not in self._singletons:
                self._singletons[key] = implementation()
            return self._singletons[key]

        return implementation()

    def clear(self):
        """Clear all registered dependencies and singletons."""
        self._dependencies.clear()
        self._singletons.clear()


def get_container():
    """Get the global dependency container."""
    return Container()


def resolve(interface):
    """Resolve a dependency from the global container."""
    return get_container().resolve(interface)
