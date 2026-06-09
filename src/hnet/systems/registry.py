"""Registry for physics systems — enables string-based construction without core changes."""
from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar

if TYPE_CHECKING:
    from hnet.systems.base import PhysicsSystem

T = TypeVar("T")

SYSTEM_REGISTRY: dict[str, type["PhysicsSystem"]] = {}


def register_system(name: str):
    """Decorator: register a PhysicsSystem subclass under a string key.

    Usage
    -----
    @register_system("my_system")
    class MySystem(PhysicsSystem): ...

    system = hnet.system("my_system")  # works immediately
    """
    def decorator(cls: type[T]) -> type[T]:
        SYSTEM_REGISTRY[name] = cls  # type: ignore[assignment]
        return cls
    return decorator


def get_system(name: str, **kwargs) -> "PhysicsSystem":
    """Instantiate a registered system by name.

    Parameters
    ----------
    name : str
        Registry key (e.g., "nonlinear_pendulum", "spin_precession").
    **kwargs
        Passed to the system constructor.
    """
    if name not in SYSTEM_REGISTRY:
        available = ", ".join(sorted(SYSTEM_REGISTRY.keys()))
        raise KeyError(
            f"Unknown system '{name}'. Available: [{available}]. "
            f"Register new systems with @hnet.register_system(name)."
        )
    return SYSTEM_REGISTRY[name](**kwargs)
