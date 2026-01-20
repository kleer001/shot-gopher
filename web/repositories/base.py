"""Base repository interface."""

from abc import ABC, abstractmethod
from typing import Generic, TypeVar, Optional, List

T = TypeVar('T')


class Repository(ABC, Generic[T]):
    """
    Base repository interface.

    Abstracts data access - could be filesystem, database, etc.
    Follows Repository pattern for easy testing and swapping implementations.
    """

    @abstractmethod
    def get(self, id: str) -> Optional[T]:
        """Get entity by ID."""
        pass

    @abstractmethod
    def list(self) -> List[T]:
        """List all entities."""
        pass

    @abstractmethod
    def save(self, entity: T) -> T:
        """Save entity (create or update)."""
        pass

    @abstractmethod
    def delete(self, id: str) -> bool:
        """Delete entity by ID. Returns True if deleted, False if not found."""
        pass
