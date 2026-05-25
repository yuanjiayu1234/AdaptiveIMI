__all__ = ["Hybrid8WayIMICache"]


def __getattr__(name):
    if name == "Hybrid8WayIMICache":
        from .cache import Hybrid8WayIMICache

        return Hybrid8WayIMICache
    raise AttributeError(name)
