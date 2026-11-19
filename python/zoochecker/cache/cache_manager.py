"""
Unified interface for cache, maybe the kind of cache factory like the old ones I did
"""

from ._base import Cache

class UnifiedCacheManager:
    def create_cache_something_something(self) -> Cache:
        """
        Generic cache handler for this thing, maybe
        """
        pass

    def create_prompt_cache(self, prompt) -> Cache:
        pass

    def append(self, prompt) -> None:
        pass