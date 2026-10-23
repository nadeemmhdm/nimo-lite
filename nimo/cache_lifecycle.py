"""Conversation lifecycle hooks for Nimo's inference cache."""
from __future__ import annotations

_INSTALLED = False


def install_cache_lifecycle() -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    from .conversation import ConversationStore
    from .fast_inference import reset_inference_cache

    original_create = ConversationStore.create

    def create_with_fresh_cache(self, *args, **kwargs):
        # A new chat is a semantic boundary even when its first tokens happen to match
        # the previous chat. Edited/regenerated histories are handled by exact-prefix
        # validation inside FastInferenceEngine.
        reset_inference_cache()
        return original_create(self, *args, **kwargs)

    ConversationStore.create = create_with_fresh_cache
    _INSTALLED = True
