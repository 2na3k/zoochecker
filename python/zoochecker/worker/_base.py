from abc import ABC, abstractmethod
import asyncio

from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Dict, List, Optional

class GenerateOutput:
    """
    Idk but get me the schema for the OpenAI for me
    """
    pass

class BaseWorker(ABC):
    """
    Explain: base model handler for this thing

    Inspired by omlx + huggingface api for this, will be filled in later

    This would be the wrapper for the execution of model, model would be the abstraction 
    to be fit into this thing

    Would later to be inherited to be TPWorker for tensor parallel worker,
    or normal worker.

    Why need to divide the abstraction of worker and model:
    - Model might be added with the 
    """

    @property
    def model_name(self) -> str:
        return self.model_name

    @property
    @abstractmethod
    def tokenizer(self) -> Any:
        pass

    @abstractmethod
    async def start(self) -> None:
        pass

    @abstractmethod
    async def stop(self) -> None:
        pass

    @abstractmethod
    async def generate(self, **kwargs) -> GenerateOutput:
        pass


    @abstractmethod
    async def chat(self, **kwargs) -> GenerateOutput:
        pass


    @abstractmethod
    async def stream_generate(self, **kwargs) -> AsyncIterator[GenerateOutput]:
        pass
