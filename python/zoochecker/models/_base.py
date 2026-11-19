from abc import ABC, abstractmethod
from typing import Any

class BaseModel(ABC):
    """
    Separate from the interface of the engine, mostly
    going to be the process being put in the things
    """
    @abstractmethod
    def forward(self, **kwargs) -> Any:
        """
        Technically going to modify those things right in this to modify the 
        tensor x to the output

        input x: tensor, output tensor output

        this would be a GPU bounded process so the thing must be separate from
        the engine, the engine would handle the non-GPU works (tokenizer, remap output, etc)
        """
        raise NotImplementedError("Forward method must be implemented!!")