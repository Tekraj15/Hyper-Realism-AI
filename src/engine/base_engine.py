from abc import ABC, abstractmethod

class GenerativeEngine(ABC):
    @abstractmethod
    def load_models(self):
        """Load models into memory."""
        pass

    @abstractmethod
    def generate(self, **kwargs):
        """Run the generation pipeline."""
        pass