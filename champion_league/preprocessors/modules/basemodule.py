class BaseModule:
    def embed(self, *args, **kwargs):
        """Base class for preprocessing modules."""
        raise NotImplementedError
