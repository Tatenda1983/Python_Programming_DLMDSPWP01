class DataLoadError(Exception):
    """Raised when a data file cannot be loaded or validated."""
    pass


class MappingError(Exception):
    """Raised when test-point mapping fails unexpectedly."""
    pass