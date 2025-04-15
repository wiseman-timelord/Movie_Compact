# Script: `.\scripts\exceptions.py`

# Exception classes
class MovieCompactError(Exception):
    """Base exception for Movie Consolidator errors."""
    pass

class HardwareError(Exception):
    """Exception for hardware-related errors."""
    pass

class ConfigurationError(Exception):
    """Exception for configuration errors."""
    pass

class ProcessingError(Exception):
    """Exception for processing errors."""
    pass
    
class AnalysisError(MovieCompactError):
    """Exception for analysis-related errors."""
    pass