class BaseSimCLRException(Exception):
    """Base exception"""
    
class InvalidBackBoneError(BaseSimCLRException):
    """Raised when the choice of backbone Convnet is invalid."""
    
class InvalidDatasetSelection(BaseSimCLRException):
    """Raised when the choice of dataset is invalid."""
    