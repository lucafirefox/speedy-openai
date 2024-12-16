__version__ = "0.2.1"
__author__ = "Luca Ferrario"
__license__ = "MIT"
__copyright__ = "Copyright 2024 Luca F."
__maintainer__ = "Luca Ferrario"
__email__ = "lucaferrario199@gmail.com"

from .client import OpenAIClient
from .configs import Configs

__all__ = [
    "Configs",
    "OpenAIClient",
    "__author__",
    "__license__",
    "__version__",
]
