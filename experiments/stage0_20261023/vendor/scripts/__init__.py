"""Use the isolated copied script modules before any other scripts package."""
from pkgutil import extend_path
__path__=extend_path(__path__,__name__)
