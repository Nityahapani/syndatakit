from .gaussian_copula     import GaussianCopulaGenerator
from .dp_gaussian_copula  import DPGaussianCopulaGenerator

def VineCopulaGenerator(*args, **kwargs):
    try:
        from .vine_copula import VineCopulaGenerator as _V
        return _V(*args, **kwargs)
    except ImportError:
        raise ImportError("VineCopulaGenerator requires: pip install syndatakit[vine]")

__all__ = ["GaussianCopulaGenerator", "DPGaussianCopulaGenerator", "VineCopulaGenerator"]
