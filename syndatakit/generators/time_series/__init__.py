from .var import VARGenerator

def VECMGARCHGenerator(*args, **kwargs):
    try:
        from .vecm_garch import VECMGARCHGenerator as _V
        return _V(*args, **kwargs)
    except ImportError:
        raise ImportError("VECMGARCHGenerator requires: pip install syndatakit[timeseries]")

__all__ = ["VARGenerator", "VECMGARCHGenerator"]
