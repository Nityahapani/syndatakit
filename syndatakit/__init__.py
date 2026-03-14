"""
syndatakit v2 — research-grade synthetic data for finance & econometrics.

Quick start:
    from syndatakit.generators import GaussianCopulaGenerator
    from syndatakit.catalog    import load_seed
    from syndatakit.fidelity   import fidelity_report
    from syndatakit.privacy    import privacy_audit
    from syndatakit.calibration import apply_scenario
"""
from .generators.base                      import BaseGenerator
from .generators.cross_sectional           import GaussianCopulaGenerator
from .generators.time_series               import VARGenerator
from .generators.panel                     import FixedEffectsGenerator
from .catalog                              import list_datasets, get_dataset_info, load_seed
from .fidelity                             import fidelity_report, format_report
from .privacy                              import privacy_audit, format_audit
from .calibration                          import apply_scenario, list_scenarios
from .io                                   import read, write, validate

__version__ = "2.0.1"
__all__ = [
    "BaseGenerator","GaussianCopulaGenerator","VARGenerator","FixedEffectsGenerator",
    "list_datasets","get_dataset_info","load_seed",
    "fidelity_report","format_report",
    "privacy_audit","format_audit",
    "apply_scenario","list_scenarios",
    "read","write","validate",
]
