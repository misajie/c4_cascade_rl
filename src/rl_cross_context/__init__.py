"""Cross-context (K562↔RPE1) active-acquisition RL scaffold.

Hard rules (v1):
- RL is core; no LLM / diffusion / KG / extra cell actions
- cost = 1 per perturbation
- reward = L_t - L_{t+1} on fixed audit set H; H never enters policy state
- paired acquisition queue across methods
"""

__version__ = "0.1.0"

from .config import CrossContextConfig, load_config

__all__ = ["CrossContextConfig", "load_config", "__version__"]
