from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

from curobo.cuda_robot_model.types import KinematicsTensorConfig

@dataclass
class MultiplyChainKinematicsTensorConfig(KinematicsTensorConfig):
    
    ee_links: Optional[str] = None
    multi_ee_idx: Optional[List[int]] = None
    
    def copy_(self, new_config: KinematicsTensorConfig) -> MultiplyChainKinematicsTensorConfig:
        super().copy_(new_config)
        
        if isinstance(new_config, MultiplyChainKinematicsTensorConfig):
            self.multi_ee_idx = new_config.multi_ee_idx
            self.ee_links = new_config.ee_links