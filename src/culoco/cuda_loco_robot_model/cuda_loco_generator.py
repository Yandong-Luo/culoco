# Standard Library
import os
import copy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from curobo.cuda_robot_model.urdf_kinematics_parser import UrdfKinematicsParser
from curobo.cuda_robot_model.cuda_robot_generator import CudaRobotGeneratorConfig, CudaRobotGenerator
from culoco.util_file import get_assets_path, get_robot_configs_path, join_path
from curobo.util.logger import log_error, log_info, log_warn

@dataclass
class CudaLocoGeneratorConfig():
    
    base_config: CudaRobotGeneratorConfig
    
    enable_manipulation: bool = True
    
    # Indices of leg joints in joint_names
    leg_joint_indices: List[int] = field(default_factory=list)
    
    # Indices of arm joints in joint_names
    arm_joint_indices: List[int] = field(default_factory=list)
    
    # Names of end-effector links for the four legs
    leg_ee_links: List[str] = field(default_factory=list)
    
    # # Names of all end-effector links for loco-manipulation
    # ee_links: List[str] = field(default_factory=list)
    
    # Name of manipulator arm end-effector link, defaults to parent class ee_link
    arm_ee_link: Optional[str] = None
    
    # # Base link representing the robot's trunk
    # base_link_name: Optional[str] = None
    
    #: Path to load robot urdf.
    urdf_path: Optional[str] = None
    
    asset_root_path: str = get_assets_path()
    # Height threshold for contact detection
    contact_threshold: float = 0.01
    
    def __post_init__(self):
        # super().__post_init__()
        """Post initialization adds absolute paths, converts dictionaries to objects."""
        self.base_config.__post_init__()
        
        asset_path = get_assets_path()
        # mannually update the urdf path
        if self.urdf_path is not None:
            self.base_config.urdf_path = join_path(asset_path, self.urdf_path)


class CudaLocoGenerator(CudaLocoGeneratorConfig):
    def __init__(self, config: CudaLocoGeneratorConfig):
        super().__init__(**vars(config))
        self.robot_generator = CudaRobotGenerator(config.base_config)
        
        # print(self.robot_generator.kinematics_config)
    
    def _build_chain(
        self,
        base_link: str,
        ee_link: str,
        other_links: List[str],
    ) -> List[str]:
        """Build kinematic tree for loco-manipulation robot with multiple end-effectors.
        
        Args:
            base_link: Name of base link for the chain.
            ee_link: Name of primary end-effector link (arm).
            other_links: List of other links to add to the chain.
            
        Returns:
            List[str]: List of link names in the chain.
        """
        self._n_dofs = 0
        self._controlled_links = []
        self._bodies = []
        self._name_to_idx_map = dict()
        self.base_link = base_link
        self.ee_link = ee_link
        self.joint_names = []
        self._fixed_transform = []
        
        # Start with the main chain (typically the arm)
        chain_link_names = self._kinematics_parser.get_chain(base_link, ee_link)
        self._add_body_to_tree(chain_link_names[0], base=True)
        for i, l_name in enumerate(chain_link_names[1:]):
            self._add_body_to_tree(l_name)
        
        # Add leg end-effector chains
        for leg_ee in self.leg_ee_links:
            leg_chain = self._kinematics_parser.get_chain(base_link, leg_ee)
            for link in leg_chain:
                if link in chain_link_names:
                    continue
                chain_link_names.append(link)
                self._add_body_to_tree(link, base=False)
        
        # Check if all links in other_links are in the built tree
        for i in other_links:
            if i in self._name_to_idx_map:
                continue
            if i not in self.extra_links.keys():
                chain_l_names = self._kinematics_parser.get_chain(base_link, i)
                
                for k in chain_l_names:
                    if k in chain_link_names:
                        continue
                    # if link name is not in chain, add to chain
                    chain_link_names.append(k)
                    # add to tree:
                    self._add_body_to_tree(k, base=False)
        
        # Add extra links
        for i in self.extra_links.keys():
            if i not in chain_link_names:
                self._add_body_to_tree(i, base=False)
                chain_link_names.append(i)
        
        self.non_fixed_joint_names = self.joint_names.copy()
        return chain_link_names