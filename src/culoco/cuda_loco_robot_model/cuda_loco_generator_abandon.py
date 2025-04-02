# Standard Library
import os
import copy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import torch
import torch.autograd.profiler as profiler

from curobo.cuda_robot_model.urdf_kinematics_parser import UrdfKinematicsParser
from curobo.cuda_robot_model.cuda_robot_generator import CudaRobotGeneratorConfig, CudaRobotGenerator
from culoco.util_file import get_assets_path, get_robot_configs_path, join_path
from curobo.util.logger import log_error, log_info, log_warn
from curobo.cuda_robot_model.types import (
    CSpaceConfig,
    JointLimits,
    JointType,
    KinematicsTensorConfig,
    SelfCollisionKinematicsConfig,
)

try:
    # CuRobo
    from curobo.cuda_robot_model.usd_kinematics_parser import UsdKinematicsParser
except ImportError:
    log_info(
        "USDParser failed to import, install curobo with pip install .[usd] "
        + "or pip install usd-core, NOTE: Do not install this if using with Isaac Sim."
    )

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
        # self.initialize_tensors()
        # print(self.robot_generator.kinematics_config)
        
    @profiler.record_function("robot_generator/initialize_tensors")
    def initialize_tensors(self):
        """Initialize tensors for kinematics representatiobn."""
        rg = self.robot_generator
        rg._joint_limits = None
        rg._self_collision_data = None
        rg.lock_jointstate = None
        rg.lin_jac, rg.ang_jac = None, None

        rg._link_spheres_tensor = torch.empty(
            (0, 4), device=rg.tensor_args.device, dtype=rg.tensor_args.dtype
        )
        rg._link_sphere_idx_map = torch.empty(
            (0), dtype=torch.int16, device=rg.tensor_args.device
        )
        rg.total_spheres = 0
        rg.self_collision_distance = (
            torch.zeros(
                (rg.total_spheres, rg.total_spheres),
                dtype=rg.tensor_args.dtype,
                device=rg.tensor_args.device,
            )
            - torch.inf
        )
        rg.self_collision_offset = torch.zeros(
            (rg.total_spheres), dtype=rg.tensor_args.dtype, device=rg.tensor_args.device
        )
        # create a mega list of all links that we need:
        other_links = copy.deepcopy(rg.link_names)

        for i in rg.collision_link_names:
            if i not in rg.link_names:
                other_links.append(i)
        for i in rg.extra_links:
            p_name = rg.extra_links[i].parent_link_name
            if p_name not in rg.link_names and p_name not in other_links:
                other_links.append(p_name)

        # other_links = list(set(self.link_names + self.collision_link_names))

        # load kinematics parser based on file type:
        # NOTE: Also add option to load from data buffers.
        if rg.use_usd_kinematics:
            rg._kinematics_parser = UsdKinematicsParser(
                rg.usd_path,
                flip_joints=rg.usd_flip_joints,
                flip_joint_limits=rg.usd_flip_joint_limits,
                extra_links=rg.extra_links,
                usd_robot_root=rg.usd_robot_root,
            )
        else:
            rg._kinematics_parser = UrdfKinematicsParser(
                rg.urdf_path,
                mesh_root=rg.asset_root_path,
                extra_links=rg.extra_links,
                load_meshes=rg.load_meshes,
            )

        if rg.lock_joints is None:
            self._build_loco_kinematics(rg.base_link, self.arm_ee_link, self.leg_ee_links, other_links, rg.link_names)
        else:
            rg._build_kinematics_with_lock_joints(
                rg.base_link, rg.ee_link, other_links, rg.link_names, rg.lock_joints
            )
        if rg.cspace is None:
            jpv = rg._get_joint_position_velocity_limits()
            rg.cspace = CSpaceConfig.load_from_joint_limits(
                jpv["position"][1, :], jpv["position"][0, :], rg.joint_names, self.tensor_args
            )

        rg.cspace.inplace_reindex(rg.joint_names)
        rg._update_joint_limits()
        rg._ee_idx = rg.link_names.index(rg.ee_link)

        # create kinematics tensor:
        rg._kinematics_config = KinematicsTensorConfig(
            fixed_transforms=rg._fixed_transform,
            link_map=rg._link_map,
            joint_map=rg._joint_map,
            joint_map_type=rg._joint_map_type,
            joint_offset_map=rg._joint_offset_map,
            store_link_map=rg._store_link_map,
            link_chain_map=rg._link_chain_map,
            link_names=rg.link_names,
            link_spheres=rg._link_spheres_tensor,
            link_sphere_idx_map=rg._link_sphere_idx_map,
            n_dof=rg._n_dofs,
            joint_limits=rg._joint_limits,
            non_fixed_joint_names=rg.non_fixed_joint_names,
            total_spheres=rg.total_spheres,
            link_name_to_idx_map=rg._name_to_idx_map,
            joint_names=rg.joint_names,
            debug=rg.debug,
            ee_idx=rg._ee_idx,
            mesh_link_names=rg.mesh_link_names,
            cspace=rg.cspace,
            base_link=rg.base_link,
            ee_link=rg.ee_link,
            lock_jointstate=rg.lock_jointstate,
            mimic_joints=rg._mimic_joint_data,
        )
        if rg.asset_root_path is not None and rg.asset_root_path != "":
            rg._kinematics_parser.add_absolute_path_to_link_meshes(rg.asset_root_path)
    
    def _build_loco_chain(
        self,
        base_link: str,
        arm_ee_link: str,
        leg_ee_links: List[str],
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
        rg = self.robot_generator
        rg._n_dofs = 0
        rg._controlled_links = []
        rg._bodies = []
        rg._name_to_idx_map = dict()
        rg.base_link = base_link
        # self.ee_link = ee_link
        rg.joint_names = []
        rg._fixed_transform = []
        
        # Start with the main chain (typically the arm)
        chain_link_names = rg._kinematics_parser.get_chain(base_link, arm_ee_link)
        rg._add_body_to_tree(chain_link_names[0], base=True)
        for i, l_name in enumerate(chain_link_names[1:]):
            rg._add_body_to_tree(l_name)
        
        # Add leg end-effector chains
        for leg_ee in self.leg_ee_links:
            leg_chain = rg._kinematics_parser.get_chain(base_link, leg_ee)
            for link in leg_chain:
                if link in chain_link_names:
                    continue
                chain_link_names.append(link)
                rg._add_body_to_tree(link, base=False)
        
        # Check if all links in other_links are in the built tree
        for i in other_links:
            if i in self._name_to_idx_map:
                continue
            if i not in self.extra_links.keys():
                chain_l_names = rg._kinematics_parser.get_chain(base_link, i)
                
                for k in chain_l_names:
                    if k in chain_link_names:
                        continue
                    # if link name is not in chain, add to chain
                    chain_link_names.append(k)
                    # add to tree:
                    rg._add_body_to_tree(k, base=False)
        
        # Add extra links
        for i in self.extra_links.keys():
            if i not in chain_link_names:
                rg._add_body_to_tree(i, base=False)
                chain_link_names.append(i)
        
        self.non_fixed_joint_names = self.joint_names.copy()
        print("chain_link_names",chain_link_names)
        return chain_link_names
    
    @profiler.record_function("robot_generator/build_kinematics")
    def _build_loco_kinematics(
        self, base_link: str, arm_ee_link: str, leg_ee_links: str, other_links: List[str], link_names: List[str]
    ):
        """Build kinematics tensors given base link, end-effector link and other links.

        Args:
            base_link: Name of base link for the kinematic tree.
            ee_link: Name of end-effector link for the kinematic tree.
            other_links: List of other links to add to the kinematic tree.
            link_names: List of link names to store poses after kinematics computation.
        """
        print("_build_loco_kinematics")
        chain_link_names = self._build_loco_chain(base_link, arm_ee_link, leg_ee_links, other_links)
        self.robot_generator._build_kinematics_tensors(base_link, link_names, chain_link_names)
        if self.collision_spheres is not None and len(self.collision_link_names) > 0:
            self.robot_generator._build_collision_model(
                self.collision_spheres, self.collision_link_names, self.collision_sphere_buffer
            )