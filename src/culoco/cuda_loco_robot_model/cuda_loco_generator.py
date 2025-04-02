from __future__ import annotations



import os
import copy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import torch
import torch.autograd.profiler as profiler

from curobo.cuda_robot_model.urdf_kinematics_parser import UrdfKinematicsParser
from curobo.cuda_robot_model.cuda_robot_generator import CudaRobotGeneratorConfig, CudaRobotGenerator
from curobo.cuda_robot_model.kinematics_parser import LinkParams
from curobo.util.logger import log_error, log_info, log_warn

from culoco.util_file import get_assets_path, get_robot_configs_path, join_path, load_yaml
from culoco.cuda_loco_robot_model.type import MultiplyChainKinematicsTensorConfig

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
class CudaLocoGeneratorConfig(CudaRobotGeneratorConfig):
    """The modification for loco-manipulation"""
    #: enable multi-chain
    # enable_locomotion: bool = True
    
    # #: enable manipulation
    # enable_manipulation: bool = True
    
    #: enable multi-chain
    enable_multi_chain: bool = True
    
    #: all ee links
    ee_links: Optional[List[str]] = None
    
    #: leg ee links
    leg_ee_links: Optional[List[str]] = None
    
    #: arm ee link
    arm_ee_link: Optional[str] = None
    
    ee_link: Optional[str] = None
    
    def __post__init__(self):
        """Post initialization adds absolute paths, converts dictionaries to objects."""
        print("????????????????????????????")
        # add root path:
        # Check if an external asset path is provided:
        asset_path = get_assets_path()
        robot_path = get_robot_configs_path()
        print("asset_path",asset_path)
        print("robot_path",robot_path)
        if self.external_asset_path is not None:
            log_warn("Deprecated: external_asset_path is deprecated, use ContentPath")
            asset_path = self.external_asset_path
        if self.external_robot_configs_path is not None:
            log_warn("Deprecated: external_robot_configs_path is deprecated, use ContentPath")
            robot_path = self.external_robot_configs_path

        if self.urdf_path is not None:
            self.urdf_path = join_path(asset_path, self.urdf_path)
        if self.usd_path is not None:
            self.usd_path = join_path(asset_path, self.usd_path)
        if self.asset_root_path != "":
            self.asset_root_path = join_path(asset_path, self.asset_root_path)
        elif self.urdf_path is not None:
            self.asset_root_path = os.path.dirname(self.urdf_path)
        print("__post_init__",self.link_names)
        if self.collision_spheres is None and (
            self.collision_link_names is not None and len(self.collision_link_names) > 0
        ):
            log_error("collision link names are provided without robot collision spheres")
        if self.load_link_names_with_mesh:
            if self.link_names is None:
                self.link_names = copy.deepcopy(self.mesh_link_names)
            else:
                for i in self.mesh_link_names:
                    if i not in self.link_names:
                        self.link_names.append(i)
        if self.link_names is None:
            if self.enable_multi_chain:
                self.link_names = copy.deepcopy(self.ee_links)
            else:
                self.link_names = [self.ee_link]
        if self.collision_link_names is None:
            self.collision_link_names = []
        if self.enable_multi_chain == False:
            if self.ee_link not in self.link_names:
                self.link_names.append(self.ee_link)
        else:
            for ee_link in self.ee_links:
                if ee_link not in self.link_names:
                    self.link_names.append(ee_link)
        print("__post_init__",self.link_names)
        if self.collision_spheres is not None:
            if isinstance(self.collision_spheres, str):
                coll_yml = join_path(robot_path, self.collision_spheres)
                coll_params = load_yaml(coll_yml)

                self.collision_spheres = coll_params["collision_spheres"]
            if self.extra_collision_spheres is not None:
                for k in self.extra_collision_spheres.keys():
                    new_spheres = [
                        {"center": [0.0, 0.0, 0.0], "radius": -10.0}
                        for n in range(self.extra_collision_spheres[k])
                    ]
                    self.collision_spheres[k] = new_spheres
        if self.use_usd_kinematics and self.usd_path is None:
            log_error("usd_path is required to load kinematics from usd")
        if self.usd_flip_joints is None:
            self.usd_flip_joints = {}
        if self.usd_flip_joint_limits is None:
            self.usd_flip_joint_limits = []
        if self.extra_links is None:
            self.extra_links = {}
        else:
            for k in self.extra_links.keys():
                if isinstance(self.extra_links[k], dict):
                    self.extra_links[k] = LinkParams.from_dict(self.extra_links[k])
        if isinstance(self.cspace, Dict):
            self.cspace = CSpaceConfig(**self.cspace, tensor_args=self.tensor_args)
            
        print("post init collision link names", self.collision_link_names)

class CudaLocoGenerator(CudaRobotGenerator, CudaLocoGeneratorConfig):
    
    def __init__(self, config: CudaLocoGeneratorConfig) -> None:
        """Initialize the robot generator.

        Args:
            config: Parameters to initialize the robot generator.
        """
        super().__init__(**vars(config))
        # config.__post_init__()
        
        self.cpu_tensor_args = self.tensor_args.cpu()

        self._self_collision_data = None
        self.non_fixed_joint_names = []
        self._n_dofs = 1
        self._kinematics_config = None
        
        print("generator __init__ collision link names", self.collision_link_names)
        self.initialize_tensors()
    
    @profiler.record_function("robot_generator/initialize_tensors")
    def initialize_tensors(self):
        """Initialize tensors for kinematics representatiobn."""
        self._joint_limits = None
        self._self_collision_data = None
        self.lock_jointstate = None
        self.lin_jac, self.ang_jac = None, None

        # 初始化碰撞球（用于碰撞检测）
        # 表示每个链接上的碰撞球体（中心坐标 + 半径），每行是 (x, y, z, r)。
        self._link_spheres_tensor = torch.empty(
            (0, 4), device=self.tensor_args.device, dtype=self.tensor_args.dtype
        )
        # 映射碰撞球体对应的 link 的索引。
        self._link_sphere_idx_map = torch.empty(
            (0), dtype=torch.int16, device=self.tensor_args.device
        )
        # 总共的碰撞球数目设为 0（还未填充）。
        self.total_spheres = 0
        # 初始化自身碰撞的距离阈值矩阵和偏移
        # 初始化一个碰撞距离矩阵，默认设为 -inf，表示默认两两之间不可碰撞（或未定义距离）。

        # 随着 total_spheres 增加，这个矩阵会被重新生成。
        self.self_collision_distance = (
            torch.zeros(
                (self.total_spheres, self.total_spheres),
                dtype=self.tensor_args.dtype,
                device=self.tensor_args.device,
            )
            - torch.inf
        )
        # 每个碰撞球体的碰撞检测偏移值（一般为一个小的正数）。
        self.self_collision_offset = torch.zeros(
            (self.total_spheres), dtype=self.tensor_args.dtype, device=self.tensor_args.device
        )
        # create a mega list of all links that we need:
        # 构建 link 列表，包括正常 link、用于碰撞检测的 link 及额外 link. 初始化一个新的 other_links 列表，用于构造运动学链。
        other_links = copy.deepcopy(self.link_names)
        print(other_links)
        print("initialize_tensors collision link names", self.collision_link_names)
        # 将 collision_link_names 中未在原始 link_names 中的 link 也加入（因为用于碰撞）。
        for i in self.collision_link_names:
            if i not in self.link_names:
                other_links.append(i)
        # 对额外添加的 link（比如虚拟关节或自定义结构）添加其父节点进来，确保运动学链闭环。
        for i in self.extra_links:
            p_name = self.extra_links[i].parent_link_name
            if p_name not in self.link_names and p_name not in other_links:
                other_links.append(p_name)

        # other_links = list(set(self.link_names + self.collision_link_names))

        # load kinematics parser based on file type:
        # NOTE: Also add option to load from data buffers.
        if self.use_usd_kinematics:
            self._kinematics_parser = UsdKinematicsParser(
                self.usd_path,
                flip_joints=self.usd_flip_joints,
                flip_joint_limits=self.usd_flip_joint_limits,
                extra_links=self.extra_links,
                usd_robot_root=self.usd_robot_root,
            )
        else:
            self._kinematics_parser = UrdfKinematicsParser(
                self.urdf_path,
                mesh_root=self.asset_root_path,
                extra_links=self.extra_links,
                load_meshes=self.load_meshes,
            )

        # 构造运动学链
        if self.enable_multi_chain == False:
            if self.lock_joints is None:
                self._build_kinematics(self.base_link, self.ee_link, other_links, self.link_names)
            else:
                self._build_kinematics_with_lock_joints(
                    self.base_link, self.ee_link, other_links, self.link_names, self.lock_joints
                )
            if self.cspace is None:
                jpv = self._get_joint_position_velocity_limits()
                self.cspace = CSpaceConfig.load_from_joint_limits(
                    jpv["position"][1, :], jpv["position"][0, :], self.joint_names, self.tensor_args
                )
        else:
            if self.lock_joints is None:
                self._build_multi_chain_kinematics(self.base_link, self.ee_links, other_links, self.link_names)
            else:
                self._build_kinematics_with_lock_joints(
                    self.base_link, self.ee_link, other_links, self.link_names, self.lock_joints
                )
            if self.cspace is None:
                jpv = self._get_joint_position_velocity_limits()
                self.cspace = CSpaceConfig.load_from_joint_limits(
                    jpv["position"][1, :], jpv["position"][0, :], self.joint_names, self.tensor_args
                )

        self.cspace.inplace_reindex(self.joint_names)
        self._update_joint_limits()
        self._ee_idx = -1
        self._mutli_ee_idx = None
        if self.enable_multi_chain:
            for i, ee_link in enumerate(self.ee_links):
                self._mutli_ee_idx[i] = self.link_names.index(ee_link)
        else:
            self._ee_idx[0] = self.link_names.index(self.ee_link)

        # create kinematics tensor:
        self._kinematics_config = MultiplyChainKinematicsTensorConfig(
            fixed_transforms=self._fixed_transform,
            link_map=self._link_map,
            joint_map=self._joint_map,
            joint_map_type=self._joint_map_type,
            joint_offset_map=self._joint_offset_map,
            store_link_map=self._store_link_map,
            link_chain_map=self._link_chain_map,
            link_names=self.link_names,
            link_spheres=self._link_spheres_tensor,
            link_sphere_idx_map=self._link_sphere_idx_map,
            n_dof=self._n_dofs,
            joint_limits=self._joint_limits,
            non_fixed_joint_names=self.non_fixed_joint_names,
            total_spheres=self.total_spheres,
            link_name_to_idx_map=self._name_to_idx_map,
            joint_names=self.joint_names,
            debug=self.debug,
            ee_idx=self._ee_idx,
            multi_ee_idx = self._mutli_ee_idx,
            ee_links = self.ee_links,
            mesh_link_names=self.mesh_link_names,
            cspace=self.cspace,
            base_link=self.base_link,
            ee_link=self.ee_link,
            lock_jointstate=self.lock_jointstate,
            mimic_joints=self._mimic_joint_data,
        )
        if self.asset_root_path is not None and self.asset_root_path != "":
            self._kinematics_parser.add_absolute_path_to_link_meshes(self.asset_root_path)
            
            
    @profiler.record_function("robot_generator/build_multi_chain")
    def _build_multi_chain(
        self,
        base_link: str,
        # arm_ee_link: str,
        ee_links: List[str],
        other_links: List[str],
    ) -> List[str]:
        """Build kinematic tree of the robot.

        Args:
            base_link: Name of base link for the chain.
            ee_link: Name of end-effector link for the chain.
            other_links: List of other links to add to the chain.

        Returns:
            List[str]: List of link names in the chain.
        """
        self._n_dofs = 0
        self._controlled_links = []
        self._bodies = []
        self._name_to_idx_map = dict()
        self.base_link = base_link
        # self.arm_ee_link = arm_ee_link
        # self.leg_ee_links = leg_ee_links
        self.ee_links = ee_links
        self.joint_names = []
        self._fixed_transform = []
        chain_link_names = []
        
        for i, ee_link in enumerate(ee_links):
            chain = self._kinematics_parser.get_chain(base_link, ee_link)
            if i == 0:
                self._add_body_to_tree(chain[0], base=True)
                chain_link_names.append(chain[0])
            for j, l_name in enumerate(chain[1:]):
                    self._add_body_to_tree(l_name)
                    chain_link_names.append(l_name)
            
        # check if all links are in the built tree:

        for i in other_links:
            if i in self._name_to_idx_map:
                continue
            if i not in self.extra_links.keys():
                print("other_links",i)
                chain_l_names = self._kinematics_parser.get_chain(base_link, i)

                for k in chain_l_names:
                    if k in chain_link_names:
                        continue
                    # if link name is not in chain, add to chain
                    chain_link_names.append(k)
                    # add to tree:
                    self._add_body_to_tree(k, base=False)
        for i in self.extra_links.keys():
            if i not in chain_link_names:
                self._add_body_to_tree(i, base=False)
                chain_link_names.append(i)

        self.non_fixed_joint_names = self.joint_names.copy()
        return chain_link_names
    
    @profiler.record_function("robot_generator/build_multi_chain_kinematics")
    def _build_multi_chain_kinematics(
        self, base_link: str, ee_links: List[str], other_links: List[str], link_names: List[str]
    ):
        """Build kinematics tensors given base link, end-effector link and other links.

        Args:
            base_link: Name of base link for the kinematic tree.
            ee_link: Name of end-effector link for the kinematic tree.
            other_links: List of other links to add to the kinematic tree.
            link_names: List of link names to store poses after kinematics computation.
        """
        chain_link_names = self._build_multi_chain(base_link, ee_links, other_links)
        self._build_kinematics_tensors(base_link, link_names, chain_link_names)
        if self.collision_spheres is not None and len(self.collision_link_names) > 0:
            self._build_collision_model(
                self.collision_spheres, self.collision_link_names, self.collision_sphere_buffer
            )