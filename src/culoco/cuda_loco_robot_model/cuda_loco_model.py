from __future__ import annotations
# Standard Library
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from curobo.cuda_robot_model.types import (
    CSpaceConfig,
    JointLimits,
    KinematicsTensorConfig,
    SelfCollisionKinematicsConfig,
)
from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModelConfig, CudaRobotModel
from curobo.cuda_robot_model.cuda_robot_generator import CudaRobotGeneratorConfig
from culoco.cuda_loco_robot_model.cuda_loco_generator import CudaLocoGeneratorConfig, CudaLocoGenerator
from curobo.types.base import TensorDeviceType
from culoco.util_file import get_assets_path, get_robot_configs_path, join_path

@dataclass
class CudaLocoModelConfig():
    """
    Extended robot model configuration for loco-manipulation systems
    
    Provides specific configuration parameters for quadruped robots with manipulator arms
    """
    #: Device and floating point precision to use for kinematics.
    # tensor_args: TensorDeviceType
    
    basic_model_config: CudaRobotModelConfig
    
    # # Separate collision detection configuration for the arm
    # arm_collision_config: Optional[SelfCollisionKinematicsConfig] = None
    
    # # Separate collision detection configuration for the legs
    # leg_collision_config: Optional[SelfCollisionKinematicsConfig] = None
    
    @staticmethod
    def from_basic_urdf(
        urdf_path: str,
        base_link: str,
        leg_ee_links: List[str],
        arm_ee_link: str,
        tensor_args: TensorDeviceType = TensorDeviceType(),
    ) -> CudaLocoModelConfig:
        config = CudaLocoGeneratorConfig(
                                         CudaRobotGeneratorConfig(base_link,
                                                                  arm_ee_link,
                                                                  tensor_args,
                                                                  urdf_path=urdf_path,
                                                                  asset_root_path=get_assets_path()),
                                         leg_ee_links = leg_ee_links,
                                         arm_ee_link = arm_ee_link, 
                                         urdf_path = urdf_path,                                        
                                         )
        return CudaLocoModelConfig.from_config(config)
    
    @staticmethod
    def from_config(config: CudaLocoGeneratorConfig) -> CudaLocoModelConfig:
        """Create a robot model configuration from a generator configuration.

        Args:
            config: Input robot generator configuration.

        Returns:
            CudaRobotModelConfig: robot model configuration.
        """
        # create a config generator and load all values
        generator = CudaLocoGenerator(config)
        return CudaLocoModelConfig(CudaRobotModelConfig(
                                    tensor_args=generator.robot_generator.tensor_args,
                                    link_names=generator.robot_generator.link_names,
                                    kinematics_config=generator.robot_generator.kinematics_config,
                                    self_collision_config=generator.robot_generator.self_collision_config,
                                    kinematics_parser=generator.robot_generator.kinematics_parser,
                                    use_global_cumul=generator.robot_generator.use_global_cumul,
                                    compute_jacobian=generator.robot_generator.compute_jacobian,
                                    generator_config=config.base_config)
                                   )
    
#     @staticmethod
#     def from_robot_config(
#         config: CudaRobotModelConfig,
#         leg_joint_indices: List[int],
#         arm_joint_indices: List[int],
#         leg_ee_links: List[str],
#         arm_ee_link: Optional[str] = None,
#         base_link_name: Optional[str] = None,
#         contact_threshold: float = 0.01
#     ) -> 'LocoManipulationModelConfig':
#         """Create LocoManipulationModelConfig from existing CudaRobotModelConfig
        
#         Args:
#             config: Base robot configuration
#             leg_joint_indices: Indices of leg joints
#             arm_joint_indices: Indices of arm joints
#             leg_ee_links: Names of leg end-effector links
#             arm_ee_link: Name of arm end-effector link
#             base_link_name: Name of base link
#             contact_threshold: Threshold for contact detection
            
#         Returns:
#             LocoManipulationModelConfig: Extended configuration object
#         """
#         return LocoManipulationModelConfig(
#             tensor_args=config.tensor_args,
#             link_names=config.link_names,
#             kinematics_config=config.kinematics_config,
#             self_collision_config=config.self_collision_config,
#             kinematics_parser=config.kinematics_parser,
#             use_global_cumul=config.use_global_cumul,
#             compute_jacobian=config.compute_jacobian,
#             generator_config=config.generator_config,
#             leg_joint_indices=leg_joint_indices,
#             arm_joint_indices=arm_joint_indices,
#             leg_ee_links=leg_ee_links,
#             arm_ee_link=arm_ee_link,
#             base_link_name=base_link_name,
#             contact_threshold=contact_threshold
#         )
    
#     @staticmethod
#     def auto_configure_from_urdf(
#         base_config: CudaRobotModelConfig,
#         leg_joint_prefix: List[str],  # e.g. ["LF_", "RF_", "LH_", "RH_"]
#         arm_joint_prefix: List[str],  # e.g. ["arm_", "manipulator_"]
#         leg_ee_suffix: str = "_foot",  # e.g. suffix for leg end-effector links
#         arm_ee_link: Optional[str] = None,
#         base_link_name: Optional[str] = None
#     ) -> 'LocoManipulationModelConfig':
#         """Automatically configure based on joint and link name prefixes/suffixes
        
#         Suitable for robot URDFs following naming conventions, e.g. quadruped robots 
#         with LF_/RF_/LH_/RH_ prefixes
        
#         Args:
#             base_config: Base robot configuration
#             leg_joint_prefix: List of prefixes for leg joint names
#             arm_joint_prefix: List of prefixes for arm joint names
#             leg_ee_suffix: Suffix for leg end-effector link names
#             arm_ee_link: Name of arm end-effector link
#             base_link_name: Name of base link
            
#         Returns:
#             LocoManipulationModelConfig: Automatically configured object
#         """
#         # Auto-detect leg joint indices
#         leg_joint_indices = []
#         for i, name in enumerate(base_config.kinematics_config.joint_names):
#             for prefix in leg_joint_prefix:
#                 if name.startswith(prefix):
#                     leg_joint_indices.append(i)
#                     break
        
#         # Auto-detect arm joint indices
#         arm_joint_indices = []
#         for i, name in enumerate(base_config.kinematics_config.joint_names):
#             for prefix in arm_joint_prefix:
#                 if name.startswith(prefix):
#                     arm_joint_indices.append(i)
#                     break
        
#         # Auto-detect leg end-effector links
#         leg_ee_links = []
#         for link in base_config.link_names:
#             for prefix in leg_joint_prefix:
#                 if link.startswith(prefix) and link.endswith(leg_ee_suffix):
#                     leg_ee_links.append(link)
#                     break
        
#         return LocoManipulationModelConfig.from_robot_config(
#             base_config,
#             leg_joint_indices,
#             arm_joint_indices,
#             leg_ee_links,
#             arm_ee_link,
#             base_link_name
#         )


class CudaLocoModel(CudaLocoModelConfig):
    "CUDA Accelerated Loco-manipulation Model"
    def __init__(self, config: CudaLocoModelConfig):
        """Initialize kinematics instance with a robot model configuration.

        Args:
            config: Input robot model configuration.
        """
        super().__init__(**vars(config))
        # self._batch_size = 0
        # self.update_batch_size(1, reset_buffers=True)