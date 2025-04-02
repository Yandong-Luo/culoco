from __future__ import annotations

# Standard Library
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

from culoco.cuda_loco_robot_model.cuda_loco_generator import (
    CudaLocoGenerator,
    CudaLocoGeneratorConfig,
)
from curobo.types.base import TensorDeviceType

# from curobo.cuda_robot
from curobo.cuda_robot_model.cuda_robot_model import (
    CudaRobotModelConfig
)

@dataclass
class CudaLocoModelConfig(CudaRobotModelConfig):
    
    @staticmethod
    def from_basic_urdf(
        urdf_path: str,
        base_link: str,
        ee_links: List[str],
        tensor_args: TensorDeviceType = TensorDeviceType(),
    ) -> CudaRobotModelConfig:
        """Load a cuda robot model from only urdf. This does not support collision queries.

        Args:
            urdf_path : Path of urdf file.
            base_link : Name of base link.
            ee_link : Name of end-effector link.
            tensor_args : Device to load robot model. Defaults to TensorDeviceType().

        Returns:
            CudaRobotModelConfig: cuda robot model configuration.
        """
        config = CudaLocoGeneratorConfig(base_link=base_link, ee_links=ee_links, tensor_args=tensor_args, urdf_path=urdf_path)
        return CudaLocoModelConfig.from_config(config)
    
    @staticmethod
    def from_config(config: CudaLocoGeneratorConfig) -> CudaRobotModelConfig:
        """Create a robot model configuration from a generator configuration.

        Args:
            config: Input robot generator configuration.

        Returns:
            CudaRobotModelConfig: robot model configuration.
        """
        # create a config generator and load all values
        generator = CudaLocoGenerator(config)
        return CudaRobotModelConfig(
            tensor_args=generator.tensor_args,
            link_names=generator.link_names,
            kinematics_config=generator.kinematics_config,
            self_collision_config=generator.self_collision_config,
            kinematics_parser=generator.kinematics_parser,
            use_global_cumul=generator.use_global_cumul,
            compute_jacobian=generator.compute_jacobian,
            generator_config=config,
        )