#
# Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#
""" Example computing fk using curobo

"""
# Third Party
import torch

# CuRobo
from culoco.cuda_loco_robot_model.cuda_robot_model import CudaLocoRobotModel
from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModel
from curobo.types.base import TensorDeviceType
from culoco.types.robot import RobotConfig
from curobo.util.logger import setup_curobo_logger
from culoco.util_file import get_robot_path, join_path, load_yaml, get_assets_path

def demo_basic_robot():
    tensor_args = TensorDeviceType()

    # load a urdf:
    config_file = load_yaml(join_path(get_robot_path(), "go2.yml"))
    print("get_robot_path",get_assets_path())
    # asset_root_path = join_path(get_robot_path(), )
    urdf_file = config_file["robot_cfg"]["kinematics"][
        "urdf_path"
    ]  # Send global path starting with "/"
    base_link = config_file["robot_cfg"]["kinematics"]["base_link"]
    # ee_link = config_file["robot_cfg"]["kinematics"]["ee_link"]
    leg_ee_links = config_file["robot_cfg"]["kinematics"]["leg_ee_links"]
    ee_links = config_file["robot_cfg"]["kinematics"]["ee_links"]
    robot_cfg = RobotConfig.from_basic(urdf_file, base_link, ee_links, tensor_args)

    print(robot_cfg)
#     kin_model = CudaRobotModel(robot_cfg.kinematics)

#     # compute forward kinematics:

#     print(kin_model.basic_model_config.get_dof())
#     q = torch.rand((10, kin_model.basic_model_config.get_dof()), **(tensor_args.as_torch_dict()))
#     out = kin_model.get_state(q)
#     # here is the kinematics state:
#     # print(out)

def demo_full_config_robot():
    setup_curobo_logger("info")
    tensor_args = TensorDeviceType()
    # load a urdf:
    config_file = load_yaml(join_path(get_robot_path(), "go2.yml"))["robot_cfg"]
    robot_cfg = RobotConfig.from_dict(config_file, tensor_args)

    # print(robot_cfg.kinematics.)
    kin_model = None
    if robot_cfg.kinematics.generator_config.enable_multi_chain == False:
        kin_model = CudaRobotModel(robot_cfg.kinematics)
    else:
        kin_model = CudaLocoRobotModel(robot_cfg.kinematics)

    # compute forward kinematics:
    # q = torch.rand((10, kin_model.get_dof()), **(tensor_args.as_torch_dict()))
    # out = kin_model.get_state(q)
    # here is the kinematics state:
    # print(out)


if __name__ == "__main__":
    demo_basic_robot()
    demo_full_config_robot()
