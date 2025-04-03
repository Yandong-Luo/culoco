# Changelog
All notable changes to this project will be documented in this file.


## [0.0.1] - 2024-03-29
### Changed
- Yandong Luo: Overwrite the model configuration based on curobo (Unverified): cuda_loco_generator.py, cuda_loco_model.py

## [0.0.2] - 2024-03-30
### Changed
- Yandong Luo: No longer use inheritance. Currently it can basically run and compile

## [0.0.3] - 2024-04-01
### Changed
- Yandong Luo: The construction of multi-chain kinematic has been initially completed, and some verification has been passed (tensor and link_map)

## [0.0.4] - 2024-04-02
### Changed
- Yandong Luo: Since the gripper of go2_arx does not support movement in urdf, I have temporarily given up using go2_arx. I am currently using b1_z1. The yml should be configured correctly, and I have added the sphere, which can be displayed in isaacsim. However, due to the lack of KF, the sphere is still embedded in the ground.

## [0.0.4] - 2024-04-02
### Changed
- Yandong Luo: Add foam file for sphere