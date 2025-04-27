cmake_minimum_required(VERSION 3.10)
project(pinocchio_kinematics)

# Find Pinocchio and its dependencies
find_package(pinocchio REQUIRED)
find_package(Eigen3 REQUIRED)

# Create the executable
add_executable(pinocchio_kinematics pinocchio_kinematic.cpp)

# Link against Pinocchio
target_link_libraries(pinocchio_kinematics pinocchio::pinocchio)

# Include directories for Eigen
target_include_directories(pinocchio_kinematics PRIVATE ${EIGEN3_INCLUDE_DIR})

# C++11 standard required
set_target_properties(pinocchio_kinematics PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
)