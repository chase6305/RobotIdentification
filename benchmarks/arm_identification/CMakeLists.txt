cmake_minimum_required(VERSION 3.15)
project(arm_identification LANGUAGES CXX)

find_package(robot_identification 0.2.0 REQUIRED)
find_package(robot_identification_pinocchio 0.2.0 REQUIRED)

add_executable(arm_identification fit_arm.cpp forward_validation.cpp observable_parameters.cpp
  payload_fit.cpp physical_arm_fit.cpp raw_preprocessing.cpp)
target_link_libraries(arm_identification PRIVATE robot_identification::pinocchio)
