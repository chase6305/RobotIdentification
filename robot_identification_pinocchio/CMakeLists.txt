cmake_minimum_required(VERSION 3.15)
project(robot_identification_pinocchio VERSION 0.2.2 LANGUAGES CXX)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)
find_package(pinocchio REQUIRED)
find_package(robot_identification 0.1.1 REQUIRED)

add_library(robot_identification_pinocchio src/pinocchio_regressor.cpp)
add_library(robot_identification::pinocchio ALIAS robot_identification_pinocchio)
set_target_properties(robot_identification_pinocchio PROPERTIES EXPORT_NAME pinocchio)
target_compile_features(robot_identification_pinocchio PUBLIC cxx_std_17)
target_link_libraries(robot_identification_pinocchio
  PUBLIC robot_identification::robot_identification pinocchio::pinocchio)
target_include_directories(robot_identification_pinocchio PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(robot_identification_pinocchio PRIVATE -Wall -Wextra -Wpedantic)
endif()

install(TARGETS robot_identification_pinocchio
  EXPORT robot_identification_pinocchioTargets
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(EXPORT robot_identification_pinocchioTargets
  FILE robot_identification_pinocchioTargets.cmake
  NAMESPACE robot_identification::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/robot_identification_pinocchio)
configure_package_config_file(
  cmake/robot_identification_pinocchioConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/robot_identification_pinocchioConfig.cmake
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/robot_identification_pinocchio)
write_basic_package_version_file(
  ${CMAKE_CURRENT_BINARY_DIR}/robot_identification_pinocchioConfigVersion.cmake
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY SameMajorVersion)
install(FILES
  ${CMAKE_CURRENT_BINARY_DIR}/robot_identification_pinocchioConfig.cmake
  ${CMAKE_CURRENT_BINARY_DIR}/robot_identification_pinocchioConfigVersion.cmake
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/robot_identification_pinocchio)
install(FILES README.md ../LICENSE DESTINATION ${CMAKE_INSTALL_DATADIR}/robot_identification_pinocchio)

include(CTest)
if(BUILD_TESTING)
  find_package(Threads REQUIRED)
  add_executable(pinocchio_regressor_test test/pinocchio_regressor_test.cpp)
  target_link_libraries(pinocchio_regressor_test PRIVATE robot_identification::pinocchio Threads::Threads)
  add_test(NAME pinocchio_regressor_test COMMAND pinocchio_regressor_test)

  if(robot_identification_VERSION VERSION_GREATER_EQUAL "0.2.0")
    add_executable(pinocchio_design_test test/pinocchio_design_test.cpp)
    target_link_libraries(pinocchio_design_test PRIVATE robot_identification::pinocchio)
    add_test(NAME pinocchio_design_test COMMAND pinocchio_design_test)
  endif()

  set(PINOCCHIO_CONSUMER_INSTALL "${CMAKE_CURRENT_BINARY_DIR}/consumer-install")
  set(PINOCCHIO_CONSUMER_BUILD "${CMAKE_CURRENT_BINARY_DIR}/consumer-build")
  add_test(NAME robot_identification_pinocchio_install
    COMMAND ${CMAKE_COMMAND} --install ${CMAKE_CURRENT_BINARY_DIR}
      --prefix ${PINOCCHIO_CONSUMER_INSTALL})
  add_test(NAME robot_identification_pinocchio_consumer_configure
    COMMAND ${CMAKE_COMMAND}
      -S ${CMAKE_CURRENT_SOURCE_DIR}/test/consumer
      -B ${PINOCCHIO_CONSUMER_BUILD}
      "-DCMAKE_PREFIX_PATH=${PINOCCHIO_CONSUMER_INSTALL};${CMAKE_PREFIX_PATH}"
      "-DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}"
      "-DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}"
      "-DCMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS}"
      "-DCMAKE_EXE_LINKER_FLAGS=${CMAKE_EXE_LINKER_FLAGS}")
  add_test(NAME robot_identification_pinocchio_consumer_build
    COMMAND ${CMAKE_COMMAND} --build ${PINOCCHIO_CONSUMER_BUILD})
  add_test(NAME robot_identification_pinocchio_consumer_run
    COMMAND ${PINOCCHIO_CONSUMER_BUILD}/robot_identification_pinocchio_consumer)
  set_tests_properties(robot_identification_pinocchio_consumer_configure PROPERTIES
    DEPENDS robot_identification_pinocchio_install)
  set_tests_properties(robot_identification_pinocchio_consumer_build PROPERTIES
    DEPENDS robot_identification_pinocchio_consumer_configure)
  set_tests_properties(robot_identification_pinocchio_consumer_run PROPERTIES
    DEPENDS robot_identification_pinocchio_consumer_build)
endif()
