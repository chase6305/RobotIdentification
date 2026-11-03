cmake_minimum_required(VERSION 3.15)
project(robot_identification VERSION 0.2.12 LANGUAGES CXX)

option(ROBOT_IDENTIFICATION_BUILD_EXAMPLES "Build standalone identification examples" OFF)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)
find_package(Eigen3 3.3.1 REQUIRED NO_MODULE)

add_library(robot_identification src/identification.cpp src/base_parameters.cpp src/experiment_design.cpp)
add_library(robot_identification::robot_identification ALIAS robot_identification)
target_compile_features(robot_identification PUBLIC cxx_std_17)
target_link_libraries(robot_identification PUBLIC Eigen3::Eigen)
target_include_directories(robot_identification PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(robot_identification PRIVATE -Wall -Wextra -Wpedantic)
endif()

install(TARGETS robot_identification
  EXPORT robot_identificationTargets
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(EXPORT robot_identificationTargets
  FILE robot_identificationTargets.cmake
  NAMESPACE robot_identification::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/robot_identification)
configure_package_config_file(
  cmake/robot_identificationConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/robot_identificationConfig.cmake
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/robot_identification)
write_basic_package_version_file(
  ${CMAKE_CURRENT_BINARY_DIR}/robot_identificationConfigVersion.cmake
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY SameMajorVersion)
install(FILES
  ${CMAKE_CURRENT_BINARY_DIR}/robot_identificationConfig.cmake
  ${CMAKE_CURRENT_BINARY_DIR}/robot_identificationConfigVersion.cmake
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/robot_identification)
install(FILES README.md ../LICENSE DESTINATION ${CMAKE_INSTALL_DATADIR}/robot_identification)
install(DIRECTORY schema/ DESTINATION ${CMAKE_INSTALL_DATADIR}/robot_identification/schema)

include(CTest)
if(BUILD_TESTING)
  foreach(name IN ITEMS base_parameters experiment_design covariance preprocessing trajectory csv)
    add_executable(robot_identification_${name}_test test/${name}_test.cpp)
    target_link_libraries(robot_identification_${name}_test PRIVATE robot_identification::robot_identification)
    add_test(NAME robot_identification_${name}_test COMMAND robot_identification_${name}_test)
  endforeach()

  add_executable(robot_identification_assembly_test test/regression_assembly_test.cpp)
  target_link_libraries(robot_identification_assembly_test PRIVATE robot_identification::robot_identification)
  add_test(NAME robot_identification_assembly_test COMMAND robot_identification_assembly_test)

  add_executable(robot_identification_test test/identification_test.cpp)
  target_link_libraries(robot_identification_test PRIVATE robot_identification::robot_identification)
  add_test(NAME robot_identification_test COMMAND robot_identification_test)

  # Interchange tests use the repository's pure-Python tools, without ROS.
  # A standalone copy of this library can still build its own C++ tests.
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../robot_identification_tools/robot_identification_tools/dataset_inspect.py"
     AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../robot_identification_ros2/robot_identification_ros2/raw_joint_state_session.py")
    find_package(Python3 QUIET COMPONENTS Interpreter)
    if(Python3_Interpreter_FOUND)
      add_executable(robot_identification_csv_roundtrip test/csv_roundtrip.cpp)
      target_link_libraries(robot_identification_csv_roundtrip PRIVATE robot_identification::robot_identification)
      add_test(NAME robot_identification_csv_interop
        COMMAND ${Python3_EXECUTABLE} -B ${CMAKE_CURRENT_SOURCE_DIR}/test/csv_interop_test.py
          $<TARGET_FILE:robot_identification_csv_roundtrip>)
    endif()
  endif()

  set(ROBOT_IDENTIFICATION_CONSUMER_INSTALL "${CMAKE_CURRENT_BINARY_DIR}/consumer-install")
  set(ROBOT_IDENTIFICATION_CONSUMER_BUILD "${CMAKE_CURRENT_BINARY_DIR}/consumer-build")
  add_test(NAME robot_identification_install
    COMMAND ${CMAKE_COMMAND} --install ${CMAKE_CURRENT_BINARY_DIR} --prefix ${ROBOT_IDENTIFICATION_CONSUMER_INSTALL})
  add_test(NAME robot_identification_consumer_configure
    COMMAND ${CMAKE_COMMAND}
      -S ${CMAKE_CURRENT_SOURCE_DIR}/test/consumer
      -B ${ROBOT_IDENTIFICATION_CONSUMER_BUILD}
      "-DCMAKE_PREFIX_PATH=${ROBOT_IDENTIFICATION_CONSUMER_INSTALL};${CMAKE_PREFIX_PATH}"
      "-DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}"
      "-DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}"
      "-DCMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS}"
      "-DCMAKE_EXE_LINKER_FLAGS=${CMAKE_EXE_LINKER_FLAGS}")
  add_test(NAME robot_identification_consumer_build
    COMMAND ${CMAKE_COMMAND} --build ${ROBOT_IDENTIFICATION_CONSUMER_BUILD})
  add_test(NAME robot_identification_consumer_run
    COMMAND ${ROBOT_IDENTIFICATION_CONSUMER_BUILD}/robot_identification_consumer)
  set_tests_properties(robot_identification_consumer_configure PROPERTIES
    DEPENDS robot_identification_install)
  set_tests_properties(robot_identification_consumer_build PROPERTIES
    DEPENDS robot_identification_consumer_configure)
  set_tests_properties(robot_identification_consumer_run PROPERTIES
    DEPENDS robot_identification_consumer_build)
endif()

if(ROBOT_IDENTIFICATION_BUILD_EXAMPLES)
  foreach(name IN ITEMS generic_regressor_example identification_design_example)
    add_executable(${name} examples/${name}.cpp)
    target_link_libraries(${name} PRIVATE robot_identification::robot_identification)
    install(TARGETS ${name} RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
    if(BUILD_TESTING)
      add_test(NAME ${name}_runs COMMAND ${name})
      set_tests_properties(${name}_runs PROPERTIES LABELS "example")
    endif()
  endforeach()
endif()
