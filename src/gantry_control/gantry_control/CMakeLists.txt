cmake_minimum_required(VERSION 3.8)
project(gantry_control)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(std_srvs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(rosidl_default_generators REQUIRED)

# Generate custom messages and services
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/GantryState.msg"
  "msg/TrajCmd.msg"
  "srv/SetMode.srv"
  "srv/MoveTo.srv"
  DEPENDENCIES std_msgs
)

# sFoundation library (installed at /usr/local/lib)
set(SFOUNDATION_INC_DIR "$ENV{HOME}/teknic/linux/Linux_Software/sFoundation/inc/inc-pub")
find_library(SFOUNDATION_LIB sFoundation20
  PATHS /usr/local/lib
  REQUIRED
)

# ── Gantry Controller Node ──
add_executable(gantry_controller
  src/gantry_controller.cpp
)

target_include_directories(gantry_controller PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${SFOUNDATION_INC_DIR}
)

target_link_libraries(gantry_controller
  ${SFOUNDATION_LIB}
)

ament_target_dependencies(gantry_controller
  rclcpp
  std_msgs
  std_srvs
  geometry_msgs
  nav_msgs
  sensor_msgs
  tf2_ros
)

# Link with generated message/service headers
rosidl_get_typesupport_target(cpp_typesupport_target ${PROJECT_NAME} rosidl_typesupport_cpp)
target_link_libraries(gantry_controller "${cpp_typesupport_target}")

# ── Install ──
install(TARGETS
  gantry_controller
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY
  launch config docs web
  DESTINATION share/${PROJECT_NAME}
)

install(PROGRAMS
  scripts/gantry_cli.py
  scripts/traj_player.py
  scripts/traj_stream_example.py
  scripts/adaptive_id_player.py
  scripts/adaptive_shaper_player.py
  scripts/adaptive_two_phase_player.py
  scripts/adaptive_paper_tdf_player.py
  scripts/adaptive_step_is_player.py
  scripts/paper_test_campaign.py
  scripts/paper_single_test.py
  scripts/plot_residual_energy.py
  scripts/jog_position_verify.py
  scripts/crane_dashboard_server.py
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY
  include/
  DESTINATION include
)

ament_package()
