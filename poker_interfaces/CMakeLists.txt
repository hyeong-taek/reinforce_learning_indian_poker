cmake_minimum_required(VERSION 3.8)
project(poker_interfaces)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)

# 액션이 암묵적으로 참조하는 인터페이스들
find_package(builtin_interfaces REQUIRED)
find_package(unique_identifier_msgs REQUIRED)
find_package(action_msgs REQUIRED)
find_package(std_msgs REQUIRED)

set(msg_files
  "msg/Voice/HumanIntent.msg"
)

set(action_files 
  "action/Robot/ShowCard.action"
  "action/Robot/BetChips.action"
  "action/Robot/MoveNamedPose.action"
  "action/Robot/PickCard.action"
  "action/Camera/DetectCardOnce.action"
  "action/Camera/ObserveGestures.action"
  "action/Camera/DetectChip.action"
)

rosidl_generate_interfaces(${PROJECT_NAME}
${msg_files}  
${action_files}
  DEPENDENCIES 
    builtin_interfaces 
    unique_identifier_msgs 
    action_msgs
    std_msgs

)

ament_export_dependencies(
  rosidl_default_runtime
  builtin_interfaces
  unique_identifier_msgs
  action_msgs
  std_msgs
)

ament_package()
