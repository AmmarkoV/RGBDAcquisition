cmake_minimum_required(VERSION 2.8.3)
project(camera_broadcast)
 
#We are using a stripped acquisition
add_definitions(-DALL_HEADERS_IN_SAME_DIR)
 
if(NOT OpenCV_DIR)
 find_path(OpenCV_DIR "OpenCVConfig.cmake" DOC "Root directory of OpenCV" HINTS "/opt/ros/noetic/share/OpenCV/")
endif(NOT OpenCV_DIR)
FIND_PACKAGE(OpenCV REQUIRED)

#Force TF2
add_definitions(-DUSE_TENSORFLOW2)

#Force INTEL/AMD build.. ( wont work on ARM )  
#add_definitions(-DINTEL_OPTIMIZATIONS)
#set(CMAKE_C_FLAGS "${CMAKE_CXX_FLAGS} -msse3")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -msse2")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -msse3")

set(CAMERA_BROADCAST_LIBRARIES  
    png
    jpeg 
   ) 

find_package(catkin REQUIRED COMPONENTS 
  sensor_msgs 
  OpenCV 
  sensor_msgs
  cv_bridge 
  roscpp 
  std_msgs 
  genmsg 
  image_transport 
  geometry_msgs 
  roslib 
  tf)
  
#Messages -----------------------------------
#add_message_files(DIRECTORY msg FILES bvhSkeleton.msg )

#Services -----------------------------------
add_service_files(DIRECTORY srv FILES singleFloat.srv )

## Generate added messages and services
generate_messages(DEPENDENCIES std_msgs camera_broadcast)

#This needs to be after generate messages
catkin_package()

include_directories( ${catkin_INCLUDEDIR} ${catkin_INCLUDE_DIRS} ${OpenCV_INCLUDEDIR})
add_executable(camera_broadcast  src/main.cpp src/jpgInput.c src/AmmClient.c src/network.c src/protocol.c src/tools.c)
target_link_libraries(camera_broadcast  ${catkin_LIBRARIES} ${OpenCV_LIBRARIES}  ${CAMERA_BROADCAST_LIBRARIES} )

add_dependencies(${PROJECT_NAME} ${PROJECT_NAME}_gencpp)

install(TARGETS camera_broadcast RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})
#install(FILES camera_broadcast.xml DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})


