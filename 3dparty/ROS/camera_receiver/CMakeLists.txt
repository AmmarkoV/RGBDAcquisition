cmake_minimum_required(VERSION 2.8.3)
project(camera_receiver)
 
#We are using a stripped acquisition
add_definitions(-DALL_HEADERS_IN_SAME_DIR)
 
if(NOT OpenCV_DIR)
 find_path(OpenCV_DIR "OpenCVConfig.cmake" DOC "Root directory of OpenCV" HINTS    "/opt/ros/kinetic/share/OpenCV/")
endif(NOT OpenCV_DIR)
FIND_PACKAGE(OpenCV REQUIRED)

set(CAMERA_RECEIVER_LIBRARIES  
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
add_message_files(DIRECTORY msg FILES a.msg )

#Services -----------------------------------
add_service_files(DIRECTORY srv FILES SetScale.srv )

## Generate added messages and services
generate_messages(DEPENDENCIES std_msgs camera_receiver)

#This needs to be after generate messages
catkin_package()

include_directories( ${catkin_INCLUDEDIR} ${catkin_INCLUDE_DIRS} ${OpenCV_INCLUDEDIR} )
add_executable(camera_receiver  src/main.cpp  src/jpgInput.c src/AmmClient.c src/network.c src/protocol.c src/tools.c )
target_link_libraries(camera_receiver ${catkin_LIBRARIES} ${OpenCV_LIBRARIES}  ${CAMERA_RECEIVER_LIBRARIES})

add_dependencies(${PROJECT_NAME} ${PROJECT_NAME}_gencpp)

install(TARGETS camera_receiver RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})
#install(FILES camera_receiver.xml DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})


