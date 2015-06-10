project( Quaternions ) 
cmake_minimum_required( VERSION 2.8.7 )
 

add_executable(EulerToQuaternions eulerToQuaternions.c ../AmMatrix/quaternions.c ) 
target_link_libraries(EulerToQuaternions rt dl m  )
set_target_properties(EulerToQuaternions PROPERTIES DEBUG_POSTFIX "D")   
        
add_executable(QuaternionsToEuler quaternionsToEuler.c ../AmMatrix/quaternions.c ) 
target_link_libraries(QuaternionsToEuler rt dl m  )
set_target_properties(QuaternionsToEuler PROPERTIES DEBUG_POSTFIX "D")   
        
add_executable(RotationBetweenQuaternions rotationBetweenQuaternions.c ../AmMatrix/quaternions.c ) 
target_link_libraries(RotationBetweenQuaternions rt dl m  )
set_target_properties(RotationBetweenQuaternions PROPERTIES DEBUG_POSTFIX "D")   
       
 
add_executable(TestQuaternions testQuaternions.c ../AmMatrix/quaternions.c ) 
target_link_libraries(TestQuaternions rt dl m  )
set_target_properties(TestQuaternions PROPERTIES DEBUG_POSTFIX "D")   
       
 






set_target_properties(EulerToQuaternions PROPERTIES 
                       DEBUG_POSTFIX "D"
                       ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                       LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                       RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                      )

set_target_properties(QuaternionsToEuler PROPERTIES 
                       DEBUG_POSTFIX "D"
                       ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                       LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                       RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                      )

set_target_properties(RotationBetweenQuaternions PROPERTIES 
                       DEBUG_POSTFIX "D"
                       ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                       LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                       RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                      ) 

set_target_properties(TestQuaternions PROPERTIES 
                       DEBUG_POSTFIX "D"
                       ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                       LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                       RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                      )

