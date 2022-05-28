project( bvhtransform )
cmake_minimum_required( VERSION 2.8.7 )
 
#Make fast, lean and platform independent binaries..
set(CMAKE_CXX_FLAGS "-s -O3 -fPIC -march=native -mtune=native")
set(CMAKE_C_FLAGS "-s -O3 -fPIC -march=native -mtune=native") 

OPTION(OPENCL_OPTIMIZATIONS OFF)

if (OPENCL_OPTIMIZATIONS)
add_definitions(-DOPENCL_OPTIMIZATIONS)
endif(OPENCL_OPTIMIZATIONS)

set_source_files_properties(mat_mul.cl PROPERTIES HEADER_FILE_ONLY TRUE)
#add_library(bvhtransform STATIC mat_mul.c)
add_executable(bvhtransform bvh_transform_ocl.c
               openCLTools.c
               ../bvh_loader.c
               ../calculate/bvh_transform.c 
               ../calculate/bvh_project.c 
               ../calculate/smoothing.c
               ../calculate/bvh_to_tri_pose.c
               ../import/fromBVH.c
               ../export/bvh_export.c
               ../export/bvh_to_c.c
               ../export/bvh_to_bvh.c
               ../export/bvh_to_svg.c
               ../export/bvh_to_csv.c
               ../export/bvh_to_json.c
               ../export/bvh_to_trajectoryParserTRI.c
               ../export/bvh_to_trajectoryParserPrimitives.c
               ../edit/bvh_randomize.c 
               ../edit/bvh_rename.c 
               ../edit/bvh_remapangles.c 
               ../edit/bvh_interpolate.c 
               ../edit/bvh_merge.c 
               ../edit/bvh_filter.c 
               ../edit/bvh_cut_paste.c 
               ../ik/bvh_inverseKinematics.c 
               ../ik/hardcodedProblems_inverseKinematics.c
               #../ik/levmar.c
               ../metrics/bvh_measure.c
               ../tests/test.c
               ../../../Library/TrajectoryParser/InputParser_C.c
               ../../../../../../tools/AmMatrix/matrix4x4Tools.c
               ../../../../../../tools/AmMatrix/matrixMultiplicationOptimization.c
               ../../../../../../tools/AmMatrix/matrixOpenGL.c
               ../../../../../../tools/AmMatrix/quaternions.c
               ../../../../../../tools/AmMatrix/simpleRenderer.c 
               )  
target_link_libraries(bvhtransform m rt OpenCL pthread )


set_target_properties(bvhtransform PROPERTIES
                       DEBUG_POSTFIX "D"
                       ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                       LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                       RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                      )
