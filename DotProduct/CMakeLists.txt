cmake_minimum_required(VERSION 3.15)
project( dot )

option(USE_CUDA "Use CUDA" ON)

# OpenCV
find_package( OpenCV REQUIRED )

# CUDA
if (USE_CUDA)
    find_package( CUDA )
    if(CUDA_FOUND)
        add_compile_definitions(WITH_CUDA)
        #set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode arch=compute_20,code=sm_20)
    endif()
endif()

if (USE_CUDA AND CUDA_FOUND)
    enable_language( CUDA )
    add_executable( ${PROJECT_NAME} main.cu )
    set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif()

target_link_libraries( ${PROJECT_NAME} ${OpenCV_LIBS} )

install(TARGETS ${PROJECT_NAME} CONFIGURATIONS Release RUNTIME DESTINATION ${CMAKE_SOURCE_DIR}/bin)

message(STATUS "USE_CUDA: ${USE_CUDA}")
message(STATUS "Found CUDA ${CUDA_VERSION_STRING} at ${CUDA_TOOLKIT_ROOT_DIR}")