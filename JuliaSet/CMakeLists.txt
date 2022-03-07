cmake_minimum_required(VERSION 3.15)
project( julia )

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
    add_executable( julia main.cu )
    set_target_properties(julia PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif()

target_link_libraries( julia ${OpenCV_LIBS} )

message(STATUS "USE_CUDA: ${USE_CUDA}")
message(STATUS "Found CUDA ${CUDA_VERSION_STRING} at ${CUDA_TOOLKIT_ROOT_DIR}")