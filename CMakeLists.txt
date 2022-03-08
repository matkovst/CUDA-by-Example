cmake_minimum_required(VERSION 3.15)
project( CUDA-by-Example )

if (NOT EXISTS ${CMAKE_BINARY_DIR}/CMakeCache.txt)
  if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "" FORCE)
  endif()
endif()

file(MAKE_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)

# OpenCV
find_package( OpenCV REQUIRED )

# CUDA
find_package( CUDA REQUIRED )
if (CUDA_FOUND)
  add_compile_definitions(WITH_CUDA)
  #set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode arch=compute_20,code=sm_20)
  enable_language( CUDA )
  message(STATUS "Found CUDA ${CUDA_VERSION_STRING} at ${CUDA_TOOLKIT_ROOT_DIR}")
endif()

add_subdirectory(DotProduct)
add_subdirectory(GrayScale)
add_subdirectory(Histogram)
add_subdirectory(HeatTransfer)
