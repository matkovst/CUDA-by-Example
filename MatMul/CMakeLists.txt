cmake_minimum_required(VERSION 3.15)
project( matmul )

add_executable( ${PROJECT_NAME} main.cu )
set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries( ${PROJECT_NAME} ${OpenCV_LIBS} )

install(TARGETS ${PROJECT_NAME} CONFIGURATIONS Release RUNTIME DESTINATION ${CMAKE_SOURCE_DIR}/bin)