cmake_minimum_required(VERSION 3.15)
project( dot )

add_executable( ${PROJECT_NAME} main.cu )
set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.18.0" AND NOT CUDA_ARCHITECTURES)
    set_property(TARGET ${PROJECT_NAME} PROPERTY CUDA_ARCHITECTURES OFF)
endif()

target_link_libraries( ${PROJECT_NAME} ${OpenCV_LIBS} )

install(TARGETS ${PROJECT_NAME} CONFIGURATIONS Release RUNTIME DESTINATION ${CMAKE_SOURCE_DIR}/bin)