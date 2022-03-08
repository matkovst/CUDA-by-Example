cmake_minimum_required(VERSION 3.15)
project( hist )

add_executable( ${PROJECT_NAME} main.cu )
set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries( ${PROJECT_NAME} ${OpenCV_LIBS} ${CMAKE_SOURCE_DIR}/common/lib/glut64.lib )

get_target_property(LINKED_LIBS ${PROJECT_NAME} LINK_LIBRARIES)

install(TARGETS ${PROJECT_NAME} CONFIGURATIONS Release RUNTIME DESTINATION ${CMAKE_SOURCE_DIR}/bin)

file(GLOB GLUT_DLLS ${CMAKE_SOURCE_DIR}/common/bin/*.dll)
add_custom_command(TARGET ${PROJECT_NAME}
                POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                ${GLUT_DLLS}
                ${CMAKE_SOURCE_DIR}/bin)