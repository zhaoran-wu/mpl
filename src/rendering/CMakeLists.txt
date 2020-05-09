add_library(${PROJECT_NAME}_rendering
mesh.cpp
shader.cpp
synetic_image.cpp
)

target_link_libraries(${PROJECT_NAME}_rendering
${EIGEN_LIBRARIES}
GLAD
/usr/lib/x86_64-linux-gnu/libglog.so
glfw
${OpenCV_LIBRARIES}
util
)