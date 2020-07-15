add_library(${PROJECT_NAME}_base
config.cpp
cam_data.cpp
image_pyramid.cpp
frame.cpp
)

target_link_libraries(${PROJECT_NAME}_base
util
)