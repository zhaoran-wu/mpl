add_library(${PROJECT_NAME}_base
config.cpp
cam_data.cpp
image_pyramid.cpp
pixel_selector.cpp
affine_light.cpp
frame.cpp
)

target_link_libraries(${PROJECT_NAME}_base
util
)