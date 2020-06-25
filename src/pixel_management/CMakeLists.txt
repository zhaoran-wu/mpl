add_library(${PROJECT_NAME}_pixel_management
pixel_selector.cpp
candidate_manager.cpp
)

target_link_libraries(${PROJECT_NAME}_pixel_management
util
${PROJECT_NAME}_base
)