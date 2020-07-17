add_library(${PROJECT_NAME}_visualization
HelloPangolin.cpp
visualizer.cpp
)

target_link_libraries(${PROJECT_NAME}_visualization
${Pangolin_LIBRARIES}
)