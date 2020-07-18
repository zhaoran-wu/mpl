add_library(${PROJECT_NAME}_visualizer
visualizer.cpp
)

target_link_libraries(${PROJECT_NAME}_visualizer
${Pangolin_LIBRARIES}
)