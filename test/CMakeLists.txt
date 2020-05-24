
add_executable(test_mesh test_mesh.cpp)
target_link_libraries(test_mesh 
util
GLAD
/usr/lib/x86_64-linux-gnu/libglog.so
${OpenCV_LIBRARIES}
glfw
${PROJECT_NAME}_rendering)

add_executable(test_synetic_image test_synetic_image.cpp)
target_link_libraries(test_synetic_image
util
GLAD
/usr/lib/x86_64-linux-gnu/libglog.so
glfw
${PROJECT_NAME}_rendering
)

add_executable(test_ply test_read_ply.cpp)
target_link_libraries(test_ply
util
GLAD
/usr/lib/x86_64-linux-gnu/libglog.so
glfw
${PROJECT_NAME}_rendering
)


add_executable(test_direct_simple test_direct_simple.cpp)
target_link_libraries(test_direct_simple
${CERES_LIBRARIES}
${OpenCV_LIBRARIES}
${Eigen_LIBRARIES}
${Boost_LIBRARIES}
${Sophus_LIBRARIS}
util
${PROJECT_NAME}_base
${PROJECT_NAME}_pixel_management
)

add_executable(test_pyramid test_pyramid.cpp)
target_link_libraries(test_pyramid
${CERES_LIBRARIES}
${OpenCV_LIBRARIES}
${Eigen_LIBRARIES}
${Boost_LIBRARIES}
util
${PROJECT_NAME}_base

)
add_executable(test_selector test_pixel_selector.cpp)
target_link_libraries(test_selector
${CERES_LIBRARIES}
${OpenCV_LIBRARIES}
${Eigen_LIBRARIES}
${Boost_LIBRARIES}
util
${PROJECT_NAME}_pixel_management
)
add_executable(test_tracker test_tracker.cpp)
target_link_libraries(test_tracker
${CERES_LIBRARIES}
${OpenCV_LIBRARIES}
${Eigen_LIBRARIES}
${Boost_LIBRARIES}
util
${PROJECT_NAME}_base
${PROJECT_NAME}_tracking
${PROJECT_NAME}_pixel_management
)

add_executable(test_tracking_optimizer test_tracking_optimizer.cpp)
target_link_libraries(test_tracking_optimizer
${CERES_LIBRARIES}
${OpenCV_LIBRARIES}
${Eigen_LIBRARIES}
${Boost_LIBRARIES}
util
${PROJECT_NAME}_base
${PROJECT_NAME}_tracking
${PROJECT_NAME}_pixel_management
)


#add_executable(test_main test_main.cpp)
#target_link_libraries(test_main
#${CERES_LIBRARIES}
#${OpenCV_LIBRARIES}
#${Eigen_LIBRARIES}
#${Boost_LIBRARIES}
#util
#${PROJECT_NAME}_base
#${PROJECT_NAME}_tracking
#${PROJECT_NAME}_rendering
#${PROJECT_NAME}_window_optimization
#)