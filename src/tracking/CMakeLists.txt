add_library(${PROJECT_NAME}_tracking
##tracker.cpp
point_cloud_pyramid.cpp
tracking_optimizer.cpp
)

target_link_libraries(${PROJECT_NAME}_tracking
${EIGEN_LIBRARIES}
/usr/lib/x86_64-linux-gnu/libglog.so
${PROJECT_NAME}_base
)