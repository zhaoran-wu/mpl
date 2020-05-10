add_library(${PROJECT_NAME}_tracking
tracker.cpp
point_cloud_reference.cpp
)

target_link_libraries(${PROJECT_NAME}_tracking
${EIGEN_LIBRARIES}
/usr/lib/x86_64-linux-gnu/libglog.so
${POEJCET_NAME}_base
)