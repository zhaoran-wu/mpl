add_library(util STATIC
trajectory.cpp
utility.cpp
ply.cpp
tictoc.cpp
)
target_link_libraries(util
${Eigen_LIBRARIES}
yaml-cpp
${Boost_LIBRARIES}
${OpenCV_LIBRARIES}
/usr/lib/x86_64-linux-gnu/libglog.so
)
