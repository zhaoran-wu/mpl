add_library(${PROJECT_NAME}_window_optimization
ba_residual_block.cpp
bundle_adjustment.cpp
./ceres/FrameParameterBlock.cpp
./ceres/PhotometricBA.cpp
./ceres/PhotometricBAConfig.cpp 
./ceres/PhotometricBAIterationCallback.cpp 
./ceres/PhotometricResidual.cpp 
./ceres/PointParameterBlock.cpp 
)

target_link_libraries(${PROJECT_NAME}_window_optimization
${EIGEN_LIBRARIES}
${CERES_LIBRARIES}
/usr/lib/x86_64-linux-gnu/libglog.so
${PROJECT_NAME}_base
)
