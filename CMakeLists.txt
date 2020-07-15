cmake_minimum_required(VERSION 3.1)

project(mpl)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_SOURCE_DIR}/cmake)

if(NOT CMAKE_BUILD_TYPE)
	set(CMAKE_BUILD_TYPE RELEASE)
endif()

# gprof
OPTION(USE_GPROF  "Use gprof"  FALSE)

if(USE_GPROF)
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pg")
    SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -pg")
    SET(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -pg")
endif(USE_GPROF)


set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

set(Sophus_DIR /usr/local/include/sophus)

find_package(Eigen3 REQUIRED)
find_package(glfw3 REQUIRED)
find_package(OpenCV 3.2 REQUIRED)
find_package(Boost COMPONENTS system filesystem program_options)
find_package(Ceres REQUIRED)
find_package(Sophus REQUIRED)

include_directories(/usr/include/eigen3)
include_directories(${OpenCV_INCLUDE_DIR})
include_directories(${CMAKE_SOURCE_DIR}/third)
include_directories(${CMAKE_SOURCE_DIR}/src/rendering)
include_directories(${CMAKE_SOURCE_DIR}/src/base)
include_directories(${CMAKE_SOURCE_DIR}/src/pixel_management)
include_directories(${CMAKE_SOURCE_DIR}/src/window_optimization)
include_directories(${CMAKE_SOURCE_DIR}/src/tracking)



include_directories(${CMAKE_SOURCE_DIE}/util)
include_directories(${Boost_INCLUDE_DIRS})
include_directories(${CERES_INCLUDE_DIRS})
include_directories(${Sophus_DIR})

add_library(GLAD third/glad/glad.c)
target_link_libraries(GLAD PUBLIC ${CMAKE_DL_LIBS})
#
#
add_subdirectory(util)
add_subdirectory(src/rendering)
add_subdirectory(src/base)
add_subdirectory(src/tracking)
add_subdirectory(src/window_optimization)
add_subdirectory(src/pixel_management)
add_subdirectory(test)
