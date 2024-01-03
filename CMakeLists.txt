cmake_minimum_required(VERSION 3.0.0)
project(lanedetection)

find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

add_executable(lanedetection main.cpp src/LaneDetection/detect.cpp src/LaneDetection/detect.hpp src/LaneDetection/helpers.hpp)


target_link_libraries( lanedetection ${OpenCV_LIBS} )
