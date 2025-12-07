cmake_minimum_required(VERSION 3.0.0)
project(camera VERSION 0.1.0)
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -pthread")

aux_source_directory(./src ALL_SRCS)
add_library(camera STATIC ${ALL_SRCS})