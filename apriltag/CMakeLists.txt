cmake_minimum_required(VERSION 3.0.0)
project(apriltag VERSION 0.1.0)
set(CMAKE_CXX_STANDARD 17)

include_directories(include)
aux_source_directory(./src DIR_SRCS)

add_library(apriltag STATIC ${DIR_SRCS})