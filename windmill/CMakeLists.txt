cmake_minimum_required(VERSION 3.10)
project(windmill VERSION 0.1.0)

# 设置 C++ 标准和编译选项
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# -pthread 用于启用多线程支持
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")

# ----------------- 1. 配置 MVS SDK 路径 -----------------
# 请根据你的实际安装路径修改下面这两行
set(MVS_INCLUDE_DIR "/opt/MVS/include")
set(MVS_LIB_DIR "/opt/MVS/lib/64")
# -----------------------------------------------------------

# 查找 OpenCV 包
find_package(OpenCV REQUIRED)

# 查找 MVS 库文件 (libMvCameraControl.so)
find_library(MVS_LIBRARY
    NAMES MvCameraControl  # 自动添加 "lib" 前缀和 ".so" 后缀
    HINTS ${MVS_LIB_DIR}   # 指定的库目录中查找
)

# 检查是否找到了 MVS 库，如果没找到就报错停止
if(NOT MVS_LIBRARY)
    message(FATAL_ERROR "MVS library (libMvCameraControl.so) not found in ${MVS_LIB_DIR}. Please check the path.")
endif()



add_library(windmill STATIC
    src/WMIdentify.cpp
    src/WMPredict.cpp
    src/traditional_detection.cpp
    src/WMInference.cpp
    src/globalParam.cpp
)

# 目标 (windmill) 添加头文件搜索路径
target_include_directories(windmill PUBLIC
    # 项目内部的 include 目录
    ${CMAKE_CURRENT_SOURCE_DIR}/include

    # OpenCV 的 include 目录
    ${OpenCV_INCLUDE_DIRS}

    # MVS SDK 的 include 目录
    ${MVS_INCLUDE_DIR}
)


# 将可执行文件链接到所有需要的库
target_link_libraries(windmill PUBLIC
    # 链接 OpenCV 库
    ${OpenCV_LIBS}

    # === 这是链接 MVS 库的关键 ===
    ${MVS_LIBRARY}
)

