cmake_minimum_required(VERSION 3.10)
set(TARGET_NAME infantry_new)
project(${TARGET_NAME})

# ----------------- 1. 设置编译选项和宏定义 -----------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# CXX_FLAGS 会同时作用于 Debug 和 Release 模式
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")

# (可选) 为 Release 模式单独设置优化选项
# set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")

# --- 添加你的宏定义 ---
add_definitions(-DSHOW_FPS -DDEBUGMODE -DVISUALIZE_PREDICTION -DFYT -DSENDCAMERA)
#add_definitions(-DVIRTUALGRAB) # 如果用视频，取消这行注释
#add_definitions(-DNOPORT) # 如果不用串口，取消这行注释

# ----------------- 2. 查找所有依赖包 (核心修改) -----------------

# --- 查找 OpenCV ---
# 让 CMake 自动查找，并明确指定所有需要的模块
find_package(OpenCV REQUIRED COMPONENTS
    core
    highgui
    imgproc
    imgcodecs
    videoio
    calib3d
)
# 检查是否找到
if(NOT OpenCV_FOUND)
    message(FATAL_ERROR "OpenCV not found!")
endif()
message(STATUS "Found OpenCV version: ${OpenCV_VERSION}")

# --- 查找其他库 ---
# find_package(OpenVINO REQUIRED COMPONENTS Runtime) # 你的 windmill 不需要，先注释掉
find_package(Ceres REQUIRED)
find_package(Eigen3 REQUIRED)

# ----------------- 3. 设置头文件和库文件搜索路径 -----------------

include_directories(
    # 项目内部的 include 文件夹
    armor/include
    camera/include
    params/include
    serialPort/include
    windmill/include

    # 外部库的 include 文件夹
    /opt/MVS/include
    ${CERES_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
    ${OpenCV_INCLUDE_DIRS} # 由 find_package 自动提供
    # ${OpenVINO_INCLUDE_DIRS} # 暂时注释
    /usr/local/cuda/include
)

link_directories(
    /opt/MVS/lib/64
    # (如果你的系统是32位，才需要下面这个)
    # /opt/MVS/lib/32
)

# ----------------- 4. 添加子项目 -----------------
# 这些子项目会被编译成静态库
add_subdirectory(camera)
add_subdirectory(params)
add_subdirectory(serialPort)
add_subdirectory(windmill)
add_subdirectory(armor)

# ----------------- 5. 生成最终的可执行文件 -----------------
add_executable(infantry_new main.cpp)

# ----------------- 6. 链接所有库到可执行文件 (核心修改) -----------------
target_link_libraries(infantry_new
    # 链接我们自己编译的子库
    params
    camera
    windmill
    armor
    serialPort

    # 链接外部库
    ${CERES_LIBRARIES}
    # ${InferenceEngine_LIBRARIES} # OpenVINO 的库，先注释
    ${OpenCV_LIBS}     # 使用由 find_package(COMPONENTS...) 生成的完整列表
    MvCameraControl    # CMake 会自动在 link_directories 中寻找 libMvCameraControl.so
    glog               # 假设系统已安装 glog，CMake 能找到
    # openvino::runtime # OpenVINO 的另一种链接方式，先注释
)
    # # include 部分
    # include_directories(/usr/local/include/apriltag)
    # link_directories(/usr/local/lib)

    # add_subdirectory(apriltag)
    # # 链接部分
    # # 链接官方库和本功能包
    # target_link_libraries(${PROJECT_NAME} libapriltag.so apriltag)