    cmake_minimum_required(VERSION 3.0.0)
    set(CMAKE_CXX_STANDARD 17)
    set(TARGET_NAME infantry_new)
    project(${TARGET_NAME})
    set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
    # 开启多线程
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -pthread")
    SET(CMAKE_CXX_FLAGS_RELEASE "-std=c++17 -pthread")
    find_package(OpenVINO REQUIRED COMPONENTS Runtime)
    find_package(Ceres REQUIRED)
    find_package(Eigen3 REQUIRED)
    include_directories(${EIGEN3_INCLUDE_DIR})
    set(OpenCV_DIR /usr/local/lib/cmake/opencv4)
    find_package(OpenCV 4 QUIET)
    # aim debug
    # add_definitions(-DAPRILTAG)
    # add_definitions(-DPOSESTIMATION)
    add_definitions(-DSHOW_FPS)
    # add_definitions(-DDEBUGREFINE)
    add_definitions(-DDEBUGMODE)
    add_definitions(-DVISUALIZE_PREDICTION)
    # add_definitions(-DDEBUGCOLOR)
    # add_definitions(-DDEBUGNUM)
    # add_definitions(-DVIRTUALGRAB)  # 打开虚拟取流 即用视频跑
    # add_definitions(-DNOPORT)    
    # add_definitions(-DUSETRIGGERMODE)
    # add_definitions(-DTHREADANALYSIS)
    # add_definitions(-DDRONE)
    # 使用UI动态调参
    # add_definitions(-DDEBUGHIT)
    # add_definitions(-DDILATED)
    add_definitions(-DFYT)
    # add_definitions(-DCORNER)

    add_definitions(-DSENDCAMERA)
    # add_definitions(-DRECORDPIG)
    # 使用神经网络识别
    # add_definitions(-DUSEWMNET)
    # 记录视频
    # add_definitions(-DRECORDVIDEO)
    include_directories(
        /opt/MVS/include
        armor/include
        camera/include
        params/include
        serialPort/include
        windmill/include
        ${CERES_INCLUDE_DIRS}
        ${OpenCV_INCLUDE_DIRS}
        ${OpenVINO_INCLUDE_DIRS}
        /usr/local/cuda/include
        
        # /usr/local/include/apriltag
        # apriltag/include
        )
    link_directories(
        /opt/MVS/lib/64
        /opt/MVS/lib/32
        /usr/local/lib
        )
    add_subdirectory(camera)
    add_subdirectory(params)
    add_subdirectory(serialPort)
    add_subdirectory(windmill)
    add_subdirectory(armor)
    # add_subdirectory(apriltag)
    add_executable(infantry_new main.cpp)
    target_link_libraries(infantry_new
        params
        camera
        serialPort
        windmill
        armor
        # apriltag
        ${CERES_LIBRARIES}
        ${InferenceEngine_LIBRARIES}
        ${OpenCV_LIBS}
        MvCameraControl
        libMvCameraControl.so
        libglog.so
        openvino::runtime
        # libapriltag.so 
        )
    set(CPACK_PROJECT_NAME ${PROJECT_NAME})
    set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
    include(CPack)
    # # include 部分
    # include_directories(/usr/local/include/apriltag)
    # link_directories(/usr/local/lib)

    # add_subdirectory(apriltag)
    # # 链接部分
    # # 链接官方库和本功能包
    # target_link_libraries(${PROJECT_NAME} libapriltag.so apriltag)