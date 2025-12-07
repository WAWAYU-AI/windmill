# Apriltag 识别库

本功能包提供了一个基于AprilTag的识别库，可以用于识别AprilTag标签中的 tag36h11 类标签，获得标签相对相机的位置和旋转角度。

## 安装

本功能包需要 OpenCV 和 apriltag 识别库的支持。
请按照如下步骤安装 apriltag 识别库：

1. 下载 apriltag 识别库源码：

`git clone https://github.com/AprilRobotics/apriltag.git`

2. 进入 apriltag 目录，打开终端，运行如下命令进行编译并安装：
   
```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target install
```

安装完成后，就可以编译本功能包了。

## 使用

请在父功能包的 CMakeLists.txt 中添加如下内容：

```cmake
# include 部分
include_directories(/usr/local/include/apriltag)
link_directories(/usr/local/lib)

add_subdirectory(apriltag)
# 链接部分
# 链接官方库和本功能包
target_link_libraries(${PROJECT_NAME} libapriltag.so apriltag)
```