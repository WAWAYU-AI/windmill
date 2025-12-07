# 2024年笃行战队视觉组代码文档

# 简介

本文档主要介绍代码结构、代码流程、代码内容、算法原理、调参细节与其他内容。

# 环境依赖

1. MVS，可以前往[HIKROBOT官网](https://www.hikrobotics.com/cn/machinevision/service/download?module=0)找到Linux版本的MVS进行下载，之后解压压缩包，运行压缩包中的setup.sh即可。
2. glog，`sudo apt-get install libgoogle-glog-dev`。
3. Eigen3，`sudo apt-get install libeigen3-dev`。
4. ceres，通过 `sudo apt-get install`确保已经下载依赖 `libgoogle-glog-dev`、`libgflags-dev`、`libatlas-base-dev`、`libeigen3-dev`、`libsuitesparse-dev`，之后前往github中[ceres主页](https://github.com/ceres-solver/ceres-solver/tags)下载ceres1.1.4，使用cmake编译的方式安装。
5. openvino，下载[官方密钥](https://apt.repos.intel.com/openvino/2021/GPG-PUB-KEY-INTEL-OPENVINO-2021)，之后依次输入以下指令 `sudo apt-key add <PATH_TO_DOWNLOADED_GPG_KEY>`，`echo "deb https://apt.repos.intel.com/openvino/2021 all main" | sudo tee /etc/apt/sources.list.d/intel-openvino-2021.list`，`sudo apt update`、`sudo apt install intel-openvino-runtime-ubuntu20-2021.4.752`，其中可能在echo这一步骤出错，导致update无法进行，则删除list文件之后使用 `gedit`或者 `vim`指令手动创建，然后输入echo的内容。
6. opencv，使用openvino自带的opencv。为了播放视频，需要安装依赖：`sudo apt-get install gstreamer1.0-libav`

# 代码结构

本代码当前的代码结构如下：

```bash
.
├── AimautoConfig.yaml
├── armor
│   ├── CMakeLists.txt
│   ├── include
│   │   ├── AimAuto.hpp
│   │   ├── detector.hpp
│   │   ├── KalmanFilter.hpp
│   │   ├── KuhnMunkres.hpp
│   │   ├── number_classifier.hpp
│   │   └── tracker.hpp
│   └── src
│       ├── AimAuto.cpp
│       ├── detector.cpp
│       ├── KalmanFilter.cpp
│       ├── number_classifier.cpp
│       └── tracker.cpp
├── autostart.sh
├── camera
│   ├── CMakeLists.txt
│   ├── include
│   │   └── camera.hpp
│   └── src
│       └── camera.cpp
├── CMakeLists.txt
├── config
│   ├── AimautoConfig.yaml
│   ├── CamaraConfig.yaml
│   ├── DetectionConfig.yaml
│   ├── WMConfigBlue.yaml
│   ├── WMConfigBlue（备份）.yaml
│   ├── WMConfigRed.yaml
│   └── WMConfigRed（备份）.yaml
├── main.cpp
├── model
│   ├── label.txt
│   ├── mlp.onnx
│   ├── wm_0524_4n_416.bin
│   ├── wm_0524_4n_416_int8.bin
│   ├── wm_0524_4n_416_int8.xml
│   └── wm_0524_4n_416.xml
├── params
│   ├── CMakeLists.txt
│   ├── include
│   │   ├── common.hpp
│   │   ├── gaoning.hpp
│   │   ├── globalParam.hpp
│   │   ├── monitor.hpp
│   │   └── UIManager.hpp
│   └── src
│       ├── gaoning.cpp
│       ├── globalParam.cpp
│       ├── monitor.cpp
│       └── UIManager.cpp
├── readme.md
├── readme_src
│   ├── code_struct.png
│   └── WM.png
├── recompile.sh
├── restart.sh
├── serialPort
│   ├── CMakeLists.txt
│   ├── include
│   │   ├── MessageManager.hpp
│   │   └── SerialPort.hpp
│   └── src
│       ├── MessageManager.cpp
│       └── SerialPort.cpp
└── setup.sh
```

本代码框架使用嵌套的CMakeLists，即将整体程序分为多个功能包，每个功能包含有include和src文件夹，各自通过CMakeLists进行管理。

## armor

本文件夹为自瞄主体功能包，包含的所有自瞄主体流程

## camera

本文件夹为相机取流功能包，其可以通过句柄打开海康MVS相机，并且可以通过MVS提供的相关API进行取流

## CMakeLists.txt

CMakeLists，用于编译代码，进行相关配置，同时在其中可以修改宏定义

## config

本文件夹中储存yaml格式的配置文件

## log

本文件夹中储存程序运行时输出的日志

## main.cpp

主程序，运行时运行此代码

## model

本文件夹中储存代码中需要的推理使用的模型

## params

本文件夹为功能功能包，包含全局参数结构体、全局地址结构体、功能函数等

## readme.md

本文件，readme文档，介绍整体的代码

## readme_src

本文件夹为readme所需要的资源的文件夹，主要存放图片

## restart.sh

程序执行的脚本，保证程序能找到对应的串口并且连接，同时退出程序之后可以重启程序

## serialPort

本文件夹为串口通信功能包，其可以通过初始化串口，来对于串口信息进行读和写操作

## setup.sh

初始化文档，在放入新环境后执行一次，用于将当前用户加入dialout用户组

## windmill

本文件夹为能量机关(打符)功能包，其可以通过输入图片，识别其中的能量机关的状态，并且通过弹速等信息，返回一定时间后云台需要旋转至并且开火的角度变化量

# 代码流程

## 装甲板识别

## pnp解算装甲板位姿

## 卡尔曼滤波求解车辆运动模型

![代码流程](./readme_src/code_struct.png)

# 通信协议

通信协议为23位，分为电控发给视觉的信息、打符时视觉发给电控的信息、自瞄时视觉发给电控的信息

## 电控发给视觉的信息

| 信息含义  | 长度 | 数据类型 | 单位 |
| --------- | ---- | -------- | ---- |
| pitch角   | 4    | float    | 弧度 |
| yaw角     | 4    | float    | 弧度 |
| 状态位    | 1    | uint8_t  | /    |
| armor_flag  哪些车是大装甲板    | 1    | uint8_t  | /    |

## 打符时视觉发给电控的信息



## 自瞄时视觉发给电控的信息(有用)

| 信息含义                              | 长度 | 数据类型 | 单位     |
| ------------------------------------- | ---- | -------- | -------- |
| x_c: 对方车体中心x坐标     | 4    | float    | mm       |
| v_x: 对方车体运动速度x分量 | 4    | float    | mm/s       |
| y_c: 对方车体中心y坐标     | 4    | float    | mm       | 
| v_y: 对方车体运动速度y分量 | 4    | float    | mm/s       | 
| z1:  对方车第一对装甲板的高度| 4    | float    | mm       |
| z2:  对方车第二对装甲板的高度| 4    | float    | mm       |
| v_z: 对方车体运动速度z分量   | 4    | float    | mm/s       |
| r1:  对方车第一对装甲板对应的车体半径| 4    | float    | mm       |
| r2:  对方车第二对装甲板对应的车体半径| 4    | float    | mm       |
| yaw_a: 对方车第一块装甲板对应的yaw角 | 4    | float    | 弧度     |
| vyaw:  对方车旋转的角速度   | 4    | float    | 弧度/s   |
| crc校验位                             | 2    | uint16_t | /        |

## 状态位含义

| 数字 | /5 | %5 | 颜色(己方) | 状态     |
| ---- | -- | -- | ---------- | -------- |
| 0    | 0  | 0  | 红色       | 自瞄     |
| 1    | 0  | 1  | 红色       | 击打小符 |
| 2    | 0  | 2  | 红色       | /        |
| 3    | 0  | 3  | 红色       | 击打大符 |
| 4    | 0  | 4  | 红色       | /        |
| 5    | 1  | 0  | 蓝色       | 自瞄     |
| 6    | 1  | 1  | 蓝色       | 击打小符 |
| 7    | 1  | 2  | 蓝色       | /        |
| 8    | 1  | 3  | 蓝色       | 击打大符 |
| 9    | 1  | 4  | 蓝色       | /        |

# 代码内容

## 常见自定义数据类型

### GlobalParam

全局参数结构体，其中存放一切可能更改的参数。

### Translator

串口消息联合体，其联合内容包括一个长度为23的字符数组、一个自瞄时的结构体、一个打符时的结构体。

# 调参细节

本部分主要讲解调参使用的技巧以及内容。

在宏定义中开启DEBUGMODE就可以在左上角看到调参界面，在文件 `params/UIManager.cpp`中可以找到相关的键位，使用查找加统一替换可以改为你想要的键位，之后编译即可。
