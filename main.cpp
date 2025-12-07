#include "MessageManager.hpp"
#include "SerialPort.hpp"
#include "camera.hpp"
#include "globalParam.hpp"
#include "WMIdentify.hpp"
#include "WMPredict.hpp"
#include <AimAuto.hpp>
#include <UIManager.hpp>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <glog/logging.h>
#include <iostream>
#include <monitor.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <pthread.h>
#include <ratio>
#include <string>
#include <unistd.h>
#include <filesystem>


#define RECORD_FRAME_COUNT 1800

std::string GetTime(){
    std::time_t now = std::time(nullptr);
    std::tm *p_tm = std::localtime(&now);
    char time_str[50];
    std::strftime(time_str, sizeof(time_str), "%Y-%m-%d %H:%M", p_tm);
    return time_str;
}

// 全局变量参数
GlobalParam gp;
// 通信类
MessageManager MManager(gp);
#ifndef VIRTUALGRAB
// 相机类
Camera camera(gp);
#endif
// 通信数据结构
Translator temp;
Translator translator;
cv::Mat pic;
#ifdef NOPORT
const int COLOR = BLUE;
#endif // NOPORT

// 读线程，负责读取串口信息以及取流
void *ReadFunction(void *arg);
// 运算线程，负责对图片进行处理
void *OperationFunction(void *arg);

int main(int argc, char **argv)
{   
    printf("Welcome to Windmill & Armor Auto Aim System\n");
    // 参数检查
    if (argc < 3) {
        printf("Usage: %s <serial_port> <baud_rate_index>\n", argv[0]);
        return -1;
    }

    SerialPort *serialPort = new SerialPort(argv[1]);
    serialPort->InitSerialPort(int(*argv[2] - '0'), 8, 1, 'N');
    
#ifndef NOPORT
    MManager.read(temp, *serialPort);
    // 通过电控发来的模式位判断初始颜色
    MManager.initParam(temp.message.status / 5 == 0 ? RED : BLUE);
#else
    MManager.initParam(COLOR);
#endif // NOPORT

    pthread_t readThread;
    pthread_t operationThread;

    // 开启线程
    pthread_create(&readThread, NULL, ReadFunction, serialPort);
    pthread_create(&operationThread, NULL, OperationFunction, serialPort);

    // 等待运算线程结束（通常不会）
    pthread_join(operationThread,NULL);

    delete serialPort;
    return 0;
}

void *ReadFunction(void *arg) // 读线程
{
#ifdef THREADANALYSIS
    printf("Read thread init successful\n");
#endif
    SerialPort *serialPort = (SerialPort *)arg;
    while (1)
    {
        MManager.read(temp, *serialPort);
        usleep(100); // 稍微让出CPU
    }
    return NULL;
}

void *OperationFunction(void *arg)
{
#ifdef RECORDVIDEO
    // ... (视频录制代码保持不变)
#endif
#ifdef THREADANALYSIS
    printf("Operation thread init successful\n");
#endif
    SerialPort *serialPort = (SerialPort *)arg;
    
    // 实例化各模块对象
    WMIdentify WMI(gp);
    AimAuto aim(&gp);
    WMPredict WMIPRE(gp);
#ifdef DEBUGMODE
    UIManager UI(gp);
#endif
    WMI.clear();

#ifndef VIRTUALGRAB
    camera.init();
#endif 
#ifdef SHOW_FPS
    int fps = 0;
    int frame_count = 0;
    auto fps_time_stamp = std::chrono::high_resolution_clock::now();
#endif
#ifdef DEBUGMODE
    int key = 0;
    int debug_t = 1;
#endif

    int empty_frame_count = 0;
    while (1)
    {
        auto t1 = std::chrono::high_resolution_clock::now();

#ifdef RECORDVIDEO
        // ... (视频录制代码保持不变)
#endif
        
#ifndef NOPORT
        MManager.copy(temp, translator);
#else
        MManager.FakeMessage(translator); 
#endif       
        
        // --- 模式判断与相机参数设置 ---
        bool is_windmill_mode = (translator.message.status % 5 != 0 && translator.message.status % 5 != 2);
        
        if (is_windmill_mode) {
#ifndef VIRTUALGRAB
            camera.change_attack_mode(ENERGY, gp);
#endif
            gp.attack_mode = ENERGY;
        } else {
#ifndef VIRTUALGRAB
            camera.change_attack_mode(ARMOR, gp);
#endif
            gp.attack_mode = ARMOR;
        }

#ifndef NOPORT
        // 根据电控指令动态切换颜色
        int current_color = translator.message.status / 5;
        if (current_color != gp.color) {
            gp.initGlobalParam(current_color);
        }
#endif

        // --- 获取图像 ---
#ifndef VIRTUALGRAB
        camera.set_param_mult(gp);
        camera.get_pic(&pic, gp);
#else
        MManager.getFrame(pic, translator);
#endif

        if (pic.empty()){
            empty_frame_count++;
            if (empty_frame_count > 5) { // 连续5帧为空则认为相机断开
                printf("Camera disconnected or video ended.\n");
                exit(0);
            }
            continue; // 跳过此帧
        } else {
            empty_frame_count = 0;
        }

        // --- 核心逻辑分支 ---
        if (translator.message.status == 99) abort(); // 紧急停止

        if (is_windmill_mode) {
            // --- 能量机关（打符）模式 ---
            // 1. 识别：WMIdentify 现在会进行多目标检测与跟踪
            WMI.identifyWM(pic, translator);
            
            // 2. 预测：WMPredict 现在会从 WMI 获取所有 trackers，并决策攻击哪一个
            WMIPRE.StartPredict(translator, gp, WMI);
            
            // 3. 发送：将最终计算出的角度发送给电控
            MManager.write(translator, *serialPort);

        } else {
            // --- 自瞄（打装甲板）模式 ---
            WMI.clear(); // 进入自瞄时，清空所有能量机关的跟踪器
            aim.auto_aim(pic, translator); // (假设 auto_aim 接口不需要 dt)
            MManager.write(translator, *serialPort);
        }

        // --- 调试与可视化 ---
#ifdef DEBUGMODE
        // 注意：现在可视化窗口由各个模块自己管理
        // main 函数只需要一个 waitKey 来控制流程
        // UI.receive_pic(pic); // 如果 UI 需要显示最终图像，需要从 WMPredict 获取
        cv::Mat final_image_to_show = WMIPRE.GetDebugImg();
        if (final_image_to_show.empty()) {
            final_image_to_show = pic; // 如果预测模块没图，就显示原图
        }
        UI.receive_pic(final_image_to_show);
        UI.windowsManager(key, debug_t);
        key = cv::waitKey(debug_t);
        if (key == ' ') key = cv::waitKey(0);
        if (key == 27 || key == 'q') exit(0);
#endif

        // --- 帧率计算 ---
#ifdef SHOW_FPS
        frame_count++;
        auto now_time = std::chrono::high_resolution_clock::now();
        auto time_diff_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now_time - fps_time_stamp).count();
        if (time_diff_ms >= 1000)
        {
            fps = frame_count;
            printf("FPS: %d\n", fps);
            frame_count = 0;
            fps_time_stamp = now_time;
        }
#endif
    }
    return NULL;
}