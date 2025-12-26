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

// 读线程，负责读取串口信息
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
    cv::VideoWriter *recorder = NULL;
    std::string path = "../video/record/";
    path = path + GetTime() + "/";
    std::filesystem::create_directories(path);
    int coder = cv::VideoWriter::fourcc('H', '2', '6', '4');
    int cnt = RECORD_FRAME_COUNT;
    int idx = 0;
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

    double dt = 0;
    double last_time_stamp = 0;
    bool was_in_windmill_mode_last_frame = false; // 用于模式切换检测

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
    int debug_t = 33; // 默认播放延时(ms), 约30fps
    bool is_paused = false;
#endif

    int empty_frame_count = 0;
    while (1)
    {
        if (!is_paused) { // 只在非暂停状态下执行核心逻辑
            auto t1 = std::chrono::high_resolution_clock::now();

#ifdef RECORDVIDEO
            if(!pic.empty()) { // 确保 pic 非空
                cv::Mat record_frame = pic.clone();
                cv::resize(record_frame, record_frame, cv::Size(600, 450));
                cnt++;
                if (cnt > RECORD_FRAME_COUNT && idx <= 50) {
                    cnt = 0;
                    if(recorder != NULL){
                        recorder->release();
                        delete recorder;
                    }
                    idx++;
                    recorder = new cv::VideoWriter(path + std::to_string(idx) + ".mp4", coder, 60.0, cv::Size(600, 450), true);
                }
                if(idx <= 50 && recorder != NULL) recorder->write(record_frame);
            }
#endif
            
#ifndef NOPORT
            MManager.copy(temp, translator);
#else
            MManager.FakeMessage(translator); 
#endif       
            
            bool is_windmill_mode = (translator.message.status % 5 != 0 && translator.message.status % 5 != 2);
            
#ifdef DEBUGMODE
            if(gp.debug){
                std::cout << "Received Status: " << static_cast<int>(translator.message.status) 
                          << ", Is Windmill Mode: " << (is_windmill_mode ? "YES" : "NO") << std::endl;
            }
#endif

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
            int current_color = translator.message.status / 5;
            if (current_color != gp.color) {
                gp.initGlobalParam(current_color);
            }
#endif

#ifndef VIRTUALGRAB
            camera.set_param_mult(gp);
            camera.get_pic(&pic, gp);
#else
            MManager.getFrame(pic, translator);
#endif

            if (pic.empty()){
                empty_frame_count++;
                if (gp.debug) printf("[DEBUG] get_pic returned an empty frame! Count: %d\n", empty_frame_count);
                if (empty_frame_count > 10) {
                    printf("Camera disconnected or video ended.\n");
                    exit(0);
                }
                usleep(100000);
                continue;
            } else {
                empty_frame_count = 0;
            }

            if (translator.message.status == 99) abort();

            if (is_windmill_mode) {
                WMI.identifyWM(pic, translator);
                WMIPRE.StartPredict(translator, gp, WMI);
                MManager.write(translator, *serialPort);
            } else {
                if (was_in_windmill_mode_last_frame) {
                    WMI.clear(); // 仅在从打符切换到自瞄时清空一次
                }
                
                double time_stamp = std::chrono::duration<double>(
                    std::chrono::high_resolution_clock::now().time_since_epoch()
                ).count();
                
                if (last_time_stamp == 0) {
                    last_time_stamp = time_stamp;
                    dt = 0.016;
                } else {
                    dt = time_stamp - last_time_stamp;
                }
                translator.message.latency = dt * 1000;
                last_time_stamp = time_stamp;
                
                aim.auto_aim(pic, translator, dt);
                MManager.write(translator, *serialPort);
            }

            // 在循环末尾更新上一帧的状态
            was_in_windmill_mode_last_frame = is_windmill_mode;

        } // is_paused 逻辑块结束

        // --- 调试与可视化（无论是否暂停都执行） ---
#ifdef DEBUGMODE
        cv::Mat final_image_to_show;
        if (is_paused && !WMIPRE.GetDebugImg().empty()) {
            final_image_to_show = WMIPRE.GetDebugImg();
        } else if (!pic.empty()) {
            final_image_to_show = WMIPRE.GetDebugImg().empty() ? pic : WMIPRE.GetDebugImg();
        }

        if (!final_image_to_show.empty()) {
            UI.receive_pic(final_image_to_show);
            UI.windowsManager(key, debug_t);
        }
        
        int wait_time = is_paused ? 0 : debug_t;
        key = cv::waitKey(wait_time); 
        
        if (key == ' ') {
            is_paused = !is_paused;
        } else if (key == 27 || key == 'q') {
            exit(0);
        }
#endif

        // --- 帧率计算 ---
#ifdef SHOW_FPS
        if (!is_paused) {
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
        } else {
             fps_time_stamp = std::chrono::high_resolution_clock::now();
        }
#endif
    }
    return NULL;
}