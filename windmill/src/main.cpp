#include "globalParam.hpp"
#include "WMIdentify.hpp"
#include "WMPredict.hpp"
//#include "MessageManager.hpp" // 仅为使用 Translator 结构体
#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>
#include <chrono>

/**
 * @brief 主函数 - 用于本地视频的能量机关识别
 *
 * 该程序从指定的本地视频文件中逐帧读取图像，
 * 调用能量机关识别和预测算法，并将结果可视化显示。
 */

 bool g_pause = false;

void globalWaitKeyControl() {
    int key = cv::waitKey(g_pause ? 0 : 50);

    if (key == ' ') {
        g_pause = !g_pause;  // 切换暂停
    }
    else if (key == 'q' || key == 27) {
        exit(0);  // 全局退出
    }
}

int main(int argc, char **argv)
{
    // ================== 1. 配置与初始化 ==================
    
    // --- 用户需要修改的部分 ---
    // 请在这里填入您的视频文件路径
    std::string video_path = "../test_video/output.mp4"; 
    // 设置要识别的目标颜色 (RED 或 BLUE)
    const int target_color = BLUE; 
    // -------------------------

    // 初始化全局参数
    GlobalParam gp;
    gp.initGlobalParam(target_color);
    // 强制开启调试模式，以便在图像上看到绘制的识别结果
    gp.debug = true; 

    // 初始化视频捕获
    cv::VideoCapture cap(video_path);
    if (!cap.isOpened()) {
        std::cerr << "错误: 无法打开视频文件: " << video_path << std::endl;
        return -1;
    }
    std::cout << "视频文件已成功打开，按 'q' 或 'ESC' 退出，按 '空格' 暂停/继续。" << std::endl;

    // 实例化核心算法模块
    WMIdentify WMI(gp);
    WMPredict WMIPRE(gp);
    

    cv::Mat debug_image = WMIPRE.GetDebugImg(); 
    // 用于显示和控制的变量
    cv::Mat frame;
    int key = 0;
    
    // 用于计算FPS
    auto last_time = std::chrono::high_resolution_clock::now();
    int frame_count = 0;
    double fps = 0;

    // ================== 2. 主处理循环 ==================

    while (true)
    {
        // 从视频文件读取一帧
        cap >> frame;

        // 如果视频结束，则退出循环
        if (frame.empty()) {
            std::cout << "视频播放结束。" << std::endl;
            break;
        }

        // 为当前帧创建一个通信载体对象 (模拟从电控接收信息)
        Translator translator;
        // 手动设置状态为能量机关模式 (例如，1、3、4为打符模式)
        // 这会影响到 camera.change_attack_mode 等函数，此处仅为保持接口一致性
        translator.message.status = (target_color == RED) ? 6 : 1; 

        // --- 核心识别与预测逻辑 ---
        WMI.identifyWM(frame, translator);
        WMIPRE.StartPredict(translator, gp, WMI);
        // -------------------------

        // 计算并显示帧率 (FPS)
        frame_count++;
        auto current_time = std::chrono::high_resolution_clock::now();
        auto time_diff = std::chrono::duration_cast<std::chrono::milliseconds>(current_time - last_time).count();
        if (time_diff >= 1000) { // 每秒更新一次
            fps = frame_count * 1000.0 / time_diff;
            frame_count = 0;
            last_time = current_time;
        }
        std::string fps_text = "FPS: " + std::to_string(static_cast<int>(fps));
        cv::putText(frame, fps_text, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);

        //cv::Mat debug_image = WMIPRE.GetDebugImg(); 

        // 保护层
        //if (debug_image.empty()) {
        //    std::cout << "错误: 从 WMPredict 获取的 debugImg 为空！将显示原始帧。" << std::endl;
            // 如果为空，就用原始帧代替，防止程序崩溃
        //    debug_image = frame.clone(); 
        //}
        // ======================

        // 显示处理后的图像
        //cv::imshow("Windmill Detection - debug Video", debug_image);

        globalWaitKeyControl();

    }

    // ================== 3. 清理资源 ==================
    cap.release();
    cv::destroyAllWindows();

    return 0;
}