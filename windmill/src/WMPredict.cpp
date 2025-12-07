/*
 * @Author: zxh 1608278840@qq.com
 * @Date: 2023-11-08 03:11:49
 * @FilePath: /DX_aimbot/windmill/src/WMPredict.cpp
 * @Description:能量机关预测
 * Copyright (c) 2023 by ${git_name_email}, All Rights Reserved.
 */

#include "WMPredict.hpp"
#include <algorithm>
#include <cmath>
#include <deque>
#include <utility>
// #include "SerialPort.hpp"
#include "WMIdentify.hpp"
#include "globalParam.hpp"
#include "opencv2/core/mat.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc/types_c.h"
// #include <angular_velocity_fitter.hpp>
//  #include <ceres/ceres.h>
#include <chrono>
#include <complex>
#include <deque>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <opencv2/core/types.hpp>
#include <opencv2/imgproc.hpp>
#include <ostream>
#include <sstream>
#include <type_traits>
#include <unistd.h>
#include <vector>

//=====常量=====//
static double g = 9.8;       // 重力加速度
static double k = 0.05;      // 空气阻力系数  //考虑了子弹质量？
static double r = 0.7;       // 符的半径
static double s = 999999.3;  // 车距离符的水平距离
static double Exp = 2.71823; // 自然常数e
static double h0 = 1.1747;   // 符的中心点高度减去车的高度
static double pi = 3.14159;  // 圆周率
// static double delta_t = 0.10; // 超前滞后
static double diff_w = 0.01;
static double w_low = 1.5;
static double w_up = 2.5;
static double yaw_fix = 0.07; // 相机yaw的修正常数

// bool debug = false;

bool got_angle_velocity = false;
int fit_count = 0;
const int MAX_FIT_COUNT = 10;
std::vector<double> w_big_fits;
std::vector<double> A0_fits;
std::vector<double> fai_fits;
std::vector<double> b_fits;
// 新增标志位，表示是否已固定w、A、b参数
bool params_fixed = false;
// 固定后的参数值
double fixed_w_big = 0.0;
double fixed_A0 = 0.0;
double fixed_b = 0.0;

// 时间统计数据
// double convex_total_time = 0.0;
// double newton_total_time = 0.0;
// int convex_frame_count = 0;
// int newton_frame_count = 0;
// bool is_convex_running = false;
// bool is_newton_running = false;
// std::chrono::high_resolution_clock::time_point convex_start_time;
// std::chrono::high_resolution_clock::time_point newton_start_time;

WMPredict::WMPredict(GlobalParam &gp) {
  this->direction = 1;
  this->smoothData_img = cv::Mat::zeros(400, 800, CV_8UC3);
  //====大符速度参数======//
  this->A0 = 0.780;
  this->b = 1.884;
  this->fai = 1.884;
  this->w_big = 1.884;
  this->First_fit = 1;  // 是否为首次拟合,1是，0不是
  this->clockwise = -1; // 1顺时针，0逆时针,初始化为-1,表示未确定

  this->delta_t = gp.delta_t;

  // 初始化时间统计数据
  ResetTimeStatistics();

  this->now_time = 0;
  this->Fire_time = 0;
}

/**
 * @brief (已重构) 打符主流程，支持多目标决策
 * @param {Translator} &translator  串口消息
 * @param {GlobalParam} &gp  传入全局变量
 * @param {WMIdentify} &WMI  传入识别类
 * @return {*}
 */
int WMPredict::StartPredict(Translator &translator, GlobalParam &gp,
                            WMIdentify &WMI) {
    
    // --- 1. 获取所有被跟踪的目标 ---
    auto& trackers = WMI.getTrackers();

    // 如果没有任何被跟踪的目标，则直接返回
    if (trackers.empty()) {
        std::cout << "没有跟踪任何目标，不预测" << std::endl;
        translator.message.armor_flag = 10;
        return 0;
    }

    // --- 2. (新增) 决策模块：选择要攻击的目标 ---
    int best_tracker_idx = -1;
    // 简单的决策逻辑：选择第一个被跟踪的目标
    for (int i = 0; i < trackers.size(); ++i) {
        if (trackers[i].is_tracking && trackers[i].last_detection.is_fully_matched) {
            best_tracker_idx = i;
            break;
        }
    }

    // 如果没有找到正在跟踪的目标（虽然 trackers 不为空），也返回
    if (best_tracker_idx == -1) {
        std::cout << "所有跟踪器都已丢失，不预测" << std::endl;
        translator.message.armor_flag = 10;
        return 0;
    }
    
    TargetTracker& target_tracker = trackers[best_tracker_idx];

    // --- 3. 更新全局状态 ---
    // 调用简化的 UpdateData
    this->UpdateData(WMI); 

    // --- 4. 根据模式执行预测逻辑 ---
    // 判断是大符还是小符
    if (translator.message.status % 5 == 3) { // 大符模式
        // a. 检查该 tracker 的数据是否足够
        if (target_tracker.list_stat == 0) {
            std::cout << "Tracker ID " << target_tracker.id << " 数据不足，不预测" << std::endl;
            return 0;
        }

        // b. (可选) 进行速度拟合
        // 为了稳定，通常只在 tracker 生命周期的早期进行几次拟合
        if (target_tracker.fit_count < MAX_FIT_COUNT) {
            this->ConvexOptimization(target_tracker.time_list,
                                     target_tracker.angle_velocity_list, gp, translator);
            
            // 将拟合结果更新回 tracker 自身，让它记住自己的运动模型
            target_tracker.w_big = this->w_big;
            target_tracker.A0 = this->A0;
            target_tracker.fai = this->fai;
            target_tracker.b = this->b;
            target_tracker.fit_count++;
        }
        
        // c. 使用该 tracker 自己的运动模型参数进行预测
        this->w_big = target_tracker.w_big;
        this->A0 = target_tracker.A0;
        this->fai = target_tracker.fai;
        this->b = target_tracker.b;

        // d. 调用弹道解算
        this->NewtonDspBigAnyPos(target_tracker.last_detection.world2car_matrix,
                                 translator, gp, 
                                 target_tracker.last_detection.current_angle,
                                 target_tracker.last_detection.distance);
        translator.message.armor_flag = 11;

    } else { // 小符模式
        if (this->clockwise != -1) {
            // 设置固定的匀速模型
            this->A0 = 0.0;
            this->b = 1.047197551;
            this->w_big = 1.0;
            this->fai = 0.0;

            // 调用弹道解算
            this->NewtonDspBigAnyPos(target_tracker.last_detection.world2car_matrix,
                                     translator, gp, 
                                     target_tracker.last_detection.current_angle,
                                     target_tracker.last_detection.distance);
            translator.message.armor_flag = 11;
        } else {
            std::cout << "小符方向未确定，不预测" << std::endl;
        }
    }
  
    return 1;
}

/**
 * @brief (已重构) 根据识别模块更新预测器内部状态
 * @param {WMIdentify} &WMI  传入识别类
 * @return {*}
 */
void WMPredict::UpdateData(WMIdentify &WMI) {
    // 这个函数现在变得非常简单，因为它不再需要处理单个目标的姿态信息
    // 这些信息都封装在 WMIdentify 的 trackers 中
    
    // 我们可能仍然需要获取最新的图像用于绘制
    this->debugImg = WMI.getImg0();

    // 如果需要判断全局的旋转方向，可以遍历所有 trackers
    // 但更推荐的做法是让每个 tracker 自己维护方向
    // 为了简单起见，我们暂时保留一个全局的 clockwise 判断逻辑
    
    auto& trackers = WMI.getTrackers();
    if (!trackers.empty()) {
        // 以第一个被跟踪的目标为参考来判断方向
        const auto& main_tracker = trackers[0]; 
        if (main_tracker.angle_velocity_list.size() >= 20) {
            int positiveCount = 0;
            int negativeCount = 0;
            for (double vel : main_tracker.angle_velocity_list) {
                if (vel > 0) positiveCount++;
                else if (vel < 0) negativeCount++;
            }
            if (positiveCount > negativeCount + 5) this->clockwise = 1;
            else if (negativeCount > positiveCount + 5) this->clockwise = 0;
        }
        
        // 更新当前时间戳，以第一个 tracker 的最新时间为准
        if (!main_tracker.time_list.empty()) {
            this->now_time = main_tracker.time_list.back();
        }
    }
}

void WMPredict::NewtonDspSmallAnyPos(cv::Mat world2car, Translator &translator,
                                     GlobalParam &gp, double R_yaw) {
  double P0 = 12 * pi / 180;
  double fly_t0 = 0.3;
  int n = 0; // 迭代次数
  double delta_theta_delay = w * delta_t;

  // cv::Mat world_point = (cv::Mat_<double>(4, 1) << r *
  // cos(delta_theta_delay), -r * sin(delta_theta_delay), 0, 1.0);
  cv::Mat world_point = (cv::Mat_<double>(4, 1) << r, 0, 0, 1.0);
  cv::Mat world_point_car = world2car * world_point;
  double x_car = world_point_car.at<double>(0, 0);
  double y_car = world_point_car.at<double>(1, 0);
  double z_car = world_point_car.at<double>(2, 0);
  double distance = sqrt(x_car * x_car + z_car * z_car);
  this->Fire_time = this->now_time + delta_t;
  double v0 = translator.message.r1 == 0 ? 23 : translator.message.r1; // 弹速
  cv::Mat P_t = cv::Mat::zeros(2, 1, CV_64F);
  cv::Mat temp =
      (cv::Mat_<double>(2, 2) << this->f1PA(P0, x_car, fly_t0, distance, v0),
       this->f1tA(P0, delta_theta_delay, world2car, fly_t0, distance, v0),
       this->f2PA(P0, z_car, fly_t0, v0),
       this->f2tA(P0, delta_theta_delay, world2car, fly_t0, v0));
  cv::Mat temp_inv = cv::Mat::zeros(2, 2, CV_64F);
  cv::Mat b =
      (cv::Mat_<double>(2, 1)
           << this->f1A(P0, delta_theta_delay, world2car, fly_t0, distance, v0),
       this->f2A(P0, delta_theta_delay, world2car, fly_t0, v0));
  double P1 = 0;
  double fly_t1 = 0;

  do {
    n++;
    P1 = P0;
    fly_t1 = fly_t0;
    //======这里对雅可比矩阵的更新要尽可能的少，不然解变化太快容易求出无意义解（t<0)======//
    temp.at<double>(0, 0) = this->f1PA(P0, x_car, fly_t0, distance, v0);
    temp.at<double>(1, 1) =
        this->f2tA(P0, delta_theta_delay, world2car, fly_t0, v0);
    cv::invert(temp, temp_inv);
    b.at<double>(0, 0) =
        this->f1A(P0, delta_theta_delay, world2car, fly_t0, distance, v0);
    b.at<double>(1, 0) =
        this->f2A(P0, delta_theta_delay, world2car, fly_t0, v0);
    P_t = P_t - temp_inv * b;
    P0 = P_t.at<double>(0, 0);
    fly_t0 = P_t.at<double>(1, 0);
    if (n > 50)
      break;
  } while (abs(fly_t0 - fly_t1) > 1e-5 ||
           abs(P0 - P1) > 1e-5); // 当前解与上次迭代解差距很小时

  // double yaw = atan2(x_car, z_car);

  double delta_theta = w * fly_t0 + delta_theta_delay;

#ifdef VISUALIZE_PREDICTION // 添加条件编译宏
  std::vector<cv::Point3f> world_points_single;
  // cv::Point3f point3f(r * cos(delta_theta + delta_theta_delay), -r *
  // sin(delta_theta + delta_theta_delay), 0);
  cv::Point3f point3f(
      r, 0,
      0); // 注意：这里与BigAnyPos不同，保留了Small的逻辑点(r,0,0)用于绘制预测点，如果需要完全一致，应使用上面注释掉的行
  world_points_single.push_back(point3f);
  std::vector<cv::Point2f> image_points_single;
  cv::projectPoints(world_points_single, rvec, tvec, camera_matrix, dist_coeffs,
                    image_points_single);
  cv::circle(this->debugImg, image_points_single[0], 5,
             cv::Scalar(255, 255, 255), -1);
#endif // VISUALIZE_PREDICTION

  cv::Mat world_point_goal =
      (cv::Mat_<double>(4, 1) << r * cos(delta_theta + delta_theta_delay),
       -r * sin(delta_theta + delta_theta_delay), 0, 1.0);
  // cv::Mat world_point_goal = (cv::Mat_<double>(4, 1) << r, 0, 0, 1.0);
  cv::Mat world_point_goal_car = world2car * world_point_goal;

  double yaw = atan2(-world_point_goal_car.at<double>(0, 0),
                     world_point_goal_car.at<double>(2, 0));

  // 将 resultYaw 和 resultPitch 的赋值提前，以便绘图时使用
  this->resultYaw = yaw * 180 / pi;
  this->resultPitch = P0 * 180 / pi;

#ifdef VISUALIZE_PREDICTION // 添加条件编译宏
  // 使用 ostringstream 格式化输出
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(2); // 设置固定点表示法和两位小数精度

  cv::putText(this->debugImg, "Predicted Point",
              image_points_single[0] + cv::Point2f(10, 10),
              cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);

  oss.str(""); // 清空流
  oss << "Pitch: " << P0 * 180 / pi;
  cv::putText(this->debugImg, oss.str(), cv::Point(10, 50),
              cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2);

  oss.str("");
  oss << "Abs Yaw: " << yaw * 180 / pi;
  cv::putText(this->debugImg, oss.str(), cv::Point(10, 80),
              cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2);

  oss.str("");
  oss << "Center_X: " << world_point_goal_car.at<double>(0, 0);
  cv::putText(this->debugImg, oss.str(), cv::Point(10, 110),
              cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2);

  oss.str("");
  oss << "Center_Y: " << world_point_goal_car.at<double>(1, 0);
  cv::putText(this->debugImg, oss.str(), cv::Point(10, 140),
              cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2);

  oss.str("");
  oss << "Center_Z: " << world_point_goal_car.at<double>(2, 0);
  cv::putText(this->debugImg, oss.str(), cv::Point(10, 170),
              cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2);

  oss.str("");
  oss << "Distance: " << distance; // 使用 distance 变量
  cv::putText(this->debugImg, oss.str(), cv::Point(10, 200),
              cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2);

  oss.str("");
  oss << "Rot Angle: " << R_yaw * 180 / pi;
  cv::putText(this->debugImg, oss.str(), cv::Point(10, 230),
              cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2);

  oss.str("");
  oss << "Recv Yaw: " << translator.message.yaw * 180 / pi;
  cv::putText(this->debugImg, oss.str(), cv::Point(10, 260),
              cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2);

  oss.str("");
  oss << "Recv Pitch: " << translator.message.pitch * 180 / pi;
  cv::putText(this->debugImg, oss.str(), cv::Point(10, 290),
              cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2);

  oss.str("");
  oss << "Rel Yaw: " << (translator.message.yaw - yaw) * 180 / pi;
  cv::putText(this->debugImg, oss.str(), cv::Point(10, 320),
              cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2);

  oss.str("");
  oss << "Rel Pitch: " << (P0 - translator.message.pitch) * 180 / pi;
  cv::putText(this->debugImg, oss.str(), cv::Point(10, 350),
              cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2);

  // 右侧参数显示 (与大符一致，显示大符参数)
  int right_x = 10; // 右侧文本起始 x 坐标
  oss.str("");
  oss << "w_big: " << this->w_big; // 注意：这里显示的是大符参数
  cv::putText(this->debugImg, oss.str(), cv::Point(right_x, 510),
              cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 255), 2);

  oss.str("");
  oss << "A0: " << this->A0; // 注意：这里显示的是大符参数
  cv::putText(this->debugImg, oss.str(), cv::Point(right_x, 540),
              cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 255), 2);

  oss.str("");
  oss << "b: " << this->b; // 注意：这里显示的是大符参数
  cv::putText(this->debugImg, oss.str(), cv::Point(right_x, 570),
              cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 255), 2);

  oss.str("");
  oss << "fai(deg): " << this->fai * 180 / pi; // 注意：这里显示的是大符参数
  cv::putText(this->debugImg, oss.str(), cv::Point(right_x, 600),
              cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 255), 2);
#endif // VISUALIZE_PREDICTION

  translator.message.yaw_a = yaw * 180 / pi;
  translator.message.r2 = P0 * 180 / pi;

#ifdef VISUALIZE_PREDICTION   // 添加条件编译宏
  cv::Size newSize(840, 620); // 调整窗口大小以适应新的文本布局
  cv::resize(debugImg, debugImg, newSize, 0, 0, cv::INTER_LINEAR);

  // --- 曲线绘制代码 ---
  static std::vector<double> fai_history;
  static std::vector<double> time_points;
  static std::vector<double> yaw_history;
  static std::vector<double> pitch_history;
  static double start_time = cv::getTickCount() / cv::getTickFrequency();
  double current_time =
      cv::getTickCount() / cv::getTickFrequency() - start_time;

  fai_history.push_back(this->fai * 180 / pi); // 使用大符的 fai
  yaw_history.push_back(this->resultYaw);      // 使用已计算的 resultYaw
  pitch_history.push_back(this->resultPitch);  // 使用已计算的 resultPitch
  time_points.push_back(current_time);

  if (fai_history.size() > 200) {
    fai_history.erase(fai_history.begin());
    yaw_history.erase(yaw_history.begin());
    pitch_history.erase(pitch_history.begin());
    time_points.erase(time_points.begin());
  }

  cv::Mat curve(300, 600, CV_8UC3, cv::Scalar(0, 0, 0));
  cv::Mat yaw_curve(300, 600, CV_8UC3, cv::Scalar(0, 0, 0));
  cv::Mat pitch_curve(300, 600, CV_8UC3, cv::Scalar(0, 0, 0));

  cv::line(curve, cv::Point(50, 250), cv::Point(550, 250),
           cv::Scalar(255, 255, 255), 1);
  cv::line(curve, cv::Point(50, 50), cv::Point(50, 250),
           cv::Scalar(255, 255, 255), 1);

  cv::line(yaw_curve, cv::Point(50, 250), cv::Point(550, 250),
           cv::Scalar(255, 255, 255), 1);
  cv::line(yaw_curve, cv::Point(50, 50), cv::Point(50, 250),
           cv::Scalar(255, 255, 255), 1);

  cv::line(pitch_curve, cv::Point(50, 250), cv::Point(550, 250),
           cv::Scalar(255, 255, 255), 1);
  cv::line(pitch_curve, cv::Point(50, 50), cv::Point(50, 250),
           cv::Scalar(255, 255, 255), 1);

  for (size_t i = 1; i < fai_history.size(); i++) {
    double x1 = 50 + (time_points[i - 1] - time_points[0]) /
                         (time_points.back() - time_points[0]) * 500;
    double y1 = 250 - fai_history[i - 1] * 200 / 360;
    double x2 = 50 + (time_points[i] - time_points[0]) /
                         (time_points.back() - time_points[0]) * 500;
    double y2 = 250 - fai_history[i] * 200 / 360;

    cv::line(curve, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(0, 255, 0),
             2);
  }

  for (size_t i = 1; i < yaw_history.size(); i++) {
    double x1 = 50 + (time_points[i - 1] - time_points[0]) /
                         (time_points.back() - time_points[0]) * 500;
    double y1 = 150 - (yaw_history[i - 1]) * 180 / 20; // 150是中心线(0度)的位置
    double x2 = 50 + (time_points[i] - time_points[0]) /
                         (time_points.back() - time_points[0]) * 500;
    double y2 = 150 - (yaw_history[i]) * 180 / 20;

    cv::line(yaw_curve, cv::Point(x1, y1), cv::Point(x2, y2),
             cv::Scalar(0, 0, 255), 2);
  }

  cv::line(yaw_curve, cv::Point(50, 150), cv::Point(550, 150),
           cv::Scalar(100, 100, 100), 1, cv::LINE_8);

  for (size_t i = 1; i < pitch_history.size(); i++) {
    double x1 = 50 + (time_points[i - 1] - time_points[0]) /
                         (time_points.back() - time_points[0]) * 500;
    double y1 = 250 - (pitch_history[i - 1]) * 180 / 25; // 直接显示绝对值
    double x2 = 50 + (time_points[i] - time_points[0]) /
                         (time_points.back() - time_points[0]) * 500;
    double y2 = 250 - (pitch_history[i]) * 180 / 25;

    cv::line(pitch_curve, cv::Point(x1, y1), cv::Point(x2, y2),
             cv::Scalar(255, 0, 0), 2);
  }

  cv::line(yaw_curve, cv::Point(50, 60), cv::Point(550, 60),
           cv::Scalar(50, 50, 50), 1, cv::LINE_8); // +10度
  cv::line(yaw_curve, cv::Point(50, 240), cv::Point(550, 240),
           cv::Scalar(50, 50, 50), 1, cv::LINE_8); // -10度
  cv::putText(yaw_curve, "+10°", cv::Point(20, 65), cv::FONT_HERSHEY_SIMPLEX,
              0.5, cv::Scalar(200, 200, 200), 1);
  cv::putText(yaw_curve, "0°", cv::Point(30, 155), cv::FONT_HERSHEY_SIMPLEX,
              0.5, cv::Scalar(200, 200, 200), 1);
  cv::putText(yaw_curve, "-10°", cv::Point(20, 245), cv::FONT_HERSHEY_SIMPLEX,
              0.5, cv::Scalar(200, 200, 200), 1);

  cv::line(pitch_curve, cv::Point(50, 130), cv::Point(550, 130),
           cv::Scalar(50, 50, 50), 1, cv::LINE_8); // 12.5度
  cv::putText(pitch_curve, "25°", cv::Point(25, 75), cv::FONT_HERSHEY_SIMPLEX,
              0.5, cv::Scalar(200, 200, 200), 1);
  cv::putText(pitch_curve, "12.5°", cv::Point(15, 135),
              cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(200, 200, 200), 1);
  cv::putText(pitch_curve, "0°", cv::Point(30, 255), cv::FONT_HERSHEY_SIMPLEX,
              0.5, cv::Scalar(200, 200, 200), 1);

  cv::putText(curve, "Fai", cv::Point(250, 30), cv::FONT_HERSHEY_SIMPLEX, 0.8,
              cv::Scalar(255, 255, 255), 1);
  cv::putText(yaw_curve, "Yaw", cv::Point(250, 30), cv::FONT_HERSHEY_SIMPLEX,
              0.8, cv::Scalar(255, 255, 255), 1);
  cv::putText(pitch_curve, "Pitch", cv::Point(250, 30),
              cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 255, 255), 1);

  cv::imshow("Fai", curve);
  cv::imshow("Yaw", yaw_curve);
  cv::imshow("Pitch", pitch_curve);
  cv::imshow("debugImg", this->debugImg);
  cv::waitKey(1);
#endif // VISUALIZE_PREDICTION
}
void WMPredict::NewtonDspBigAnyPos(cv::Mat world2car, Translator &translator,
                                   GlobalParam &gp, double R_yaw,
                                   double R_distance) {
  // 开始计时
  is_newton_running = true;
  newton_start_time = std::chrono::high_resolution_clock::now();

  translator.message.yaw = 0;
  translator.message.pitch = 0;

  double P0 = 12 * pi / 180;
  double fly_t0 = 0.3;
  int n = 0; // 迭代次数
  double delta_theta_delay = ThetaToolForBig(delta_t, this->now_time);

  // cv::Mat world_point = (cv::Mat_<double>(4, 1) << r *
  // cos(delta_theta_delay), -r * sin(delta_theta_delay), 0, 1.0);
  cv::Mat world_point = (cv::Mat_<double>(4, 1) << r, 0, 0, 1.0);
  cv::Mat world_point_car = world2car * world_point;
  double x_car = world_point_car.at<double>(0, 0);
  double y_car = world_point_car.at<double>(1, 0);
  double z_car = world_point_car.at<double>(2, 0);
  double distance = sqrt(x_car * x_car + y_car * y_car + z_car * z_car);
  this->Fire_time = this->now_time + delta_t;
  //  double v0 = 23.0; // 弹速
  double v0 = translator.message.r1 < 20 ? 23 : translator.message.r1; // 弹速
  std::cout << "==================================" << v0
            << "=====" << std::endl;

  cv::Mat P_t = cv::Mat::zeros(2, 1, CV_64F);
  cv::Mat temp =
      (cv::Mat_<double>(2, 2) << this->F1PA(P0, x_car, fly_t0, distance, v0),
       this->F1tA(P0, delta_theta_delay, world2car, fly_t0, distance, v0),
       this->F2PA(P0, z_car, fly_t0, v0),
       this->F2tA(P0, delta_theta_delay, world2car, fly_t0, v0));
  cv::Mat temp_inv = cv::Mat::zeros(2, 2, CV_64F);
  cv::Mat b =
      (cv::Mat_<double>(2, 1)
           << this->F1A(P0, delta_theta_delay, world2car, fly_t0, distance, v0),
       this->F2A(P0, delta_theta_delay, world2car, fly_t0, v0));
  double P1 = 0;
  double fly_t1 = 0;

  do {
    n++;
    P1 = P0;
    fly_t1 = fly_t0;
    //======这里对雅可比矩阵的更新要尽可能的少，不然解变化太快容易求出无意义解（t<0)======//
    temp.at<double>(0, 0) = this->F1PA(P0, x_car, fly_t0, distance, v0);
    temp.at<double>(1, 1) =
        this->F2tA(P0, delta_theta_delay, world2car, fly_t0, v0);
    cv::invert(temp, temp_inv);
    b.at<double>(0, 0) =
        this->F1A(P0, delta_theta_delay, world2car, fly_t0, distance, v0);
    b.at<double>(1, 0) =
        this->F2A(P0, delta_theta_delay, world2car, fly_t0, v0);
    P_t = P_t - temp_inv * b;
    P0 = P_t.at<double>(0, 0);
    fly_t0 = P_t.at<double>(1, 0);
    if (n > 50)
      break;
  } while (abs(fly_t0 - fly_t1) > 1e-5 ||
           abs(P0 - P1) > 1e-5); // 当前解与上次迭代解差距很小时

  // double yaw = atan2(x_car, z_car);

  double delta_theta = ThetaToolForBig(fly_t0, this->Fire_time);

#ifdef VISUALIZE_PREDICTION // 添加条件编译宏
  std::vector<cv::Point3f> world_points_single;
  cv::Point3f point3f(r * cos(delta_theta + delta_theta_delay),
                      -r * sin(delta_theta + delta_theta_delay), 0);
  world_points_single.push_back(point3f);
  std::vector<cv::Point2f> image_points_single;
  cv::projectPoints(world_points_single, rvec, tvec, camera_matrix, dist_coeffs,
                    image_points_single);
  cv::circle(this->debugImg, image_points_single[0], 5,
             cv::Scalar(255, 255, 255), -1);
#endif // VISUALIZE_PREDICTION

  cv::Mat world_point_goal =
      (cv::Mat_<double>(4, 1) << r * cos(delta_theta + delta_theta_delay),
       -r * sin(delta_theta + delta_theta_delay), 0, 1.0);
  cv::Mat world_point_goal_car = world2car * world_point_goal;

  double yaw = atan2(-world_point_goal_car.at<double>(0, 0),
                     world_point_goal_car.at<double>(2, 0));

  // std::cout << yaw * 180 / pi << std::endl;

#ifdef VISUALIZE_PREDICTION // 添加条件编译宏
  // 使用 ostringstream 格式化输出
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(2); // 设置固定点表示法和两位小数精度

  cv::putText(this->debugImg, "Predicted Point",
              image_points_single[0] + cv::Point2f(10, 10),
              cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);

  oss.str(""); // 清空流
  oss << "Pitch: " << P0 * 180 / pi;
  cv::putText(this->debugImg, oss.str(), cv::Point(10, 50),
              cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2);

  oss.str("");
  oss << "Abs Yaw: " << yaw * 180 / pi;
  cv::putText(this->debugImg, oss.str(), cv::Point(10, 80),
              cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2);

  oss.str("");
  oss << "Center_X: " << world_point_goal_car.at<double>(0, 0);
  cv::putText(this->debugImg, oss.str(), cv::Point(10, 110),
              cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2);

  oss.str("");
  oss << "Center_Y: " << world_point_goal_car.at<double>(1, 0);
  cv::putText(this->debugImg, oss.str(), cv::Point(10, 140),
              cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2);

  oss.str("");
  oss << "Center_Z: " << world_point_goal_car.at<double>(2, 0);
  cv::putText(this->debugImg, oss.str(), cv::Point(10, 170),
              cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2);

  oss.str("");
  oss << "Distance: " << R_distance; // R_distance 已经是距离，不需要 * 180 / pi
  cv::putText(this->debugImg, oss.str(), cv::Point(10, 200),
              cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2);

  oss.str("");
  oss << "Rot Angle: " << R_yaw * 180 / pi;
  cv::putText(this->debugImg, oss.str(), cv::Point(10, 230),
              cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2);

  oss.str("");
  oss << "Recv Yaw: " << translator.message.yaw * 180 / pi;
  cv::putText(this->debugImg, oss.str(), cv::Point(10, 260),
              cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2);

  oss.str("");
  oss << "Recv Pitch: " << translator.message.pitch * 180 / pi;
  cv::putText(this->debugImg, oss.str(), cv::Point(10, 290),
              cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2);

  oss.str("");
  oss << "Rel Yaw: " << (translator.message.yaw - yaw) * 180 / pi;
  cv::putText(this->debugImg, oss.str(), cv::Point(10, 320),
              cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2);

  oss.str("");
  oss << "Rel Pitch: " << (P0 - translator.message.pitch) * 180 / pi;
  cv::putText(this->debugImg, oss.str(), cv::Point(10, 350),
              cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2);

  // 右侧参数显示
  int right_x = 10; // 右侧文本起始 x 坐标
  oss.str("");
  oss << "w_big: " << this->w_big;
  cv::putText(this->debugImg, oss.str(), cv::Point(right_x, 510),
              cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 255), 2);

  oss.str("");
  oss << "A0: " << this->A0;
  cv::putText(this->debugImg, oss.str(), cv::Point(right_x, 540),
              cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 255), 2);

  oss.str("");
  oss << "b: " << this->b; // b 通常是弧度制，如果需要角度显示则 * 180 / pi
  cv::putText(this->debugImg, oss.str(), cv::Point(right_x, 570),
              cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 255), 2);

  oss.str("");
  oss << "fai(deg): " << this->fai * 180 / pi; // fai 通常是弧度制，显示为角度
  cv::putText(this->debugImg, oss.str(), cv::Point(right_x, 600),
              cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 255), 2);
#endif // VISUALIZE_PREDICTION

  std::cout << "==========Y===" << (translator.message.yaw - yaw) * 180 / pi
            << std::endl;
  std::cout << "==========P===" << (translator.message.pitch - P0) * 180 / pi
            << std::endl;

  this->resultYaw = yaw * 180 / pi;
  this->resultPitch = P0 * 180 / pi;

  translator.message.yaw_a = yaw * 180 / pi;
  translator.message.r2 = P0 * 180 / pi;

#ifdef VISUALIZE_PREDICTION   // 添加条件编译宏
  cv::Size newSize(840, 620); // 调整窗口大小以适应新的文本布局
  cv::resize(debugImg, debugImg, newSize, 0, 0, cv::INTER_LINEAR);

  static std::vector<double> fai_history;
  static std::vector<double> time_points;
  static std::vector<double> yaw_history;
  static std::vector<double> pitch_history;
  static double start_time = cv::getTickCount() / cv::getTickFrequency();
  double current_time =
      cv::getTickCount() / cv::getTickFrequency() - start_time;

  fai_history.push_back(this->fai * 180 / pi);
  yaw_history.push_back(this->resultYaw);
  pitch_history.push_back(this->resultPitch);
  time_points.push_back(current_time);

  if (fai_history.size() > 200) {
    fai_history.erase(fai_history.begin());
    yaw_history.erase(yaw_history.begin());
    pitch_history.erase(pitch_history.begin());
    time_points.erase(time_points.begin());
  }

  cv::Mat curve(300, 600, CV_8UC3, cv::Scalar(0, 0, 0));
  cv::Mat yaw_curve(300, 600, CV_8UC3, cv::Scalar(0, 0, 0));
  cv::Mat pitch_curve(300, 600, CV_8UC3, cv::Scalar(0, 0, 0));

  cv::line(curve, cv::Point(50, 250), cv::Point(550, 250),
           cv::Scalar(255, 255, 255), 1);
  cv::line(curve, cv::Point(50, 50), cv::Point(50, 250),
           cv::Scalar(255, 255, 255), 1);

  cv::line(yaw_curve, cv::Point(50, 250), cv::Point(550, 250),
           cv::Scalar(255, 255, 255), 1);
  cv::line(yaw_curve, cv::Point(50, 50), cv::Point(50, 250),
           cv::Scalar(255, 255, 255), 1);

  cv::line(pitch_curve, cv::Point(50, 250), cv::Point(550, 250),
           cv::Scalar(255, 255, 255), 1);
  cv::line(pitch_curve, cv::Point(50, 50), cv::Point(50, 250),
           cv::Scalar(255, 255, 255), 1);

  for (size_t i = 1; i < fai_history.size(); i++) {
    double x1 = 50 + (time_points[i - 1] - time_points[0]) /
                         (time_points.back() - time_points[0]) * 500;
    double y1 = 250 - fai_history[i - 1] * 200 / 360;
    double x2 = 50 + (time_points[i] - time_points[0]) /
                         (time_points.back() - time_points[0]) * 500;
    double y2 = 250 - fai_history[i] * 200 / 360;

    cv::line(curve, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(0, 255, 0),
             2);
  }

  for (size_t i = 1; i < yaw_history.size(); i++) {
    double x1 = 50 + (time_points[i - 1] - time_points[0]) /
                         (time_points.back() - time_points[0]) * 500;
    double y1 = 150 - (yaw_history[i - 1]) * 180 / 20; // 150是中心线(0度)的位置
    double x2 = 50 + (time_points[i] - time_points[0]) /
                         (time_points.back() - time_points[0]) * 500;
    double y2 = 150 - (yaw_history[i]) * 180 / 20;

    cv::line(yaw_curve, cv::Point(x1, y1), cv::Point(x2, y2),
             cv::Scalar(0, 0, 255), 2);
  }

  cv::line(yaw_curve, cv::Point(50, 150), cv::Point(550, 150),
           cv::Scalar(100, 100, 100), 1, cv::LINE_8);

  for (size_t i = 1; i < pitch_history.size(); i++) {
    double x1 = 50 + (time_points[i - 1] - time_points[0]) /
                         (time_points.back() - time_points[0]) * 500;
    double y1 = 250 - (pitch_history[i - 1]) * 180 / 25; // 直接显示绝对值
    double x2 = 50 + (time_points[i] - time_points[0]) /
                         (time_points.back() - time_points[0]) * 500;
    double y2 = 250 - (pitch_history[i]) * 180 / 25;

    cv::line(pitch_curve, cv::Point(x1, y1), cv::Point(x2, y2),
             cv::Scalar(255, 0, 0), 2);
  }

  cv::line(yaw_curve, cv::Point(50, 60), cv::Point(550, 60),
           cv::Scalar(50, 50, 50), 1, cv::LINE_8); // +10度
  cv::line(yaw_curve, cv::Point(50, 240), cv::Point(550, 240),
           cv::Scalar(50, 50, 50), 1, cv::LINE_8); // -10度
  cv::putText(yaw_curve, "+10°", cv::Point(20, 65), cv::FONT_HERSHEY_SIMPLEX,
              0.5, cv::Scalar(200, 200, 200), 1);
  cv::putText(yaw_curve, "0°", cv::Point(30, 155), cv::FONT_HERSHEY_SIMPLEX,
              0.5, cv::Scalar(200, 200, 200), 1);
  cv::putText(yaw_curve, "-10°", cv::Point(20, 245), cv::FONT_HERSHEY_SIMPLEX,
              0.5, cv::Scalar(200, 200, 200), 1);

  cv::line(pitch_curve, cv::Point(50, 130), cv::Point(550, 130),
           cv::Scalar(50, 50, 50), 1, cv::LINE_8); // 12.5度
  cv::putText(pitch_curve, "25°", cv::Point(25, 75), cv::FONT_HERSHEY_SIMPLEX,
              0.5, cv::Scalar(200, 200, 200), 1);
  cv::putText(pitch_curve, "12.5°", cv::Point(15, 135),
              cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(200, 200, 200), 1);
  cv::putText(pitch_curve, "0°", cv::Point(30, 255), cv::FONT_HERSHEY_SIMPLEX,
              0.5, cv::Scalar(200, 200, 200), 1);

  cv::putText(curve, "Fai", cv::Point(250, 30), cv::FONT_HERSHEY_SIMPLEX, 0.8,
              cv::Scalar(255, 255, 255), 1);
  cv::putText(yaw_curve, "Yaw", cv::Point(250, 30), cv::FONT_HERSHEY_SIMPLEX,
              0.8, cv::Scalar(255, 255, 255), 1);
  cv::putText(pitch_curve, "Pitch", cv::Point(250, 30),
              cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 255, 255), 1);

  cv::imshow("Fai", curve);
  cv::imshow("Yaw", yaw_curve);
  cv::imshow("Pitch", pitch_curve);
  cv::imshow("debugImg", this->debugImg);
  cv::waitKey(1);
#endif // VISUALIZE_PREDICTION

  // 结束计时并累计统计数据
  auto end_time = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
                      end_time - newton_start_time)
                      .count();
  newton_total_time += duration / 1000.0; // 转换为毫秒
  newton_frame_count++;
  is_newton_running = false;

  // 显示统计信息
  ShowTimeStatistics();
}

/**
 * @description: 输入w后，进行P参数的最优估计
 * @param {double} w
 * @param {double} &p1
 * @param {double} &p2
 * @param {double} &p3
 * @param {deque<double>} x_data  时间队列
 * @param {deque<double>} y_data  角速度队列
 * @return {double} 返回残差
 */
double WMPredict::Estim(double w, double &p1, double &p2, double &p3,
                        std::deque<double> x_data, std::deque<double> y_data) {
  std::vector<double> x1;
  std::vector<double> x2;
  std::vector<double> temp1;
  std::vector<double> temp2;
  std::vector<double> temp3;
  for (auto x : x_data) {
    x2.push_back(cos(w * x));
    x1.push_back(sin(w * x));
  }

  // 对两个向量进行操作
  std::transform(x1.begin(), x1.end(), x2.begin(), std::back_inserter(temp1),
                 [](double a, double b) { return a * b; });
  std::transform(x1.begin(), x1.end(), y_data.begin(),
                 std::back_inserter(temp2),
                 [](double a, double b) { return a * b; });
  std::transform(x2.begin(), x2.end(), y_data.begin(),
                 std::back_inserter(temp3),
                 [](double a, double b) { return a * b; });

  // 最小二乘法的求和
  double sum_x1x2 = std::accumulate(temp1.begin(), temp1.end(), 0.0);
  double sum_x1y = std::accumulate(temp2.begin(), temp2.end(), 0.0);
  double sum_x2y = std::accumulate(temp3.begin(), temp3.end(), 0.0);
  double sum_y = std::accumulate(y_data.begin(), y_data.end(), 0.0);
  double sum_x1 = std::accumulate(x1.begin(), x1.end(), 0.0);
  double sum_x2 = std::accumulate(x2.begin(), x2.end(), 0.0);
  double sum_x1x1 = std::accumulate(
      x1.begin(), x1.end(), 0.0, [](double a, double b) { return a + b * b; });
  double sum_x2x2 = std::accumulate(
      x2.begin(), x2.end(), 0.0, [](double a, double b) { return a + b * b; });

  // 定义矩阵
  cv::Mat_<double> A(3, 3);
  cv::Mat_<double> b(3, 1);
  A << sum_x1x1, sum_x1x2, sum_x1, sum_x1x2, sum_x2x2, sum_x2, sum_x1, sum_x2,
      y_data.size();
  b << sum_x1y, sum_x2y, sum_y;
  // 求解 Ax=b
  cv::Mat_<double> x = A.inv() * b;
  p1 = x(0);
  p2 = x(1);
  p3 = x(2);
  // 求出残差和
  double err_sum = 0;
  for (int i = 0; i < x1.size(); i++) {
    err_sum += pow(x1[i] * p1 + x2[i] * p2 + p3 - y_data[i], 2);
  }

  return err_sum;
}
/**
 * @description: 拟合函数，遍历w选取残差最小参数组
 * @param {deque<double>} x_data
 * @param {deque<double>} y_data
 * @param {GlobalParam} &gp
 * @param {Translator} &tr
 * @return {*}
 */
void WMPredict::ConvexOptimization(std::deque<double> x_data,
                                   std::deque<double> y_data, GlobalParam &gp,
                                   Translator &tr) {
  // 开始计时
  is_convex_running = true;
  convex_start_time = std::chrono::high_resolution_clock::now();

  for (int i = 0; i < y_data.size(); i++) {
    y_data[i] = abs(y_data[i]); // 保持对y_data取绝对值
  }
  double w0 = 0;
  double p1 = 0;
  double p2 = 0;
  double p3 = 0;
  double err_min = std::numeric_limits<double>::max();
  if (params_fixed) {
    w0 = fixed_w_big;
    p3 = fixed_b;

    double fai_step = 0.01;
    err_min = std::numeric_limits<double>::max(); // 重置err_min
    double best_fai = 0;
    double p1_best = 0, p2_best = 0;

    for (double fai_temp = -M_PI; fai_temp < M_PI; fai_temp += fai_step) {
      // 用固定的w_big和A0计算p1和p2
      double p1_temp = fixed_A0 * cos(fai_temp);
      double p2_temp = fixed_A0 * sin(fai_temp);

      // 计算当前fai的误差 (建议使用SSE，与Estim统一或修改Estim)
      double error_sum = 0.0;
      for (int i = 0; i < x_data.size(); i++) {
        double predicted =
            p1_temp * cos(w0 * x_data[i]) + p2_temp * sin(w0 * x_data[i]) + p3;
        double actual = y_data[i];
        error_sum += pow(predicted - actual, 2);
      }

      if (error_sum < err_min) {
        err_min = error_sum;
        p1_best = p1_temp;
        p2_best = p2_temp;
        best_fai = fai_temp;
      }
    }
    p1 = p1_best;
    p2 = p2_best;

    this->w_big = fixed_w_big;
    this->fai = best_fai;
    this->A0 = fixed_A0;
    this->b = fixed_b;
  } else {
    const double gr = (sqrt(5.0) + 1.0) / 2.0;
    double a = w_low;
    double b_w = w_up;
    double tol = 1e-6;

    double c = b_w - (b_w - a) / gr;
    double d = a + (b_w - a) / gr;

    double p1_c, p2_c, p3_c, p1_d, p2_d, p3_d;
    double fc = Estim(c, p1_c, p2_c, p3_c, x_data, y_data);
    double fd = Estim(d, p1_d, p2_d, p3_d, x_data, y_data);

    while (abs(c - d) > tol) {
      if (fc < fd) {
        b_w = d;
        d = c;
        fd = fc;
        p1_d = p1_c;
        p2_d = p2_c;
        p3_d = p3_c;
        c = b_w - (b_w - a) / gr;
        fc = Estim(c, p1_c, p2_c, p3_c, x_data, y_data);
      } else {
        a = c;
        c = d;
        fc = fd;
        p1_c = p1_d;
        p2_c = p2_d;
        p3_c = p3_d;
        d = a + (b_w - a) / gr;
        fd = Estim(d, p1_d, p2_d, p3_d, x_data, y_data);
      }
    }

    if (fc < fd) {
      w0 = c;
      err_min = fc;
      p1 = p1_c;
      p2 = p2_c;
      p3 = p3_c;
    } else {
      w0 = d;
      err_min = fd;
      p1 = p1_d;
      p2 = p2_d;
      p3 = p3_d;
    }
    // 如果需要更高精度，可以在最后调用一次Estim获取最终参数
    // err_min = Estim(w0, p1, p2, p3, x_data, y_data);

    this->w_big = w0;
    // 确保 p1 和 p2 不是零，避免 atan2(0,0)
    if (std::abs(p1) > 1e-9 || std::abs(p2) > 1e-9) {
      this->fai = atan2(p2, p1);
      // 计算 A0 时，优先使用 p1 或 p2 中绝对值较大的一个，提高数值稳定性
      if (std::abs(p1) >= std::abs(p2)) {
        this->A0 = p1 / cos(this->fai);
      } else {
        this->A0 = p2 / sin(this->fai);
      }
    } else {
      // 如果 p1 和 p2 都接近零，说明 A0 接近零，fai 无意义
      this->fai = 0;
      this->A0 = 0;
    }
    this->b = p3;
  }

  if (w_big < 2.1 && w_big > 1.9 && A0 < 1.01 && A0 > 0.99) {
    // got_angle_velocity = true;
  }
  //  this->A0 = 2.09 - this->b;
  // this->fai = 0;

  // this->A0 = 1;
  // this->b = 1.09;
  // this->w_big = 0;
  //  this->fai = 0;

  //  this->A0 = 0;
  //  this->b = 0;

  // 结束计时并累计统计数据
  auto end_time = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
                      end_time - convex_start_time)
                      .count();
  convex_total_time += duration / 1000.0; // 转换为毫秒
  convex_frame_count++;
  is_convex_running = false;

    std::cout << "w:" << this->w << std::endl;
    std::cout << "A:" << this->A0 << std::endl;
    std::cout << "fai:" << this->fai << std::endl;

  // 显示统计信息
  ShowTimeStatistics();
}

double WMPredict::ThetaToolForBig(double dt,
                                  double t0) // 计算t0->t0+dt的大符角度
{
  if (this->clockwise == 1) {
    return -(this->b * dt + this->A0 / this->w_big *
                                (cos(this->w_big * t0 + this->fai) -
                                 cos(this->w_big * (t0 + dt) + this->fai)));
  } else {
    return (this->b * dt + this->A0 / this->w_big *
                               (cos(this->w_big * t0 + this->fai) -
                                cos(this->w_big * (t0 + dt) + this->fai)));
  }
}

double WMPredict::f1A(double P0, double delta_theta_delay,
                      const cv::Mat &world2car, double fly_t0, double distance,
                      double v0) {
  cv::Mat world_point =
      (cv::Mat_<double>(4, 1) << r * cos(delta_theta_delay + w * fly_t0),
       -r * sin(delta_theta_delay + w * fly_t0), 0, 1.0);
  cv::Mat world_point_car = world2car * world_point;
  return sqrt(distance * distance - pow(world_point_car.at<double>(1, 0), 2)) -
         v0 * cos(P0) / k + v0 / k * cos(P0) * pow(Exp, -k * fly_t0);
}

double WMPredict::f2A(double P0, double delta_theta_delay,
                      const cv::Mat &world2car, double fly_t0, double v0) {
  cv::Mat world_point =
      (cv::Mat_<double>(4, 1) << r * cos(delta_theta_delay + w * fly_t0),
       -r * sin(delta_theta_delay + w * fly_t0), 0, 1.0);
  cv::Mat world_point_car = world2car * world_point;
  return -world_point_car.at<double>(1, 0) -
         (k * v0 * sin(P0) + g -
          (k * v0 * sin(P0) + g) * pow(Exp, -k * fly_t0) - g * k * fly_t0) /
             (k * k);
}

// 关于 P0 的偏导数：f1A 对 P0
double WMPredict::f1PA(double P, double x, double fly_t, double distance,
                       double v0) {
  return -(v0 / k) * sin(P) * (exp(-k * fly_t) - 1);
}

// 关于 fly_t0 的偏导数：f1A 对 fly_t0
double WMPredict::f1tA(double P0, double delta_theta_delay,
                       const cv::Mat &world2car, double fly_t0, double distance,
                       double v0) {
  double m00 = world2car.at<double>(1, 0);
  double m01 = world2car.at<double>(1, 1);
  double m03 = world2car.at<double>(1, 3);

  double A0 = m00 * r * cos(delta_theta_delay + w * fly_t0) -
              m01 * r * sin(delta_theta_delay + w * fly_t0) + m03;
  double dA0_dt = -r * w *
                  (m00 * sin(delta_theta_delay + w * fly_t0) +
                   m01 * cos(delta_theta_delay + w * fly_t0));
  double dPart = -(A0 * dA0_dt) / sqrt(distance * distance - A0 * A0);
  double dB_dt = -v0 * cos(P0) * exp(-k * fly_t0);
  return dPart + dB_dt;
}

double WMPredict::f2PA(double P, double z, double fly_t, double v0) {
  return -(v0 * cos(P) * (1 - exp(-k * fly_t))) / k;
}

// 关于 fly_t0 的偏导数：f2A 对 fly_t0
double WMPredict::f2tA(double P0, double delta_theta_delay,
                       const cv::Mat &world2car, double fly_t0, double v0) {
  double m10 = world2car.at<double>(1, 0);
  double m11 = world2car.at<double>(1, 1);
  double m13 = world2car.at<double>(1, 3);

  double A1 = m10 * r * cos(delta_theta_delay + w * fly_t0) -
              m11 * r * sin(delta_theta_delay + w * fly_t0) + m13;
  double dA1_dt = -r * w *
                  (m10 * sin(delta_theta_delay + w * fly_t0) +
                   m11 * cos(delta_theta_delay + w * fly_t0));
  // L = k*v0*sin(P0) + g
  double L = k * v0 * sin(P0) + g;
  double dQ_dt = (L * exp(-k * fly_t0) - g) / k;
  return -dA1_dt + dQ_dt;
}

double WMPredict::F1A(double P0, double delta_theta_delay,
                      const cv::Mat &world2car, double fly_t0, double distance,
                      double v0) {
  cv::Mat world_point =
      (cv::Mat_<double>(4, 1)
           << r * cos(delta_theta_delay +
                      ThetaToolForBig(fly_t0, this->Fire_time)),
       -r * sin(delta_theta_delay + ThetaToolForBig(fly_t0, this->Fire_time)),
       0, 1.0);
  cv::Mat world_point_car = world2car * world_point;

  return sqrt(distance * distance - pow(world_point_car.at<double>(1, 0), 2)) -
         v0 * cos(P0) / k + v0 * cos(P0) * pow(Exp, -k * fly_t0) / k;
}
double WMPredict::F2A(double P0, double delta_theta_delay,
                      const cv::Mat &world2car, double fly_t0, double v0) {
  cv::Mat world_point =
      (cv::Mat_<double>(4, 1)
           << r * cos(delta_theta_delay +
                      ThetaToolForBig(fly_t0, this->Fire_time)),
       -r * sin(delta_theta_delay + ThetaToolForBig(fly_t0, this->Fire_time)),
       0, 1.0);
  cv::Mat world_point_car = world2car * world_point;

  return -world_point_car.at<double>(1, 0) -
         (k * v0 * sin(P0) + g -
          (k * v0 * sin(P0) + g) * pow(Exp, -k * fly_t0) - g * k * fly_t0) /
             (k * k);
}
double WMPredict::F1PA(double P, double x, double fly_t, double distance,
                       double v0) {
  return v0 * sin(P) / k * (1 - pow(Exp, -k * fly_t));
}

double WMPredict::F1tA(double P0, double delta_theta_delay,
                       const cv::Mat &world2car, double fly_t0, double distance,
                       double v0) {
  double M00 = world2car.at<double>(1, 0);
  double M01 = world2car.at<double>(1, 1);
  double M03 = world2car.at<double>(1, 3);

  double angle = delta_theta_delay + ThetaToolForBig(fly_t0, this->Fire_time);

  double x = r * M00 * cos(angle) - r * M01 * sin(angle) + M03;

  double x_dot = -w * r * (M00 * sin(angle) + M01 * cos(angle));

  double d_sqrt = -(x * x_dot) / sqrt(distance * distance - x * x);

  double d_exp = -v0 * cos(P0) * exp(-k * fly_t0);

  return d_sqrt + d_exp;
}

double WMPredict::F2PA(double P, double z, double fly_t, double v0) {
  return v0 * cos(P) / k * (pow(Exp, -k * fly_t) - 1);
}

double WMPredict::F2tA(double P0, double delta_theta_delay,
                       const cv::Mat &world2car, double fly_t0, double v0) {
  double M10 = world2car.at<double>(1, 0);
  double M11 = world2car.at<double>(1, 1);
  double M13 = world2car.at<double>(1, 3);

  double angle = delta_theta_delay + ThetaToolForBig(fly_t0, this->Fire_time);

  double d_first_term = w * r * (M10 * sin(angle) + M11 * cos(angle));

  double B = k * v0 * sin(P0) + g;
  double dQ = (B * exp(-k * fly_t0) - g) / k;

  return d_first_term - dQ;
}

cv::Mat WMPredict::GetDebugImg() { return this->debugImg; }
void WMPredict::GiveDebugImg(cv::Mat debugImg) {
  this->debugImg = debugImg;
  // 每次更新debug图像时都显示统计信息
  ShowTimeStatistics();
}
/**
 * @description: 弹速单位转化为m/s,同时判断弹速能否激活能量机关
 * @param {Translator} &translator
 * @param {GlobalParam} &gp
 * @return {*}
 */
int WMPredict::BulletSpeedProcess(Translator &translator) {
  // 弹速过小时无法命中
  if (translator.message.r1 < 20) {
    std::cout << "弹速小" << std::endl;
    return 0;
  }
  return 1;
}

// 重置时间统计数据
void WMPredict::ResetTimeStatistics() {
  convex_total_time = 0.0;
  newton_total_time = 0.0;
  convex_frame_count = 0;
  newton_frame_count = 0;
  is_convex_running = false;
  is_newton_running = false;
}

// 在debugImg上显示统计信息
void WMPredict::ShowTimeStatistics() {
  if (!debugImg.empty()) {
    // 获取图像尺寸
    int img_height = debugImg.rows;
    int img_width = debugImg.cols;

    // 计算帧率
    double convex_fps = (convex_frame_count > 0 && convex_total_time > 0)
                            ? convex_frame_count / (convex_total_time / 1000.0)
                            : 0.0;
    double newton_fps = (newton_frame_count > 0 && newton_total_time > 0)
                            ? newton_frame_count / (newton_total_time / 1000.0)
                            : 0.0;
    double total_time = convex_total_time + newton_total_time;
    double total_frames = convex_frame_count + newton_frame_count;
    double total_fps = (total_frames > 0 && total_time > 0)
                           ? total_frames / (total_time / 1000.0)
                           : 0.0;

    // 将信息格式化为字符串
    //  char convex_info[100], newton_info[100], total_info[100];
    //  sprintf(convex_info, "ConvexOpt: %.2f ms/f, %.1f FPS",
    //          convex_frame_count > 0 ? convex_total_time / convex_frame_count
    //                                 : 0.0,
    //          convex_fps);
    //  sprintf(newton_info, "NewtonDsp: %.2f ms/f, %.1f FPS",
    //          newton_frame_count > 0 ? newton_total_time / newton_frame_count
    //                                 : 0.0,
    //          newton_fps);
    //  sprintf(total_info, "Total: %.2f ms/f, %.1f FPS",
    //          total_frames > 0 ? total_time / total_frames : 0.0, total_fps);

    // 定义文本样式和位置
    int font_face = cv::FONT_HERSHEY_SIMPLEX;
    double font_scale = 1.5;
    int thickness = 2;
    int baseline = 0;
    cv::Scalar text_color(0, 255, 255); // 黄色
    int line_height = 50;               // 行高
    int margin = 10;                    // 边距

    // 计算文本尺寸以确定位置
    //  cv::Size text_size_total = cv::getTextSize(
    //      total_info, font_face, font_scale, thickness, &baseline);
    //  cv::Size text_size_newton = cv::getTextSize(
    //      newton_info, font_face, font_scale, thickness, &baseline);
    //  cv::Size text_size_convex = cv::getTextSize(
    //      convex_info, font_face, font_scale, thickness, &baseline);

    // 计算起始位置 (右下角)
    //  int start_y = img_height - margin - 2 * line_height;
    //  int start_x_total = img_width - margin - text_size_total.width;
    //  int start_x_newton = img_width - margin - text_size_newton.width;
    //  int start_x_convex = img_width - margin - text_size_convex.width;

    //  // 在debugImg上显示信息
    //  cv::putText(debugImg, convex_info, cv::Point(start_x_convex, start_y),
    //              font_face, font_scale, text_color, thickness);
    //  cv::putText(debugImg, newton_info,
    //              cv::Point(start_x_newton, start_y + line_height), font_face,
    //              font_scale, text_color, thickness);
    //  cv::putText(debugImg, total_info,
    //              cv::Point(start_x_total, start_y + 2 * line_height),
    //              font_face, font_scale, text_color, thickness);
    //  std::cout << "ConvexOpt: " << convex_total_time / convex_frame_count
    //            << " ms/f, " << convex_fps << " FPS, F: " <<
    //            convex_frame_count
    //            << std::endl;
    //  std::cout << "NewtonDsp: " << newton_total_time / newton_frame_count
    //            << " ms/f, " << newton_fps << " FPS, F: " <<
    //            newton_frame_count
    //            << std::endl;
    //  std::cout << "Total: " << total_time / total_frames << " ms/f, "
    //            << total_fps << " FPS, F: " << total_frames << std::endl;
  }
}
