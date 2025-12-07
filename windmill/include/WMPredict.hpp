#ifndef _PREDICT_HPP
#define _PREDICT_HPP
#include "globalParam.hpp"
// #include <ceres/ceres.h>
#include "WMIdentify.hpp"
#include <chrono>
#include <deque>
#include <fstream>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/core/types.hpp>
class WMPredict {
private:
  // 求解小符方程
  double f1(double P0, double fly_t0, double theta_0, double v0);
  double f2(double P0, double fly_t0, double theta_0, double v0);
  double f1P(double P, double fly_t, double theta_0, double v0);
  double f1t(double P, double fly_t, double theta_0, double v0);
  double f2P(double P, double fly_t, double theta_0, double v0);
  double f2t(double P, double fly_t, double theta_0, double v0);

  double f1A(double P0, double delta_theta_delay, const cv::Mat &world2car,
             double fly_t0, double distance, double v0);
  double f2A(double P0, double delta_theta_delay, const cv::Mat &world2car,
             double fly_t0, double v0);
  double f1PA(double P, double x, double fly_t, double distance, double v0);
  double f1tA(double P0, double delta_theta_delay, const cv::Mat &world2car,
              double fly_t0, double distance, double v0);
  double f2PA(double P, double z, double fly_t, double v0);
  double f2tA(double P0, double delta_theta_delay, const cv::Mat &world2car,
              double fly_t0, double v0);

  // 求解大符方程
  double F1(double P0, double fly_t0, double theta_0, double v0);
  double F2(double P0, double fly_t0, double theta_0, double v0);
  double F1P(double P, double fly_t, double theta_0, double v0);
  double F1t(double P, double fly_t, double theta_0, double v0);
  double F2P(double P, double fly_t, double theta_0, double v0);
  double F2t(double P, double fly_t, double theta_0, double v0);

  double F1A(double P0, double delta_theta_delay, const cv::Mat &world2car,
             double fly_t0, double distance, double v0);
  double F2A(double P0, double delta_theta_delay, const cv::Mat &world2car,
             double fly_t0, double v0);
  double F1PA(double P, double x, double fly_t, double distance, double v0);
  double F1tA(double P0, double delta_theta_delay, const cv::Mat &world2car,
              double fly_t0, double distance, double v0);
  double F2PA(double P, double z, double fly_t, double v0);
  double F2tA(double P0, double delta_theta_delay, const cv::Mat &world2car,
              double fly_t0, double v0);
  double ThetaToolForBig(double dt, double t0);
  void CeresFitting(std::deque<double> time_data_queue,
                    std::deque<double> angle_data_queue, double start_time);
  // debug - 画图
  cv::Point2f CalPointGuess(double theta);
  cv::Mat debugImg;
  cv::Mat data_img;
  cv::Mat smoothData_img;
  cv::Point2d R_center;
  std::deque<double> y_data_s;
  std::deque<double> x_data_s;
  cv::Mat rvec;          // 旋转向量
  cv::Mat tvec;          // 平移向量
  cv::Mat camera_matrix; // 相机内参矩阵
  cv::Mat dist_coeffs;   // 畸变系数

  int direction;
  double Radius; // 能量机关半径，修正角度用
  double Fire_time;
  double First_fit; // 是否为初次拟合1，0
  int clockwise;    // 1顺时针，0逆时针,初始化为-1,表示未确定
  //====大符速度参数======//
  double A0;
  double w_big;
  double b;
  double fai;
  double now_time;
  double w; // 小符角速度

  // 添加新的成员变量来跟踪最大最小值
  double min_pitch;
  double max_pitch;
  double min_yaw;
  double max_yaw;
  double resultYaw;
  double resultPitch;
  std::ofstream log_file; // 添加日志文件流

  double delta_t;

  // 时间统计相关变量
  std::chrono::high_resolution_clock::time_point convex_start_time;
  std::chrono::high_resolution_clock::time_point newton_start_time;
  double convex_total_time = 0.0; // 累计时间(ms)
  double newton_total_time = 0.0; // 累计时间(ms)
  int convex_frame_count = 0;     // 帧计数
  int newton_frame_count = 0;     // 帧计数
  bool is_convex_running = false; // 是否正在运行
  bool is_newton_running = false; // 是否正在运行

  // 在debugImg上显示统计信息
  void ShowTimeStatistics();

public:
  WMPredict(GlobalParam &gp);
  int StartPredict(Translator &translator, GlobalParam &gp, WMIdentify &WMI);

  void thetaAmend(double &theta);
  int BulletSpeedProcess(Translator &translator);
  //void UpdateData(WMIdentify &WMI, Translator translator);
  void UpdateData(WMIdentify &WMI);
  int Fit(std::deque<double> time_list, std::deque<double> angle_velocity_list,
          GlobalParam &gp, Translator &tr);
  void NewtonDspSmall(double theta_0, double alpha, Translator &translator,
                      GlobalParam &gp, double R_yaw);
  void NewtonDspBig(double theta_0, double alpha, Translator &translator,
                    GlobalParam &gp, double R_yaw);
  void NewtonDspBigAnyPos(cv::Mat world2car, Translator &translator,
                          GlobalParam &gp, double R_yaw, double R_distance,
                          const cv::Mat& rvec, const cv::Mat& tvec);
  void NewtonDspSmallAnyPos(cv::Mat world2car, Translator &translator,
                            GlobalParam &gp, double R_yaw,
                            const cv::Mat& rvec, const cv::Mat& tvec);

  double Estim(double w, double &p1, double &p2, double &p3,
               std::deque<double> x_data, std::deque<double> y_data);
  void ConvexOptimization(std::deque<double> x_data, std::deque<double> y_data,
                          GlobalParam &gp, Translator &tr);

  void ResultLog(Translator &translator, GlobalParam &gp, double R_yaw);
  void GiveDebugImg(cv::Mat debugImg);
  cv::Mat GetDebugImg();
  void ResetTimeStatistics(); // 重置时间统计

  // 获取时间统计信息
  double GetConvexTotalTime() const { return convex_total_time; }
  double GetNewtonTotalTime() const { return newton_total_time; }
  int GetConvexFrameCount() const { return convex_frame_count; }
  int GetNewtonFrameCount() const { return newton_frame_count; }
};

#endif // _PREDICT_HPP
