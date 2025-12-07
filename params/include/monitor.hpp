#include "opencv2/core/mat.hpp"
#if !defined(__MONITOR_HPP)
#define __MONITOR_HPP
#include <deque>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <globalParam.hpp>

bool polynomial_curve_fit(std::vector<cv::Point> &key_point, int n, cv::Mat &A);
bool polynomial_curve_fit(std::deque<cv::Point> &key_point, int n, cv::Mat &A);
void draw(std::deque<cv::Point3f> points3d, std::deque<double> time, int flag,float t_time,float t_points);
void drawRobotState(const Translator &ts, double times, double dt, double r);
void drawStationState(const Translator &ts, double times, double dt, double r);
void drawStat(std::deque<cv::Point3f> &points3d, std::deque<double> &times, Translator &translator);
void resizePic(cv::Mat &pic);
#endif //__MONITOR_HPP