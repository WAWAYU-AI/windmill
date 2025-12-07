#ifndef TRADITIONAL_DETECTION_HPP
#define TRADITIONAL_DETECTION_HPP

#include "globalParam.hpp"
#include <opencv2/opencv.hpp>
#include <vector>

//存储单个识别出的矩形
struct DetectedRectangle {
    int contour_idx;        // 在原始轮廓列表中的索引
    cv::Point2f center;     // 中心点
    cv::RotatedRect rect;   // 旋转矩形
};

// 存储单个识别出的圆形（扇叶或R标）信息
struct DetectedCircle {
    int contour_idx;
    cv::Point center;
    double area;
    int child_count;
    bool is_fan_candidate;    // 是否可能是扇叶
    bool is_r_logo_candidate; // 是否可能是R标
    std::vector<cv::Point> contour;
};

struct KeyPoints {
  std::vector<std::vector<cv::Point>> circleContours;
  std::vector<double> circleAreas;
  std::vector<double> circularities;
  std::vector<cv::Point2f> rectCenters;
  std::vector<cv::Point> circlePoints;
  // 定义面积范围常量
  static constexpr double min_low = 150.0;
  static constexpr double min_high = 1800.0;
  static constexpr double max_low = 2500.0;
  static constexpr double max_high = 15000.0;

  std::vector<DetectedRectangle> detected_rects; // 存储所有识别到的灯条
  std::vector<DetectedCircle> detected_circles;    // 存储所有识别到的圆

  bool isValid() const {
    // 1. 检查是否至少有一个灯条被识别
    if (detected_rects.empty()) {
        return false;
    }

    // 2. 在所有识别到的圆形中，检查是否至少有一个是扇叶候选
    bool found_at_least_one_fan = false;
    for (const auto& circle : detected_circles) {
        if (circle.is_fan_candidate) {
            found_at_least_one_fan = true;
            break; // 找到了就可以退出循环
        }
    }

    // 只有同时找到了灯条和扇叶，才认为识别结果是“有效的”
    return found_at_least_one_fan;
}
};

struct DetectionResult {
  std::vector<cv::Point> intersections;
  std::vector<cv::Point> circlePoints;
  cv::Mat processedImage;
  double processingTime;
  KeyPoints all_key_points;
  std::vector<std::vector<cv::Point>> contours;
};

// 描述一个完整能量机关目标的数据结构
struct WindmillTarget {
    // 基础零件
    DetectedRectangle light_bar;
    DetectedCircle fan_blade_center;
    DetectedCircle r_logo_center;
    
    // 用于PnP的几何点 (由 WMIdentify 的 group_targets 填充)
    cv::Point2f apex_point;
    cv::Point2f conjugate_point1;
    cv::Point2f conjugate_point2;

    // 姿态信息 (由 WMIdentify 的 identifyWM 填充)
    cv::Mat world2car_matrix;
    cv::Mat pnp_rvec;
    cv::Mat pnp_tvec;
    double current_angle;
    double current_rot_angle;
    double distance;

    bool is_fully_matched;
    
    // 添加一个默认构造函数，以避免未初始化问题
    WindmillTarget() : is_fully_matched(false), distance(0.0) {}
};

DetectionResult detect(const cv::Mat &inputImage, WMBlade &blade,
                       GlobalParam &gp, int is_blue, Translator &translator);

KeyPoints detect_key_points(
    const std::vector<std::vector<cv::Point>> &contours, 
    const std::vector<cv::Vec4i> &hierarchy,
    cv::Mat &processedImage,
    WMBlade &blade,
    GlobalParam &gp
);

std::vector<cv::Point>
findIntersectionsByEquation(const cv::Point &center1, const cv::Point &center2,
                            double radius, const cv::RotatedRect &ellipse,
                            cv::Mat &pic, GlobalParam &gp, WMBlade &blade);

                            

#endif // TRADITIONAL_DETECTION_HPP
