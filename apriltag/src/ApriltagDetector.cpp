#include "ApriltagDetector.hpp"
#include "apriltag.h"
#include "common/zarray.h"
#include "opencv2/calib3d.hpp"
#include "opencv2/core/mat.hpp"
#include "opencv2/core/types.hpp"
#include "opencv2/imgproc/types_c.h"
#include "tag36h11.h"
#include <string>

ApriltagDetector::ApriltagDetector(double tag_size, double fx, double fy, double cx, double cy,
                                   double k1, double k2, double k3, double p1, double p2)
{
    // 创建 apriltag 的检测器
    td = apriltag_detector_create();
    // 创建 apriltag 的检测器的默认配置
    tf = tag36h11_create();
    // 设置 apriltag 检测器检测的 tag 类型
    apriltag_detector_add_family(td, tf);
    // 设置 apriltag 的检测器的参数
    td->quad_decimate = 2.0;
    td->quad_sigma = 0.5;
    td->refine_edges = 1;
    // 设置 apriltag 的检测器的 tag 的大小
    tag_size_ = tag_size;
    // 设置相机内参
    fx_ = fx;
    fy_ = fy;
    cx_ = cx;
    cy_ = cy;
    // 设置畸变系数
    k1_ = k1;
    k2_ = k2;
    k3_ = k3;
    p1_ = p1;
    p2_ = p2;
}

ApriltagDetector::~ApriltagDetector(){
    // 释放 apriltag 的检测器
    apriltag_detector_destroy(td);
    // 释放 apriltag 的检测器的默认配置
    tag36h11_destroy(tf);
}

void ApriltagDetector::detect(const cv::Mat &image, std::vector<std::vector<cv::Point2d>> &corners, std::vector<int> &ids){
    image_u8_t im = {image.cols, image.rows, image.cols, image.data};
    zarray_t* detections = apriltag_detector_detect(td, &im);

    corners.clear();
    ids.clear();
    // 单个结果的四角点
    std::vector<cv::Point2d> one_corner;

    for (int i=0;i < zarray_size(detections); i++){
        apriltag_detection_t* detection;
        // 获得单个识别结果
        zarray_get(detections, i, &detection);
        // 存储识别结果 id（tag 的 id）
        ids.push_back(detection->id);
        // 不用担心，push_back 会复制 one_corner 的所有内容，这样即使执行了 one_corner.clear()，之前已经添加的内容也不会丢失
        one_corner.clear();
        for (int i=0;i<4;i++){
            // 存储识别结果的四个角点
            one_corner.emplace_back(detection->p[i][0], detection->p[i][1]);
        }
        corners.push_back(one_corner);
    }
    // 清除识别结果，防止内存泄漏
    apriltag_detections_destroy(detections);
}

void ApriltagDetector::draw(cv::Mat &image, const std::vector<cv::Point2d> &corners, int id){
    // 绘制矩形框
    cv::line(image, corners[0], corners[1], cv::Scalar(0,255, 0), 2);
    cv::line(image, corners[0], corners[3], cv::Scalar(0, 0, 255), 2);
    cv::line(image, corners[1], corners[2], cv::Scalar(255, 0, 0), 2);
    cv::line(image, corners[2], corners[3], cv::Scalar(255, 0, 0), 2);
    // 绘制 id
    std::string text = std::to_string(id);
    int fontface = cv::FONT_HERSHEY_SCRIPT_SIMPLEX;
    double fontscale = 1.0;
    int baseline;
    cv::Size textsize = cv::getTextSize(text, fontface, fontscale, 2,
                                    &baseline);
    // 计算 tag 的中心点
    int width_center = (corners[0].x + corners[1].x + corners[2].x + corners[3].x) / 4;
    int height_center = (corners[0].y + corners[1].y + corners[2].y + corners[3].y) / 4;
    // 把 id 绘制在 tag 的正中心处
    cv::putText(image, text, cv::Point(width_center-textsize.width/2,
                                height_center+textsize.height/2),
            fontface, fontscale, cv::Scalar(0xff, 0x99, 0), 2);
}

void ApriltagDetector::draw(cv::Mat &image, const std::vector<std::vector<cv::Point2d>> &corners, const std::vector<int> &ids){
    // 检查两个数组的大小是否相同
    assert(corners.size() == ids.size());

    for (int i=0;i<corners.size();i++){
        draw(image, corners[i], ids[i]);
    }
}


void ApriltagDetector::solvePnP(const std::vector<cv::Point2d> &corners, cv::Mat &rvec, cv::Mat &tvec){
    // 世界坐标系下的四个角点
    std::vector<cv::Point3f> objectPoints;
    objectPoints.emplace_back(-tag_size_ / 2, tag_size_ / 2, 0);
    objectPoints.emplace_back(tag_size_ / 2, tag_size_ / 2, 0);
    objectPoints.emplace_back(tag_size_ / 2, -tag_size_ / 2, 0);
    objectPoints.emplace_back(-tag_size_ / 2, -tag_size_ / 2, 0);
    std::vector<cv::Point2d> correctImagePoints;
    correctImagePoints.push_back(corners[3]);
    correctImagePoints.push_back(corners[2]);
    correctImagePoints.push_back(corners[1]);
    correctImagePoints.push_back(corners[0]);
    // 这里不直接调用 solvePnP 是为了使用 IPPE_SQUARE 而不是下方默认的 IPPE 方法。
    // 旋转向量
    cv::Mat rvec_;
    // 平移向量
    cv::Mat tvec_;
    // 相机内参
    cv::Mat cameraMatrix = (cv::Mat_<double>(3, 3) << fx_, 0, cx_, 0, fy_, cy_, 0, 0, 1);
    // 畸变系数
    cv::Mat distCoeffs = (cv::Mat_<double>(5, 1) << k1_, k2_, p1_, p2_, k3_);
    // 通过 solvePnP 解算相机位姿
    cv::solvePnP(objectPoints, correctImagePoints, cameraMatrix, distCoeffs, rvec_, tvec_, false, cv::SOLVEPNP_IPPE_SQUARE);
    // 旋转向量转换为旋转矩阵
    cv::Rodrigues(rvec_, rvec);
    tvec = tvec_;
}

void ApriltagDetector::solvePnP(const std::vector<cv::Point2d> &corners, const std::vector<cv::Point3f> &objectPoints, cv::Mat &rvec, cv::Mat &tvec){
    // 旋转向量
    cv::Mat rvec_;
    // 平移向量
    cv::Mat tvec_;
    // 相机内参
    cv::Mat cameraMatrix = (cv::Mat_<double>(3, 3) << fx_, 0, cx_, 0, fy_, cy_, 0, 0, 1);
    // 畸变系数
    cv::Mat distCoeffs = (cv::Mat_<double>(5, 1) << k1_, k2_, p1_, p2_, k3_);
    // 通过 solvePnP 解算相机位姿
    cv::solvePnP(objectPoints, corners, cameraMatrix, distCoeffs, rvec_, tvec_, false, cv::SOLVEPNP_IPPE);
    // 旋转向量转换为旋转矩阵
    cv::Rodrigues(rvec_, rvec);
    tvec = tvec_;
}

void ApriltagDetector::solvePnP(const std::vector<cv::Point2d> &corners, cv::Vec3d& rvec, cv::Vec3d& tvec){
    cv::Mat rvec_, tvec_;
    this->solvePnP(corners, rvec_, tvec_);
    // 平移向量
    tvec = {tvec_.at<double>(0), tvec_.at<double>(1), tvec_.at<double>(2)};
    rVectorToEuler(rvec_, rvec);
}

void ApriltagDetector::solvePnP(const std::vector<cv::Point2d> &corners, const std::vector<cv::Point3f> &objectPoints, cv::Vec3d& rvec, cv::Vec3d& tvec){
    // 和上一个函数差不多，只不过这里返回的是旋转角和平移向量
    cv::Mat rvec_, tvec_;
    this->solvePnP(corners, objectPoints, rvec_, tvec_);
    // 平移向量
    tvec = {tvec_.at<double>(0), tvec_.at<double>(1), tvec_.at<double>(2)};
    rVectorToEuler(rvec_, rvec);
}

void ApriltagDetector::rVectorToEuler(const cv::Mat &rvec_, cv::Vec3d &euler){
    // 旋转角
    double yaw = std::atan2(rvec_.at<double>(0, 2), rvec_.at<double>(2, 2));
    if (yaw >= 0)
    {
        yaw = -(CV_PI - yaw);
    }
    else
    {
        yaw = CV_PI + yaw;
    }
    euler[0] = -yaw / CV_PI * 180;
    euler[1] = std::atan2(-rvec_.at<double>(1, 2), std::sqrt(rvec_.at<double>(1, 0)
            * rvec_.at<double>(1, 0) + rvec_.at<double>(1, 1) * rvec_.at<double>(1, 1)))
            / CV_PI * 180;
    double roll = -std::atan2(rvec_.at<double>(1, 0), rvec_.at<double>(1, 1));
    while (roll < -CV_PI / 4){
        roll += CV_PI / 2;
    }
    while (roll > CV_PI / 4){
        roll -= CV_PI / 2;
    }
    euler[2] = roll / CV_PI * 180;
}
