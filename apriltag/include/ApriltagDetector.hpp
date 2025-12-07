#ifndef APRILTAGDETECTOR_HPP
#define APRILTAGDETECTOR_HPP

extern "C" {
    #include "apriltag.h"
    #include "tag36h11.h"
}

#include <opencv2/opencv.hpp>


class ApriltagDetector
{
public:
    /**
    * @brief Construct a new Apriltag Detector object
    * 
    * @param tag_size Apriltag 正方形在现实中的大小，单位为 mm
    * @param fx 相机内参 fx
    * @param fy 相机内参 fy
    * @param cx 相机内参 cx
    * @param cy 相机内参 cy
    */
    ApriltagDetector(double tag_size, double fx, double fy, double cx, double cy, double k1, double k2, double k3, double p1, double p2);
    ~ApriltagDetector();
    /**
     * @brief 识别图片中的所有 apriltag，并给出其角点坐标和id（tag 的类型 id）
     * 
     * @param image 待识别的图片，要求为灰度图片
     * @param corners 识别出的 apriltag 的角点坐标，从 apriltag 正方形左下角开始，逆时针方向的四个角点。
     * 注意：无论 apriltag 是什么方向，都是从 tag 左下角的角点开始，而非形状上最左下方的角点。
     * @param ids 识别出的 apriltag 的 id
     */
    void detect(const cv::Mat &image, std::vector<std::vector<cv::Point2d>> &corners, std::vector<int> &ids);
    /**
     * @brief 给出一个 apriltag 识别结果以及其 id,在图片上绘制该 tag，绘制方式和官方示例相同，底边为绿色，左边为红色，剩余两边为蓝色
     * 
     * @param image 待绘制的目标图片
     * @param corners 一个 apriltag 识别结果，是正方形的四个角，从 apriltag 正方形左下角开始，逆时针方向的四个角点。
     * @param id 此 apriltag 的 id
     */
    void draw(cv::Mat &image, const std::vector<cv::Point2d> &corners, int id);
    /**
     * @brief 给出多个 apriltag 识别结果以及其 id,在图片上绘制所有 tag
     * 
     * @param image 待绘制的目标图片
     * @param corners 多个 apriltag 识别结果
     * @param ids 每个 apriltag 对应的 id
     */
    void draw(cv::Mat &image, const std::vector<std::vector<cv::Point2d>> &corners, const std::vector<int>& ids);
    /**
     * @brief 通过 apriltag 识别结果，解算相机位姿
     * @note 此方法下，apriltag 的世界坐标为 (-tag_size, -tag_size, 0), (tag_size, -tag_size, 0), (tag_size, tag_size, 0), (-tag_size, tag_size, 0)
     * @param corners 一个 apriltag 识别结果，是正方形的四个角，从 apriltag 正方形左下角开始，逆时针方向的四个角点。
     * @param rvec 旋转向量
     * @param tvec 平移向量
     */
    void solvePnP(const std::vector<cv::Point2d> &corners, cv::Mat &rvec, cv::Mat &tvec);
    /**
     * @brief 通过 apriltag 识别结果，解算相机位姿
     * 
     * @param corners 一个 apriltag 识别结果，是正方形的四个角，从 apriltag 正方形左下角开始，逆时针方向的四个角点。
     * @param objectPoints apriltag 的世界坐标
     * @param rvec 旋转向量
     * @param tvec 平移向量
     */
    void solvePnP(const std::vector<cv::Point2d> &corners, const std::vector<cv::Point3f> &objectPoints, cv::Mat &rvec, cv::Mat &tvec);
    /**
     * @brief 通过 apriltag 识别结果，解算相机位姿，返回内容为平移向量（xyz）和旋转角（yaw, pitch, roll
     * 
     * @param corners 一个 apriltag 识别结果，是正方形的四个角，从 apriltag 正方形左下角开始，逆时针方向的四个角点
     * @param rvec 旋转角（yaw, pitch, roll）
     * @param tvec 平移向量（xyz）
     */
    void solvePnP(const std::vector<cv::Point2d> &corners, cv::Vec3d& rvec, cv::Vec3d& tvec);
    /**
     * @brief 通过 apriltag 识别结果，解算相机位姿，返回内容为平移向量（xyz）和旋转角（yaw, pitch, roll）
     * 
     * @param corners 一个 apriltag 识别结果，是正方形的四个角，从 apriltag 正方形左下角开始，逆时针方向的四个角点。
     * @param objectPoints apriltag 的世界坐标
     * @param tvec 平移向量（xyz）
     * @param rvec 旋转角（yaw, pitch, roll）
     */
    void solvePnP(const std::vector<cv::Point2d> &corners, const std::vector<cv::Point3f> &objectPoints, cv::Vec3d& rvec, cv::Vec3d& tvec);
private:
    /**
     * @brief 将 pnp 解算出的旋转矩阵转换为欧拉角
     * 
     * @param rvec 旋转矩阵
     * @param euler 欧拉角向量，依次为 yaw, pitch, roll
     */
    void rVectorToEuler(const cv::Mat &rvec, cv::Vec3d &euler);
    apriltag_detector_t *td;
    apriltag_family_t *tf;
    double tag_size_;
    double fx_;
    double fy_;
    double cx_;
    double cy_;
    double k1_;
    double k2_;
    double k3_;
    double p1_;
    double p2_;
};

#endif // APRILTAGDETECTOR_HPP