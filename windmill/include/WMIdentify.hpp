#ifndef __WMIDENTIFY_HPP
#define __WMIDENTIFY_HPP

#include "traditional_detection.hpp" // 需要包含新的结构体定义
#include "globalParam.hpp"
#include <opencv2/opencv.hpp>
#include <deque>
#include <vector>
#include <chrono>

// 前向声明，避免不必要的头文件包含
class GlobalParam;

// ========================================================================
// ===                   新增的 TargetTracker 类声明                    ===
// ========================================================================
class TargetTracker {
public:
    int id;                         // 目标的唯一ID
    WindmillTarget last_detection;  // 最新一帧的检测结果
    bool is_tracking;               // 本帧是否被成功匹配和更新
    int frames_since_seen;          // 多少帧没看到了
    
    std::chrono::milliseconds starting_time; // 记录 tracker 被创建时的时间戳

    // 历史数据队列
    std::deque<double> time_list;
    std::deque<double> angle_list;      // 绝对角度(通过PnP第一帧法计算)
    std::deque<double> rot_angle_list;  // 像素角度(通过atan2计算)
    std::deque<double> angle_velocity_list;
    
    int list_stat; // 列表状态 (0:收集中, 1:数据足够可用于拟合)

    // PnP 姿态历史 (每个 tracker 都有自己的参考系)
    cv::Mat first_rvec, first_tvec;
    cv::Mat first_rotation_matrix;
    bool has_established_world_frame;

    // 运动模型参数 (由 WMPredict 拟合并回填)
    double w_big;
    double A0;
    double fai;
    double b;
    int fit_count;

    /**
     * @brief TargetTracker 构造函数
     */
    TargetTracker(int target_id, const WindmillTarget& initial_target);

    /**
     * @brief 更新该 tracker 的历史数据队列
     */
    void update_history(double time, double new_angle, double new_rot_angle, GlobalParam& gp);
};


// ========================================================================
// ===                   重构后的 WMIdentify 类声明                     ===
// ========================================================================
class WMIdentify {
private:
    GlobalParam *gp;
    int switch_INFO;
    int switch_ERROR;

    cv::Mat img;        // 当前处理的图像副本
    cv::Mat img_0;      // 最终用于显示的图像
    cv::Mat data_img;   // (可选) 数据图

    // 相机参数
    cv::Mat camera_matrix;
    cv::Mat dist_coeffs;

    // 三维模型
    std::vector<cv::Point3f> world_points;

    // --- 新增的多目标跟踪器核心 ---
    std::vector<TargetTracker> trackers;
    int next_target_id;

    /**
     * @brief (新增) 组装函数：将离散的零件组装成完整的目标
     */
    std::vector<WindmillTarget> group_targets(const KeyPoints& all_parts, const std::vector<std::vector<cv::Point>>& contours);

    /**
     * @brief (新增) 匹配与跟踪函数
     */
    void match_detections_to_trackers(const std::vector<WindmillTarget>& new_detections, double current_time);

public:
    /**
     * @brief WMIdentify的构造函数
     */
    WMIdentify(GlobalParam &);
    /**
     * @brief WMIdentify的析构函数
     */
    ~WMIdentify();

    /**
     * @brief (已重构) 识别主入口函数，执行多目标检测、跟踪与数据更新
     */
    void identifyWM(cv::Mat &input_img, Translator &translator);
    
    /**
     * @brief 清空所有跟踪器和状态
     */
    void clear();

    /**
     * @brief 输入图像的接口
     */
    void receive_pic(cv::Mat &input_img);

    /**
     * @brief (新增) 获取所有当前正在跟踪的目标
     * @return std::vector<TargetTracker>& 对 trackers 列表的引用
     */
    std::vector<TargetTracker>& getTrackers();
    
    /**
     * @brief 获取最终用于显示的图像
     */
    cv::Mat getImg0();

    // --- 以下为保留的辅助函数，逻辑不变 ---
    cv::Mat calculateTransformationMatrix(cv::Mat R_world2cam, cv::Mat tvec, Translator &translator);
    void calculateAngle(cv::Point2f blade_tip, cv::Mat rotation_matrix, cv::Mat tvec); // 注意：这个函数可能需要重构或废弃
    double calculatePhi(cv::Mat rotation_matrix, cv::Mat tvec);
    double calculateAlpha(cv::Mat R_world2cam, cv::Mat tvec, Translator &translator);
};

#endif // __WMIDENTIFY_HPP
