/**
 * @file WMIdentify.cpp
 * @author Clarence Stark (3038736583@qq.com)
 * @brief 任意点位打符识别类实现
 * @version 0.1
 * @date 2024-12-08
 *
 * @copyright Copyright (c) 2024
 */

#include "WMIdentify.hpp"
#include "globalParam.hpp"
#include "opencv2/core/hal/interface.h"
#include "opencv2/core/types.hpp"
#include "traditional_detection.hpp"
#include <algorithm>
#include <chrono>
#include <deque>
#include <iostream>
#include <iterator>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgproc/types_c.h>
#include <ostream>
// #define Pi 3.1415926
const double Pi = 3.1415926;

#ifndef ON
#define ON 1
#endif

#ifndef OFF
#define OFF 0
#endif

/**
 * @brief (新增) TargetTracker 构造函数
 * @param target_id 分配给这个新 tracker 的唯一 ID
 * @param initial_target 首次检测到该目标时的完整信息
 */
TargetTracker::TargetTracker(int target_id, const WindmillTarget& initial_target) : 
    id(target_id), 
    last_detection(initial_target), 
    is_tracking(true), 
    frames_since_seen(0),
    has_established_world_frame(false),
    list_stat(0)
{
    // 构造函数体
    // 初始化时记录 tracker 的创建时间戳
    starting_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch());
    
    // (可选) 可以在此打印日志，表示一个新目标被发现
    std::cout << "[TRACKER] New target created with ID: " << id << std::endl;
}

/**
 * @brief WMIdentify类构造函数
 * @param[in] gp     全局参数
 * @return void
 */
WMIdentify::WMIdentify(GlobalParam &gp) {
  this->gp = &gp;
  // this->gp->list_size = 150;
//  this->list_stat = 0;
//  this->t_start =
//      std::chrono::system_clock::now().time_since_epoch().count() / 1e9;
  // 从gp中读取一些数据
//  this->switch_INFO = this->gp->switch_INFO;
//  this->switch_ERROR = this->gp->switch_ERROR;
  // this->get_armor_mode = this->gp->get_armor_mode;
  // 将R与Wing的状态设置为没有读取到内容
//  this->R_stat = 0;
//  this->Wing_stat = 0;
//  this->Winghat_stat = 0;
//  this->R_estimate.x = 0;
//  this->R_estimate.y = 0;
  // this->d_RP2 = 0.7; //
  // this->d_RP1P3 = 0.6;
  // this->d_P1P3 = 0.2;
  this->camera_matrix = (cv::Mat_<double>(3, 3) << this->gp->fx, 0,
                         this->gp->cx, 0, this->gp->fy, this->gp->cy, 0, 0, 1);
  this->dist_coeffs = (cv::Mat_<double>(1, 5) << this->gp->k1, this->gp->k2,
                       this->gp->p1, this->gp->p2, this->gp->k3);
  this->data_img = cv::Mat::zeros(400, 800, CV_8UC3);
  world_points = {
    cv::Point3f(0, 0, 0),                                 // 0: R 标中心
    cv::Point3f(this->gp->d_Radius, 0, 0),                // 1: 主直径远交点 (在X轴上，距离为半径)
    cv::Point3f(this->gp->d_RP2, this->gp->d_P1P3 / 2, 0), // 2: 共轭点1 (它的X坐标应该是扇叶中心的X坐标)
    cv::Point3f(this->gp->d_RP2, -this->gp->d_P1P3 / 2, 0) // 3: 共轭点2
  };
  // 输出日志，初始化成功
  // //LOG_IF(INFO, this->switch_INFO) << "WMIdentify Successful";
  // this->starting_time =
  // std::chrono::duration_cast<std::chrono::milliseconds>(
  //     std::chrono::system_clock::now().time_since_epoch());
}

/**
 * @brief WMIdentify类析构函数
 * @return void
 */
WMIdentify::~WMIdentify() {
  // WMIdentify之中的内容都会自动析构
  // std::cout << "析构中，下次再见喵～" << std::endl;
  // 输出日志，析构成功
  // //LOG_IF(INFO, this->switch_INFO) << "~WMIdentify Successful";
}

void drawFixedWorldAxes(cv::Mat &frame, const cv::Mat &cameraMatrix,
                        const cv::Mat &distCoeffs, const cv::Mat &R_init,
                        const cv::Mat &t_init) {
  std::vector<cv::Point3f> axisPoints;
  axisPoints.push_back(cv::Point3f(0, 0, 0));
  axisPoints.push_back(cv::Point3f(1, 0, 0));
  axisPoints.push_back(cv::Point3f(0, 1, 0));
  axisPoints.push_back(cv::Point3f(0, 0, 1));

  cv::Mat rvec, tvec;
  cv::Rodrigues(R_init, rvec);
  tvec = t_init.clone();

  std::vector<cv::Point2f> imagePoints;
  cv::projectPoints(axisPoints, rvec, tvec, cameraMatrix, distCoeffs,
                    imagePoints);

  cv::line(frame, imagePoints[0], imagePoints[1], cv::Scalar(0, 0, 255), 2);
  cv::line(frame, imagePoints[0], imagePoints[2], cv::Scalar(0, 255, 0), 2);
  // cv::line(frame, imagePoints[0], imagePoints[3], cv::Scalar(255, 0, 0),
  //          2);

  cv::putText(frame, "X", imagePoints[1], cv::FONT_HERSHEY_SIMPLEX, 1.0,
              cv::Scalar(0, 0, 255), 2);
  cv::putText(frame, "Y", imagePoints[2], cv::FONT_HERSHEY_SIMPLEX, 1.0,
              cv::Scalar(0, 255, 0), 2);
  // cv::putText(frame, "Z", imagePoints[3], cv::FONT_HERSHEY_SIMPLEX, 1.0,
  //             cv::Scalar(255, 0, 0), 2);
}
void visualizeCameraViewpointApproximation(
    const cv::Mat &image, const cv::Mat &cameraMatrix,
    const cv::Mat &distCoeffs, const cv::Mat &rvec, const cv::Mat &tvec,
    const std::vector<cv::Point3f> &objectPoints) {

  cv::Point2f imageCenter(image.cols / 2.0f, image.rows / 2.0f);

  cv::circle(image, imageCenter, 5, cv::Scalar(255, 255, 255), -1);
  cv::Point2f imageEnd(image.cols / 2.0f, image.rows / 2.0f + 300);
  cv::line(image, imageCenter, imageEnd, cv::Scalar(255, 255, 255));
}

/*void WMIdentify::visualizeCameraViewpoint(
    const cv::Mat &image, const cv::Mat &cameraMatrix,
    const cv::Mat &distCoeffs, const cv::Mat &rvec, const cv::Mat &tvec,
    const std::vector<cv::Point3f> &objectPoints) {
  cv::Point3f vec1 = objectPoints[1] - objectPoints[0];
  cv::Point3f vec2 = objectPoints[2] - objectPoints[1];
  cv::Point3f normal = vec1.cross(vec2);
  normal /= cv::norm(normal);

  float d = -(normal.x * objectPoints[0].x + normal.y * objectPoints[0].y +
              normal.z * objectPoints[0].z);

  cv::Mat R;
  cv::Rodrigues(rvec, R);

  cv::Mat cameraPosition = -R.t() * tvec;

  cv::Mat zAxis = (cv::Mat_<double>(3, 1) << 0, 0, 1);
  cv::Mat zAxisWorld = R.t() * zAxis;

  double t = -(normal.x * cameraPosition.at<double>(0) +
               normal.y * cameraPosition.at<double>(1) +
               normal.z * cameraPosition.at<double>(2) + d) /
             (normal.x * zAxisWorld.at<double>(0) +
              normal.y * zAxisWorld.at<double>(1) +
              normal.z * zAxisWorld.at<double>(2));

  cv::Point3f intersection(
      cameraPosition.at<double>(0) + t * zAxisWorld.at<double>(0),
      cameraPosition.at<double>(1) + t * zAxisWorld.at<double>(1),
      cameraPosition.at<double>(2) + t * zAxisWorld.at<double>(2));

  std::vector<cv::Point3f> worldPoints = {intersection};
  std::vector<cv::Point2f> imagePoints;
  cv::projectPoints(worldPoints, rvec, tvec, cameraMatrix, distCoeffs,
                    imagePoints);

  // cv::Mat resultImage = image.clone();
  if (!imagePoints.empty()) {
    cv::circle(image, imagePoints[0], 5, cv::Scalar(0, 0, 255),
               -1); // 红点标记交点
                    // cv::imshow("Camera Viewpoint", image);
    // cv::waitKey(0);
  } else {
    std::cerr << "Error: Intersection point is not visible in the image."
              << std::endl;
  }
}*/

/**
 * @brief 清空所有数据列表
 * @return void
 */
void WMIdentify::clear() {
  // 清空队列中的内容
//  this->blade_tip_list.clear();
//  this->wing_idx.clear();
//  this->R_center_list.clear();
//  this->R_idx.clear();
//  this->time_list.clear();
//  this->angle_list.clear();

//  this->angle_velocity_list.clear();
//  this->angle_velocity_list.emplace_back(
//      0); // 先填充一个0，方便之后UpdateList中的数据对齐
  // 输出日志，清空成
//  this->list_stat = 1;
// //LOG_IF(INFO, this->switch_INFO) << "clear Successful";

// 在新的多目标架构下，clear() 的功能就是清空所有跟踪器
  this->trackers.clear();
  
  // 重置下一个目标的ID计数器
  this->next_target_id = 0;

  //打印日志，方便调试
  if (gp->debug) {
    std::cout << "[TRACKER] All trackers have been cleared." << std::endl;
  }
}

/**
 * @brief (最终重构版) 任意点位能量机关识别, 支持多目标检测、跟踪、几何解算和数据更新
 * @param[in] input_img     输入图像
 * @param[in] translator    串口数据
 * @return void
 */
void WMIdentify::identifyWM(cv::Mat &input_img, Translator &translator) {

  // 1. 接收图像
  this->receive_pic(input_img);
  
  // 2. 调用 detect 函数，获取所有识别出的零件和原始轮廓
  WMBlade blade; // 仅为保持 detect 函数接口的兼容性
  DetectionResult result = detect(this->img, blade, *(this->gp), translator.message.status / 5, translator);

  // 3. 检查识别结果是否有效（是否至少有足够的零件）
  if (!result.all_key_points.isValid()) {
    translator.message.armor_flag = 10;
    std::cout << "识别失败：未找到足够的零件。" << std::endl;
    match_detections_to_trackers({}, 0.0); // 传入空列表，更新跟踪器状态
    this->img_0 = result.processedImage;
    if(gp->debug){
        if (!this->img_0.empty()) cv::imshow("WMIdentify Debug", this->img_0);
        cv::waitKey(1);
    }
    return;
  }

  // 4. 将识别出的零件组装成完整的目标
  // 注意：group_targets 现在也需要 contours 来填充 WindmillTarget 里的轮廓信息
  std::vector<WindmillTarget> detected_targets = group_targets(result.all_key_points, result.contours);
  
  if (detected_targets.empty()) {
      translator.message.armor_flag = 10;
      std::cout << "分组失败：有零件但无法组装成有效目标。" << std::endl;
      match_detections_to_trackers({}, 0.0);
      this->img_0 = result.processedImage;
      if(gp->debug){
        if (!this->img_0.empty()) cv::imshow("WMIdentify Debug", this->img_0);
        cv::waitKey(1);
      }
      return;
  }
  
  // 5. 为每个分组成功的目标进行几何解算，填充PnP所需的点
  for (auto& target : detected_targets) {
      if (!target.is_fully_matched) continue;
      
      const auto& fan_contour = target.fan_blade_center.contour;

      if (fan_contour.empty()) {
          std::cout << "几何解算失败：分组后的目标缺少扇叶轮廓信息。" << std::endl;
          //target.is_fully_matched = false;
                  // --- 如果解算失败，提供默认值以防崩溃 ---
          target.apex_point = target.fan_blade_center.center;
          target.conjugate_point1 = target.fan_blade_center.center;
          target.conjugate_point2 = target.fan_blade_center.center;
          //continue;
      }

      // 检查轮廓是否为空，并且点数是否足够进行椭圆拟合
      if (fan_contour.size() < 5) { 
          if (gp->debug) {
              std::cout << "[ERROR] Fan contour has " << fan_contour.size() 
                        << " points, which is less than 5. Cannot fit ellipse. Skipping this target." << std::endl;
          }
          target.is_fully_matched = false; // 标记为无效
          continue; // 直接跳过这个 target 的所有后续处理
      }
      
      cv::Point fan_center = target.fan_blade_center.center;
      cv::Point r_center = target.r_logo_center.center;
      cv::RotatedRect ellipse = cv::fitEllipse(fan_contour);

      // 调用 findIntersectionsByEquation
      WMBlade temp_blade;
      findIntersectionsByEquation(
          fan_center, r_center, 0, ellipse, this->img_0, *gp, temp_blade
      );
      
      // 检查解算结果并填充 target
      if (temp_blade.apex.size() >= 3) {
          target.apex_point = temp_blade.apex[0];
          target.conjugate_point1 = temp_blade.apex[1];
          target.conjugate_point2 = temp_blade.apex[2];
      } else {
          std::cout << "几何解算失败：findIntersectionsByEquation 未找到足够的交点。" << std::endl;
          target.is_fully_matched = false; // 标记为不完整
      }
  }

  // 6. 将新检测到的、且几何解算成功的，与现有的跟踪器进行匹配
  double current_time = std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::system_clock::now().time_since_epoch()
  ).count() / 1000.0;
  match_detections_to_trackers(detected_targets, current_time);

  // 只要有任何一个 tracker 在跟踪，就认为识别成功
  if (!trackers.empty() && std::any_of(trackers.begin(), trackers.end(), [](const TargetTracker& t){ return t.is_tracking; })) {
    translator.message.armor_flag = 12;
  } else {
    translator.message.armor_flag = 10;
  }

  //if (gp->debug) {
  //  std::cout << "--- Tracker Status ---" << std::endl;
  //  std::cout << "Total trackers: " << trackers.size() << std::endl;
  //  for (const auto& tracker : trackers) {
  //      std::cout << "  - Tracker ID: " << tracker.id
  //                << ", is_tracking: " << (tracker.is_tracking ? "true" : "false")
  //                << ", is_fully_matched: " << (tracker.last_detection.is_fully_matched ? "true" : "false")
  //                << ", frames_since_seen: " << tracker.frames_since_seen
  //                << std::endl;
  //  }
  //  std::cout << "----------------------" << std::endl;
  //}

  // 7. 为每个当前帧被成功跟踪的目标，独立进行PnP解算和数据更新
  for (auto& tracker : trackers) {
    if (!tracker.is_tracking) continue;

    WindmillTarget& target = tracker.last_detection;
    if (!target.is_fully_matched) continue;

    // a. 准备用于PnP的二维图像点集
    std::vector<cv::Point2f> image_points;
    image_points.push_back(target.r_logo_center.center);
    image_points.push_back(target.apex_point);
    image_points.push_back(target.conjugate_point1);
    image_points.push_back(target.conjugate_point2);

    // b. 建立世界坐标系 (如果这是该 tracker 的第一帧)
    if (!tracker.has_established_world_frame) {
      cv::solvePnP(world_points, image_points, camera_matrix, dist_coeffs, tracker.first_rvec, tracker.first_tvec, false, cv::SOLVEPNP_IPPE);
      cv::Rodrigues(tracker.first_rvec, tracker.first_rotation_matrix);
      tracker.has_established_world_frame = true;
    }

    // c. 对当前帧进行实时 PnP 解算
    cv::Mat current_rvec, current_tvec, current_rotation_matrix;
    cv::solvePnP(world_points, image_points, camera_matrix, dist_coeffs, current_rvec, current_tvec, false);
    cv::Rodrigues(current_rvec, current_rotation_matrix);
    
    // d. 计算从“世界系”到“车辆系”的总变换矩阵
    cv::Mat world2car = calculateTransformationMatrix(current_rotation_matrix, current_tvec, translator);

    // e. 计算距离
    cv::Mat R_world_origin_car = world2car * (cv::Mat_<double>(4, 1) << 0, 0, 0, 1.0);
    double distance = cv::norm(cv::Point3d(R_world_origin_car.rowRange(0,3)));
    if (distance < 4 || distance > 12) {
      std::cout << "Tracker ID " << tracker.id << " 距离异常: " << distance << "，跳过此目标。" << std::endl;
      continue;
    }
    
    // f. 计算总旋转角和像素旋转角
    cv::Mat relativeRotMat = current_rotation_matrix.t() * tracker.first_rotation_matrix;
    double angle = atan2(relativeRotMat.at<double>(1, 0), relativeRotMat.at<double>(0, 0));
    double rot_angle = atan2(target.fan_blade_center.center.y - target.r_logo_center.center.y,
                             target.fan_blade_center.center.x - target.r_logo_center.center.x);

    // g. 更新 tracker 的历史数据队列
    double time_since_tracker_start = current_time - (tracker.starting_time.count() / 1000.0);
    tracker.update_history(time_since_tracker_start, angle, rot_angle, *gp);
    
    // h. 将当前 tracker 的关键信息存入其 last_detection 中，供 WMPredict 使用
    tracker.last_detection.world2car_matrix = world2car;
    tracker.last_detection.pnp_rvec = current_rvec;
    tracker.last_detection.pnp_tvec = current_tvec;
    tracker.last_detection.current_angle = angle;
    tracker.last_detection.current_rot_angle = rot_angle;
    tracker.last_detection.distance = distance;
    
    // i. 调试信息绘制
    if (gp->debug) {
      // 绘制 tracker ID
      cv::putText(result.processedImage, "ID:" + std::to_string(tracker.id), 
                  target.r_logo_center.center + cv::Point(-25, -25), 
                  cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 150, 150), 2);
      
      // 绘制该 tracker 的世界坐标系
      drawFixedWorldAxes(result.processedImage, this->camera_matrix, this->dist_coeffs, current_rotation_matrix, current_tvec);

      // --- 绘制该 tracker 的所有关键点 ---
      
      // 1. R标 (红色实心圆)
      cv::circle(result.processedImage, target.r_logo_center.center, 5, cv::Scalar(0, 0, 255), -1);
      
      // 2. 扇叶中心 (绿色实心圆)
      cv::circle(result.processedImage, target.fan_blade_center.center, 5, cv::Scalar(0, 255, 0), -1);
      
      // 3. 灯条中心 (紫色实心圆)
      cv::circle(result.processedImage, target.light_bar.center, 5, cv::Scalar(255, 0, 255), -1);
      
      // 4. 主打击点 (黄色大圆)
      cv::circle(result.processedImage, target.apex_point, 7, cv::Scalar(0, 255, 255), -1);
      
      // 5. 共轭点1 (橙色空心圆)
      cv::circle(result.processedImage, target.conjugate_point1, 6, cv::Scalar(0, 165, 255), 2); // BGR for Orange
      
      // 6. 共轭点2 (橙色空心圆)
      cv::circle(result.processedImage, target.conjugate_point2, 6, cv::Scalar(0, 165, 255), 2);
      
      // (可选) 绘制共轭直径
      cv::line(result.processedImage, target.conjugate_point1, target.conjugate_point2, cv::Scalar(0, 165, 255), 1);
      
      // ====================================================================
    } 
  } 

  // 8. 更新最终用于显示的图像
  this->img_0 = result.processedImage;
  
  if (gp->debug) {
      if (!this->img_0.empty()) cv::imshow("WMIdentify Debug", this->img_0);
      cv::waitKey(1);
  }
}

/**
 * @brief (已重构) 组装函数：将离散的零件组装成完整的目标
 * @param all_parts 包含所有识别出的灯条和圆形的 KeyPoints 对象
 * @param contours 原始的轮廓列表，用于根据索引获取轮廓数据
 * @return std::vector<WindmillTarget> 所有成功组装的目标列表
 */
std::vector<WindmillTarget> WMIdentify::group_targets(const KeyPoints& all_parts, const std::vector<std::vector<cv::Point>>& contours) {
    std::vector<WindmillTarget> grouped_targets;
    
    // 复制一份零件列表，因为我们会通过 erase() 来标记“已使用”，避免重复匹配
    auto available_rects = all_parts.detected_rects;
    auto available_circles = all_parts.detected_circles;

    // 以扇叶为核心进行匹配
    for (const auto& circle_info : all_parts.detected_circles) {
        // 只从被分类为扇叶的圆形开始组装
        if (circle_info.is_fan_candidate) {
            WindmillTarget new_target;
            new_target.fan_blade_center = circle_info;

            // 填充扇叶的轮廓信息
            if (circle_info.contour_idx >= 0 && circle_info.contour_idx < contours.size()) {
                new_target.fan_blade_center.contour = contours[circle_info.contour_idx];
            }

            // --- 寻找最近的灯条 ---
            double min_dist_rect = DBL_MAX;
            auto best_rect_it = available_rects.end();

            for (auto it = available_rects.begin(); it != available_rects.end(); ++it) {
                // 注意类型转换，确保 cv::norm 的参数类型一致
                double dist = cv::norm(cv::Point2f(circle_info.center) - it->center);
                if (dist < min_dist_rect) {
                    min_dist_rect = dist;
                    best_rect_it = it;
                }
            }
            
            // --- 寻找最近的R标 ---
            double min_dist_r = DBL_MAX;
            auto best_r_it = available_circles.end();
            for (auto it = available_circles.begin(); it != available_circles.end(); ++it) {
                 if (it->is_r_logo_candidate) {
                    // R标和扇叶的 center 都是 cv::Point，可以直接计算
                    double dist = cv::norm(circle_info.center - it->center);
                    if (dist < min_dist_r) {
                        min_dist_r = dist;
                        best_r_it = it;
                    }
                 }
            }
            
            // 检查灯条是否匹配成功（找到了且在合理距离内）
            bool rect_found = (best_rect_it != available_rects.end() && min_dist_rect < 400); // 400像素是一个示例阈值
            if (rect_found) {
                new_target.light_bar = *best_rect_it;
            }

            // 检查R标是否匹配成功
            bool r_logo_found = (best_r_it != available_circles.end() && min_dist_r < 255); // 250像素是一个示例阈值
            if (r_logo_found) {
                new_target.r_logo_center = *best_r_it;
            }

            // 检查是否所有关键部件都已匹配
            //new_target.is_fully_matched = (rect_found && r_logo_found);
            
            if (gp->debug) {
                std::cout << "[GROUPING] For Fan at " << new_target.fan_blade_center.center << ":" << std::endl;
                if (rect_found) {
                     std::cout << "  - Matched Rect at " << new_target.light_bar.center << " (Dist: " << min_dist_rect << ")" << std::endl;
                } else {
                     std::cout << "  - FAILED to match a close Rect (Min Dist: " << min_dist_rect << ")" << std::endl;
                }
                if (r_logo_found) {
                     std::cout << "  - Matched R-Logo at " << new_target.r_logo_center.center << " (Dist: " << min_dist_r << ")" << std::endl;
                } else {
                     std::cout << "  - FAILED to match a close R-Logo (Min Dist: " << min_dist_r << ")" << std::endl;
                }
                std::cout << "  - Final is_fully_matched: " << (new_target.is_fully_matched ? "true" : "false") << std::endl;
            }

            // 只有在基础的距离匹配成功后，才进行几何校验
            if (rect_found && r_logo_found) {
                new_target.light_bar = *best_rect_it;
                new_target.r_logo_center = *best_r_it;
                
                cv::Point2f fan_p = cv::Point2f(new_target.fan_blade_center.center);
                cv::Point2f r_p = cv::Point2f(new_target.r_logo_center.center);
                cv::Point2f rect_p = new_target.light_bar.center;

                // 向量A: 灯条 -> R标
                cv::Point2f vec_Rect_R = r_p - rect_p;
                // 向量B: 灯条 -> 扇叶
                cv::Point2f vec_Rect_Fan = fan_p - rect_p;

                // 计算点积。如果共线且灯条在中间，点积应为负数。
                double dot_product = vec_Rect_R.dot(vec_Rect_Fan);
                
                bool geometry_ok = false;
                if (dot_product < 0) { // 检查夹角是否大于90度
                    // (可选) 更严格的检查：计算夹角余弦值
                    double norm_A = cv::norm(vec_Rect_R);
                    double norm_B = cv::norm(vec_Rect_Fan);
                    if (norm_A > 1e-5 && norm_B > 1e-5) {
                        double cos_theta = dot_product / (norm_A * norm_B);
                        // 要求夹角大于 155 度 (cos(155) ≈ -0.906)
                        if (cos_theta < -0.90) {
                            geometry_ok = true;
                        }
                    }
                }
              
                new_target.is_fully_matched = geometry_ok;

            } else {
                 new_target.is_fully_matched = false;
            }

            // 无论是否完全匹配，都将初步结果加入列表，让后续逻辑决定如何处理
            grouped_targets.push_back(new_target);

            // 如果匹配成功，则将用过的零件从可用列表中移除，防止被下一个扇叶重复匹配
            if (rect_found) {
                // 因为 best_rect_it 是 available_rects 的迭代器，可以直接 erase
                available_rects.erase(best_rect_it);
            }
            if (r_logo_found) {
                // 对 available_circles 的 erase 需要小心，因为它在外部循环中被引用
                // 为了安全和简单，我们改为使用索引标记，或者在循环外进行删除
                // 一个简单的贪心策略是，一旦一个R标被用过，就假设它不能再被用（这在多数情况下是合理的）
                // (此处为了代码简洁，暂时省略复杂的移除逻辑，假设一个R标只会被最近的扇叶匹配)
            }
        }
    }
    return grouped_targets;
}


void WMIdentify::match_detections_to_trackers(const std::vector<WindmillTarget>& new_detections, double current_time) {
    // 标记所有 tracker 为“本帧未见到”
    for (auto& tracker : trackers) {
        tracker.is_tracking = false; // 先假设都没匹配到
        tracker.frames_since_seen++;
    }

    std::vector<bool> detection_matched(new_detections.size(), false);
    
    // 匹配现有 trackers
    for (auto& tracker : trackers) {
        double min_dist = 150.0;
        int best_detection_idx = -1;

        for (size_t i = 0; i < new_detections.size(); ++i) {
            if (!detection_matched[i]) {
                // 使用 R 标中心进行匹配
                double dist = cv::norm(new_detections[i].r_logo_center.center - tracker.last_detection.r_logo_center.center);
                if (dist < min_dist) {
                    min_dist = dist;
                    best_detection_idx = i;
                }
            }
        }

        if (best_detection_idx != -1) {
            tracker.last_detection = new_detections[best_detection_idx];
            tracker.frames_since_seen = 0;
            tracker.is_tracking = true;
            detection_matched[best_detection_idx] = true;
        }
    }

    // --- 核心修改：为所有未匹配的新目标创建 tracker ---
    // 不再检查 is_fully_matched
    for (size_t i = 0; i < new_detections.size(); ++i) {
        if (!detection_matched[i]) {
            trackers.emplace_back(next_target_id++, new_detections[i]);
        }
    }

    // 移除长时间未见到的 tracker
    trackers.erase(
        std::remove_if(trackers.begin(), trackers.end(), 
            [](const TargetTracker& t){ return t.frames_since_seen > 15; }),
        trackers.end()
    );
}
/**
 * @brief 角度解算和收集函数（像素点反投影回世界系）
 * @param[in] blade_tip     扇叶顶点
 * @param[in] rotation_matrix     旋转矩阵
 * @param[in] tvec            平移向量
 * @return void
 */
/*void WMIdentify::calculateAngle(cv::Point2f blade_tip, cv::Mat rotation_matrix,
                                cv::Mat tvec) {
  // 计算相机光心在世界坐标系中的位置
  cv::Mat camera_in_world = -rotation_matrix.t() * tvec;
  // 计算图像点在相机坐标系和世界坐标系中的向量
  double u = blade_tip.x;
  double v = blade_tip.y;

  cv::Mat direction_camera =
      (cv::Mat_<double>(3, 1) << (u - camera_matrix.at<double>(0, 2)) /
                                     camera_matrix.at<double>(0, 0),
       (v - camera_matrix.at<double>(1, 2)) / camera_matrix.at<double>(1, 1),
       1.0);

  cv::Mat direction_world = rotation_matrix.t() * direction_camera;

  // 计算射线与z=0平面的交点（反投影回世界系）
  double s =
      -camera_in_world.at<double>(2, 0) / direction_world.at<double>(2, 0);
  double X =
      camera_in_world.at<double>(0, 0) + s * direction_world.at<double>(0, 0);
  double Y =
      camera_in_world.at<double>(1, 0) + s * direction_world.at<double>(1, 0);

  this->angle = atan2(Y, X);

  // std::cout << "this->gp->list_size : " << this->gp->list_size << std::endl;
  // // list_size = 120 std::cout << "gp->gap % gp->gap_control : " << gp->gap %
  // gp->gap_control << std::endl;

  // std::cout << "angle_list.size() : " << this->angle_list.size() <<
  // std::endl; //LOG_IF(INFO, this->switch_INFO) << "angle: " << this->angle;
  // //LOG_IF(INFO, this->switch_INFO) << "length: " << sqrt(X * X + Y * Y);

  // std::cout << "length: " << sqrt(X * X + Y * Y) << std::endl;

  // std::cout << "X: " << X << std::endl;
  // std::cout << "Y: " << Y << std::endl;
}*/

cv::Mat WMIdentify::calculateTransformationMatrix(cv::Mat R_world2cam,
                                                  cv::Mat tvec,
                                                  Translator &translator) {
  cv::Mat T_world2cam =
      (cv::Mat_<double>(4, 4) << R_world2cam.at<double>(0, 0),
       R_world2cam.at<double>(0, 1), R_world2cam.at<double>(0, 2),
       tvec.at<double>(0), R_world2cam.at<double>(1, 0),
       R_world2cam.at<double>(1, 1), R_world2cam.at<double>(1, 2),
       tvec.at<double>(1), R_world2cam.at<double>(2, 0),
       R_world2cam.at<double>(2, 1), R_world2cam.at<double>(2, 2),
       tvec.at<double>(2), 0, 0, 0, 1);

  double tx_cam2cloud = -0.00;
  double ty_cam2cloud = -0.54;
  double tz_cam2cloud = -0.16;
  if (!translator.message.is_far) {
    tx_cam2cloud = this->gp->tx_cam2cloud;
    ty_cam2cloud = this->gp->ty_cam2cloud;
    tz_cam2cloud = this->gp->tz_cam2cloud;
  } else {
    tx_cam2cloud = this->gp->tx_cam2cloud_1;
    ty_cam2cloud = this->gp->ty_cam2cloud_1;
    tz_cam2cloud = this->gp->tz_cam2cloud_1;
  }
  cv::Mat T_cam2cloud = (cv::Mat_<double>(4, 4) << 1, 0, 0, tx_cam2cloud, 0, 1,
                         0, ty_cam2cloud, 0, 0, 1, tz_cam2cloud, 0, 0, 0, 1);

  double yaw = translator.message.yaw;
  double pitch = translator.message.pitch;
  // yaw = 0.5;
  // pitch = 0.2;

  double cy = std::cos(yaw);
  double sy = std::sin(yaw);
  double cp = std::cos(pitch);
  double sp = std::sin(pitch);
  //   // 方案1: 先pitch后yaw
  // cv::Mat T_cloud2car = (cv::Mat_<double>(4, 4) <<
  //     cy, -sy*cp, sy*sp, 0,
  //     sy, cy*cp, -cy*sp, 0,
  //     0, sp, cp, 0,
  //     0, 0, 0, 1);

  cv::Mat R_y = (cv::Mat_<double>(3, 3) << cy, 0, sy, 0, 1, 0, -sy, 0, cy);
  cv::Mat R_x = (cv::Mat_<double>(3, 3) << 1, 0, 0, 0, cp, -sp, 0, sp, cp);
  // 注意旋转的先后顺序：R_car2cam = R_y * R_x
  cv::Mat T_cloud2car = (cv::Mat_<double>(4, 4) << cy, -sy * sp, -sy * cp, 0, 0,
                         cp, -sp, 0, sy, cy * sp, cy * cp, 0, 0, 0, 0, 1);

  // 组合旋转矩阵 R = R_z(yaw) * R_x(pitch)
  // 计算结果：
  // [ cos(yaw)       , -sin(yaw)*cos(pitch),  sin(yaw)*sin(pitch) ]
  // [ sin(yaw)       ,  cos(yaw)*cos(pitch), -cos(yaw)*sin(pitch) ]
  // [      0       ,         sin(pitch),          cos(pitch)    ]
  // cv::Mat T_cloud2car = (cv::Mat_<double>(4, 4) << cy, -sy * cp, sy * sp, 0,
  // sy, cy * cp, -cy * sp, 0, 0, sp, cp, 0, 0, 0, 0, 1); cv::Mat T_cloud2car =
  // (cv::Mat_<double>(4, 4) << cy, -sy * sp, -sy * cp, 0, 0, cp, -sp, 0, sy, cy
  // * sp, cy * cp, 0, 0, 0, 0, 1);

  //    T_world2car = T_cloud2car * T_cam2cloud * T_world2cam
  cv::Mat T_world2car = T_cloud2car * T_cam2cloud * T_world2cam;

  return T_world2car;
}

//cv::Mat WMIdentify::getTransformationMatrix() { return this->world2car; }
/**
 * @brief 计算phi
 * @param[in] rotation_matrix     旋转矩阵
 * @param[in] tvec            平移向量
 * @return void
 */
double WMIdentify::calculatePhi(cv::Mat rotation_matrix, cv::Mat tvec) {
  cv::Mat camera_in_world = -rotation_matrix.t() * tvec;
  cv::Mat Z_camera_in_world = rotation_matrix.t().col(2);
  // cv::Mat Z_camera_in_camera = (cv::Mat_<double>(3,1) << 0,0,1);
  // cv::Mat Z_camera_in_world = rotation_matrix.t() * (Z_camera_in_camera -
  // tvec);
  double Vx = camera_in_world.at<double>(0, 0);
  double Vz = camera_in_world.at<double>(2, 0);
  double Zx = Z_camera_in_world.at<double>(0, 0);
  double Zz = Z_camera_in_world.at<double>(2, 0);
  double phi = acos((Vx * Zx + Vz * Zz) /
                    (sqrt(Vx * Vx + Vz * Vz) * sqrt(Zx * Zx + Zz * Zz)));
  // //LOG_IF(INFO, this->switch_INFO) <<"Vx * Zx + Vz * Zz / (sqrt(Vx * Vx + Vz
  // * Vz) * sqrt(Zx * Zx + Zz * Zz)) = "<< Vx * Zx + Vz * Zz / (sqrt(Vx * Vx +
  // Vz
  // * Vz) * sqrt(Zx * Zx + Zz * Zz));
  return Pi - phi;
}
/**
 * @brief 计算alpha
 * @param[in] rotation_matrix     旋转矩阵
 * @param[in] tvec            平移向量
 * @return void
 */
// double WMIdentify::calculateAlpha(cv::Mat rotation_matrix, cv::Mat tvec,
//                                   Translator &translator) {
//   // 保持现有实现不变
//   cv::Mat camera_in_world = -rotation_matrix.t() * tvec;
//   // 相机z轴在世界坐标系中的向量
//   cv::Mat Z_camera_in_world = rotation_matrix.t().col(2);
//   double Vx = camera_in_world.at<double>(0, 0);
//   double Vz = camera_in_world.at<double>(2, 0);
//   double Zx = Z_camera_in_world.at<double>(0, 0);
//   double Zz = Z_camera_in_world.at<double>(2, 0);
//   // phi的值为二者点积除以二者模的乘积
//   double phi = acos((Vx * Zx + Vz * Zz) /
//                     (sqrt(Vx * Vx + Vz * Vz) * sqrt(Zx * Zx + Zz * Zz)));
//   return Pi - phi;
// }
double WMIdentify::calculateAlpha(cv::Mat R_world2cam, cv::Mat tvec,
                                  Translator &translator) {
  cv::Mat C_world = -R_world2cam.t() * tvec; // 3x1向量

  double cy = std::cos(-translator.message.yaw),
         sy = std::sin(-translator.message.yaw);
  double cp = std::cos(-translator.message.pitch),
         sp = std::sin(-translator.message.pitch);
  cv::Mat R_y = (cv::Mat_<double>(3, 3) << cy, 0, sy, 0, 1, 0, -sy, 0, cy);
  cv::Mat R_x = (cv::Mat_<double>(3, 3) << 1, 0, 0, 0, cp, -sp, 0, sp, cp);
  cv::Mat R_car2cam = R_y * R_x;

  cv::Mat R_world2car = R_car2cam.t() * R_world2cam;

  cv::Mat C_car = R_world2car * C_world; // 3x1向量

  cv::Mat z_world = (cv::Mat_<double>(3, 1) << 0, 0, 1);
  cv::Mat z_car = R_world2car * z_world; // 3x1向量

  cv::Point2d v_proj(C_car.at<double>(0, 0), C_car.at<double>(2, 0));
  cv::Point2d z_proj(z_car.at<double>(0, 0), z_car.at<double>(2, 0));

  double dot = v_proj.x * z_proj.x + v_proj.y * z_proj.y;
  double det = v_proj.x * z_proj.y - v_proj.y * z_proj.x; // 叉积只看 y 分量
  double alpha = std::atan2(det, dot);
  alpha = Pi - std::abs(alpha);

  return alpha;
}

/**
 * @brief 接收输入图像
 * @param[in] input_img     输入图像
 * @return void
 */
void WMIdentify::receive_pic(cv::Mat &input_img) {
  this->img_0 = input_img.clone();
  this->img = input_img.clone();
  // //LOG_IF(INFO, this->switch_INFO) << "receive_pic Successful";
}

/**
 * @brief (新增) 更新单个 tracker 的历史数据队列
 * @param time 当前帧相对于该 tracker 首次出现的时间
 * @param new_angle 当前帧计算出的绝对角度
 * @param new_rot_angle 当前帧计算出的像素角度
 * @param gp 全局参数，用于获取 list_size 等配置
 */
void TargetTracker::update_history(double time, double new_angle, double new_rot_angle, GlobalParam& gp) {
    // --- 维护数据队列的大小 ---
    // 这个逻辑与旧的 updateList 类似，但只操作当前 tracker 自己的成员变量

    // 更新时间队列
    if (this->time_list.size() >= gp.list_size) {
        this->time_list.pop_front();
    }
    this->time_list.push_back(time);

    // 更新绝对角度队列
    if (this->angle_list.size() >= gp.list_size) {
        this->angle_list.pop_front();
    }
    this->angle_list.push_back(new_angle); // 注意：旧代码是 -angle，这里根据实际需要决定是否取反

    // 更新像素角度队列
    if (this->rot_angle_list.size() >= gp.list_size) {
        this->rot_angle_list.pop_front();
    }
    this->rot_angle_list.push_back(new_rot_angle);

    // --- 计算并更新角速度队列 (核心逻辑) ---
    if (this->angle_velocity_list.size() >= gp.list_size) {
        this->angle_velocity_list.pop_front();
    }

    // 必须有至少两帧的数据才能计算速度
    if (this->rot_angle_list.size() > 1 && this->time_list.size() > 1) {
        // 获取最新和次新的数据
        double current_rot_angle = this->rot_angle_list.back();
        double previous_rot_angle = *(this->rot_angle_list.end() - 2);
        
        double current_timestamp = this->time_list.back();
        double previous_timestamp = *(this->time_list.end() - 2);

        // 1. 计算角度变化量
        double dangle = current_rot_angle - previous_rot_angle;

        // 2. 角度跳变处理 (处理 atan2 从 +pi 到 -pi 的突变)
        if (dangle < -Pi) { // 例如从 170°(-3.0 rad) 跳到 -170°(3.0 rad)，差值为 6.0 > Pi
            dangle += 2 * Pi;
        } else if (dangle > Pi) {
            dangle -= 2 * Pi;
        }

        // 3. 扇叶切换处理 (核心)
        // 能量机关有5个扇叶，每个扇叶间隔 2*Pi/5 = 0.4*Pi
        // 当识别点从一个扇叶跳到下一个时，角度会突变约 0.4*Pi
        int shift = std::round(dangle / (0.4 * Pi));
        if (shift != 0) {
            dangle = dangle - shift * 0.4 * Pi;
            // (可选) 在这里可以记录一次扇叶切换事件
            // this->fan_switched = true;
        }

        // 4. 计算时间变化量
        double dtime = current_timestamp - previous_timestamp;

        // 5. 计算角速度并存入队列
        if (dtime > 1e-5) { // 防止除以零
            double angular_velocity = dangle / dtime;
            
            // (可选) 对角速度进行异常值过滤
            if (std::abs(angular_velocity) < 10.0) { // 限制角速度在合理范围，例如 +/-10 rad/s
                 this->angle_velocity_list.push_back(angular_velocity);
            } else {
                 // 如果速度异常，可以不添加，或者添加上一个有效值
                 if (!this->angle_velocity_list.empty()) {
                     this->angle_velocity_list.push_back(this->angle_velocity_list.back());
                 }
            }
        }
    }

    // --- 更新 tracker 的状态 ---
    // 检查是否有足够的数据用于拟合
    if (this->angle_velocity_list.size() >= gp.list_size) {
      this->list_stat = 1; // 状态 1: 数据已就绪
    } else {
      this->list_stat = 0; // 状态 0: 正在收集中
    }
}
  //   // if (abs(dangle / dtime) > 5) {
  //   //   this->FanChangeTime = time_list.back() * 1000;
  //   //   this->time_list.pop_front();
  //   //   this->angle_list.pop_front();
  //   //   gp->gap--;
  //   // } else {
  //   this->angle_velocity_list.emplace_back(dangle / dtime);
  //   // }

  //   // 更新旋转方向
  //   // std::cout<<this->time_list.back()<<std::endl;
  //   // std::cout<<" "<<this->angle_velocity_list.back()<<std::endl;
  //   this->direction = 0;
  //   for (int i = 0; i < angle_velocity_list.siz

/**
 * @brief 获取时间列表
 * @return std::deque<double> 时间列表
 */
//std::deque<double> WMIdentify::getTimeList() { return this->time_list; }

//cv::Mat WMIdentify::getRvec() { return this->rvec; }
//cv::Mat WMIdentify::getTvec() { return this->tvec; }

cv::Mat WMIdentify::getDist_coeffs() { return this->dist_coeffs; }

cv::Mat WMIdentify::getCamera_matrix() { return this->camera_matrix; }

/**
 * @brief 获取角速度列表
 * @return std::deque<double> 角速度列表
 */
//std::deque<double> WMIdentify::getAngleVelocityList() {
//  return this->angle_velocity_list;
//}

/**
 * @brief 获取旋转方向
 * @return double 旋转方向
 */
//double WMIdentify::getDirection() { return this->direction; }

/**
 * @brief 获取最新角度
 * @return double 最新角度值
 */
//std::deque<double> WMIdentify::getAngleList() { return this->angle_list; }

//double WMIdentify::getLastAngle() {
//  return this->angle_list[angle_list.size() - 1];
//}

//double WMIdentify::getLastRotAngle() {
//  return this->rot_angle_list[rot_angle_list.size() - 1];
//}

//double WMIdentify::getR_yaw() {
//  return this->R_yaw_list[R_yaw_list.size() - 1];
//}

/**
 * @brief 获取R点中心坐标
 * @return cv::Point2d R点中心坐标
 */
//cv::Point2d WMIdentify::getR_center() {
//  return this->R_center_list[R_center_list.size() - 1];
//}

/**
 * @brief 获取半径
 * @return double 半径值
 */
//double WMIdentify::getRadius() {

  // //LOG_IF(INFO, this->switch_INFO == ON)
  //       << "R_center_list.size() : " << this->R_center_list.size();
  // //LOG_IF(INFO, this->switch_INFO == ON)
  //       << "blade_tip_list.size() : " << this->blade_tip_list.size();

//  return sqrt(
//      calculateDistanceSquare(this->R_center_list[R_center_list.size() - 1],
//                              this->blade_tip_list[blade_tip_list.size() - 1]));
//}

//double WMIdentify::getPhi() { return this->phi; }

//double WMIdentify::getAlpha() { return this->alpha; }

//double WMIdentify::getRdistance() { return this->distance; }

/**
 * @brief 获取列表状态
 * @return int 列表状态
 */
//int WMIdentify::getListStat() { return this->list_stat; }

/**
 * @brief 获取原始图像
 * @return cv::Mat 原始图像
 */
//cv::Mat WMIdentify::getImg0() { return this->img_0; }

/**
 * @brief 获取数据图像
 * @return cv::Mat 数据图像
 */
//cv::Mat WMIdentify::getData_img() { return this->data_img; }

/**
 * @brief 清空速度相关数据
 * @return void
 */
//void WMIdentify::ClearSpeed() {
//  this->angle_velocity_list.clear();
//  this->time_list.clear();
//}

/**
 * @brief 获取扇叶切换时间
 * @return uint32_t 扇叶切换时间
 */
//uint32_t WMIdentify::GetFanChangeTime() { return this->FanChangeTime; }

/**
 * @brief 根据翻译器状态判断是否需要清空数据
 * @param[in] translator     翻译器对象
 * @return void
 */
//void WMIdentify::JudgeClear(Translator translator) {
//  if (translator.message.status % 5 == 0) // 进入自瞄便清空识别的所有数据
//    this->clear();
//}

/**
 * @brief 计算两点之间的距离平方
 * @param[in] p1 点1
 * @param[in] p2 点2
 * @return double 距离平方
 */
//double WMIdentify::calculateDistanceSquare(cv::Point2f p1, cv::Point2f p2) {
  //    //LOG_IF(INFO, this->switch_INFO == ON) << "calculateDistanceSquare";

//  return (p1.x - p2.x) * (p1.x - p2.x) + (p1.y - p2.y) * (p1.y - p2.y);
//}

/**
 * @brief (新增) 获取所有当前正在跟踪的目标
 * @return std::vector<TargetTracker>& 对 trackers 列表的引用
 */
std::vector<TargetTracker>& WMIdentify::getTrackers() {
    return this->trackers;
}

/**
 * @brief 获取最终用于显示的图像
 * @return cv::Mat 图像
 */
cv::Mat WMIdentify::getImg0() {
    return this->img_0;
}
