/**
 * @file traditional_detection.cpp
 * @author Clarence Stark (3038736583@qq.com)
 * @brief 用于传统算法检测
 * @version 0.1
 * @date 2025-01-04
 *
 * @copyright Copyright (c) 2025
 */

#include "globalParam.hpp"
#include "opencv2/core/types.hpp"
#include <chrono>
#include <cmath>
#include <csignal>
#include <iostream>
#include <numeric>
#include <opencv2/opencv.hpp>
    //gp.initGlobalParam(target_color);
    // 强制开启调试模式，以便在图像上看到绘制的识别结果
    //gp.debug = true; 

#include <traditional_detection.hpp>

using namespace cv;
using namespace std;
using namespace std::chrono;
// 计算两点点距
double computeDistance(Point p1, Point p2) {
  return sqrt(pow(p1.x - p2.x, 2) + pow(p1.y - p2.y, 2));
}

/**
 * @brief 将图像进行偏航角度的透视变换
 * @param inputImage 输入图像
 * @param yawFactor 偏航角度因子
 * @return 变换后的图像
 */
cv::Mat applyYawPerspectiveTransform(const cv::Mat &inputImage,
                                     float yawFactor) {
  // 检查输入图像是否为空
  if (inputImage.empty()) {
    std::cerr << "输入图像为空！" << std::endl;
    return cv::Mat();
  }

  int rows = inputImage.rows;
  int cols = inputImage.cols;

  std::vector<cv::Point2f> pts1 = {cv::Point2f(0, 0), cv::Point2f(cols, 0),
                                   cv::Point2f(0, rows),
                                   cv::Point2f(cols, rows)};

  float horizontalOffset = cols * yawFactor; // 根据输入的因子计算水平偏移量
  std::vector<cv::Point2f> pts2 = {
      cv::Point2f(horizontalOffset, 0), cv::Point2f(cols - horizontalOffset, 0),
      cv::Point2f(horizontalOffset / 2, rows),
      cv::Point2f(cols - horizontalOffset / 2, rows)};

  cv::Mat M = cv::getPerspectiveTransform(pts1, pts2);

  cv::Mat warpedImage;
  cv::warpPerspective(inputImage, warpedImage, M, inputImage.size());

  return warpedImage;
}

const string WINDOW_NAME = "Parameter Controls";
// int circularityThreshold = 45; // 圆度阈值
// int medianBlurSize = 3;        // 中值滤波核大小

// bool gp.debug = false;        // gp.debug模式
// bool useTrackbars = gp.debug; // 是否使用滑动条动态调参
// int dilationSize = 7;      // 膨胀核大小
// int erosionSize = 3;       // 腐蚀核大小
// int thresholdValue = 108;  // 二值化阈值

// int rect_area_threshold = 2000; // 矩形面积阈值
// int circle_area_threshold = 50; // 类圆轮廓面积阈值

// int length_width_ratio_threshold = 3; // 长宽比阈值

// int minContourArea = 200; // 最小轮廓面积

void createTrackbars(GlobalParam &gp) {
  namedWindow(WINDOW_NAME, WINDOW_NORMAL);
  moveWindow(WINDOW_NAME, 0, 0);
  createTrackbar("Circularity", WINDOW_NAME, &gp.circularityThreshold, 100,
                 nullptr);
  createTrackbar("Dilation Size", WINDOW_NAME, &gp.dilationSize, 21, nullptr);
  createTrackbar("Erosion Size", WINDOW_NAME, &gp.erosionSize, 21, nullptr);
  createTrackbar("Blur Size", WINDOW_NAME, &gp.medianBlurSize, 21, nullptr);

  createTrackbar("Rect Area Threshold", WINDOW_NAME, &gp.rect_area_threshold,
                 1000, nullptr);
  createTrackbar("Min Contour Area", WINDOW_NAME, &gp.minContourArea, 1000,
                 nullptr);
  createTrackbar("Circle Area Threshold", WINDOW_NAME,
                 &gp.circle_area_threshold, 1000, nullptr);
  createTrackbar("Length Width Ratio Threshold", WINDOW_NAME,
                 &gp.length_width_ratio_threshold, 10, nullptr);
  createTrackbar("Threshold", WINDOW_NAME, &gp.thresholdValue, 255, nullptr);
  namedWindow("Simulation Controls", cv::WINDOW_NORMAL);
  createTrackbar("Fake Pitch", "Simulation Controls", &gp.fake_pitch, 360.0, nullptr);
  createTrackbar("Fake Yaw", "Simulation Controls", &gp.fake_yaw, 360.0, nullptr);
  createTrackbar("Fake Status", "Simulation Controls", &gp.fake_status, 10.0, nullptr);
}

/**
 * @brief 图像预处理函数
 * @param inputImage 输入图像
 * @param gp.debug 是否显示中间步骤
 * @return 预处理后的掩码图像
 */
cv::Mat preprocess(const cv::Mat &inputImage, GlobalParam &gp, int is_blue, Translator &translator) {
  if (gp.debug) {
    static bool trackbarsInitialized = false;
    if (!trackbarsInitialized) {
      createTrackbars(gp);
      trackbarsInitialized = true;
    }
  }

  // --- Step 1: Channel Subtraction ---
  std::vector<cv::Mat> channels;
  cv::split(inputImage, channels);
  cv::Mat blue_ch = channels[0];
  cv::Mat red_ch = channels[2];

  Mat subtracted_img;
  if (!is_blue) { // Target is RED
    subtract(red_ch, blue_ch, subtracted_img);
  } else { // Target is BLUE
    subtract(blue_ch, red_ch, subtracted_img);
  }

  // === 在这里添加可视化 1 ===
// if (gp.debug) {
//    cv::imshow("1_Channel_Subtraction", subtracted_img);
//  }
  // ==========================

  // --- Step 2: Thresholding ---
  cv::Mat threshold_mask;
  if (!is_blue) {
    if (!translator.message.is_far) {
      threshold(subtracted_img, threshold_mask, gp.thresholdValue, 255, THRESH_BINARY);
    } else {
      threshold(subtracted_img, threshold_mask, gp.thresholdValue_1, 255, THRESH_BINARY);
    }
  } else {
    if (!translator.message.is_far) {
      threshold(subtracted_img, threshold_mask, gp.thresholdValueBlue, 255, THRESH_BINARY);
    } else {
      threshold(subtracted_img, threshold_mask, gp.thresholdValueBlue_1, 255, THRESH_BINARY);
    }
  }

  // === 在这里添加可视化 2 ===
//  if (gp.debug) {
//    cv::imshow("2_After_Threshold", threshold_mask);
//  }
  // ==========================

  // --- Step 3: Median Blur ---
  cv::Mat blurred_mask = threshold_mask; // Start with the thresholded mask
  int kernelSize;
  if (!translator.message.is_far) {
    kernelSize = gp.medianBlurSize;
  } else {
    kernelSize = gp.medianBlurSize_1;
  }
  if (kernelSize % 2 == 0 && kernelSize > 0) {
    kernelSize++;
  }
  if (kernelSize > 0) {
    cv::medianBlur(threshold_mask, blurred_mask, kernelSize);
  }
  
  // === 在这里添加可视化 3 ===
  // (Note: This step is often subtle, you might not see much change)
//  if (gp.debug) {
//      cv::imshow("3_After_Blur", blurred_mask);
//  }
  // ==========================


  // --- Step 4: Dilation and Erosion ---
  cv::Mat final_mask = blurred_mask;
  Mat kernel_dilate, kernel_erode;

  if (!translator.message.is_far) {
    if (gp.dilationSize > 0) {
        kernel_dilate = getStructuringElement(MORPH_RECT, Size(gp.dilationSize, gp.dilationSize));
        dilate(blurred_mask, final_mask, kernel_dilate);
    }
    if (gp.erosionSize > 0) {
        kernel_erode = getStructuringElement(MORPH_RECT, Size(gp.erosionSize, gp.erosionSize));
        erode(final_mask, final_mask, kernel_erode); // Erode the result of dilation
    }
  } else {
    if (gp.dilationSize_1 > 0) {
        kernel_dilate = getStructuringElement(MORPH_RECT, Size(gp.dilationSize_1, gp.dilationSize_1));
        dilate(blurred_mask, final_mask, kernel_dilate);
    }
    if (gp.erosionSize_1 > 0) {
        kernel_erode = getStructuringElement(MORPH_RECT, Size(gp.erosionSize_1, gp.erosionSize_1));
        erode(final_mask, final_mask, kernel_erode); // Erode the result of dilation
    }
  }
  
  return final_mask; // This is the final result, which already has its own imshow window.
}

/**
 * @brief (已修正) 识别并分类所有可能是扇叶或R标的圆形轮廓
 * @param contours 图像中的所有轮廓
 * @param hierarchy 轮廓的层级结构
 * @param gp 全局参数
 * @return std::vector<DetectedCircle> 包含所有被分类的圆形候选者的列表
 */
std::vector<DetectedCircle> identify_all_circles(
    const std::vector<std::vector<cv::Point>>& contours,
    const std::vector<cv::Vec4i>& hierarchy,
    GlobalParam& gp
) {
    std::vector<DetectedCircle> all_circle_candidates;

    for (int i = 0; i < contours.size(); ++i) {
        const auto& contour = contours[i];
        double area = cv::contourArea(contour);

        // 过滤掉面积过小的轮廓，使用一个通用的最小轮廓面积
        if (area < gp.minContourArea) {
            continue;
        }
        
        // 计算圆度
        double perimeter = cv::arcLength(contour, true);
        if (perimeter == 0) continue;
        double circularity = 4 * CV_PI * area / (perimeter * perimeter);

        if (circularity > (gp.circularityThreshold / 100.0)) {
            
            // 计算子轮廓数量
            int childCount = 0;
            if (hierarchy[i][2] != -1) { // hierarchy[i][2] 是第一个子轮廓的索引
                int current_child_idx = hierarchy[i][2];
                while (current_child_idx != -1) {
                    childCount++;
                    current_child_idx = hierarchy[current_child_idx][0]; // 移动到下一个同级轮廓
                }
            }
            
            // 构建 DetectedCircle 对象
            DetectedCircle circle;
            circle.contour_idx = i;
            cv::Moments m = cv::moments(contour);
            if (m.m00 == 0) continue;
            circle.center = cv::Point(static_cast<int>(m.m10 / m.m00), static_cast<int>(m.m01 / m.m00));
            circle.area = area;
            circle.child_count = childCount;
            circle.is_fan_candidate = false;
            circle.is_r_logo_candidate = false;

            // ====================================================================
            // ===                      核心逻辑修改在这里                      ===
            // ====================================================================

            // 1. 扇叶候选: 只要它有多个子轮廓(>=2)，并且面积在扇叶的范围内，
            //    它就是一个扇叶候选。我们不再关心它自己是不是顶级轮廓。
            if (childCount >= 6 && 
                area >= gp.target_circle_area_min && area <= gp.target_circle_area_max) {
                circle.is_fan_candidate = true;
            }

            // 2. R标候选: 仍然保持严格的条件，必须是顶级轮廓(无父)，
            //    且无子，且面积在R标的范围内。
            bool is_top_level_contour = (hierarchy[i][3] == -1);
            if (is_top_level_contour && childCount == 0 && 
                area >= gp.R_area_min && area <= gp.R_area_max) {
                circle.is_r_logo_candidate = true;
            }

            // ====================================================================

            // 如果被分类为任何一种，就加入最终列表
            if (circle.is_fan_candidate || circle.is_r_logo_candidate) {
                all_circle_candidates.push_back(circle);
            }
        }
    }

    return all_circle_candidates;
}

/**
 * @brief (已修改) 仅识别所有可能是流水灯条的矩形轮廓
 * @param contours 输入的轮廓
 * @param hierarchy 轮廓的层级结构
 * @param is_potential_rect_contour_flags 输出参数，标记哪些轮廓是潜在的矩形
 * @param gp 全局参数
 */
void identify_potential_rects(
    const std::vector<std::vector<cv::Point>>& contours,
    const std::vector<cv::Vec4i>& hierarchy,
    std::vector<bool>& is_potential_rect_contour_flags,
    GlobalParam& gp
) {
    is_potential_rect_contour_flags.assign(contours.size(), false); // 初始化

    for (int i = 0; i < contours.size(); ++i) {
        const auto& contour = contours[i];
        double area = cv::contourArea(contour);

        if (area < gp.minContourArea) {
            continue;
        }

        // 潜在矩形检测 (流水灯条)
        if (hierarchy[i][3] == -1) { // 必须是顶级轮廓
            if (area > gp.rect_area_threshold) {
                cv::RotatedRect rect = cv::minAreaRect(contour);
                float width = rect.size.width;
                float height = rect.size.height;

                if (height == 0 || width == 0) continue;

                float aspectRatio = (width > height) ? (width / height) : (height / width);

                if (aspectRatio > gp.length_width_ratio_threshold) {
                    is_potential_rect_contour_flags[i] = true;
                }
            }
        }
    }
}
/**
 * @brief (已重构) 对初步识别的矩形进行筛选，找出所有符合条件的灯条
 * @return std::vector<DetectedRectangle> 所有通过ROI验证的灯条列表
 */
std::vector<DetectedRectangle> refine_rectangles_roi(
    const std::vector<std::vector<cv::Point>> &all_contours,
    const std::vector<bool> &is_potential_rect_contour_flags,
    cv::Mat &processed_image, // 注意：必须是原始BGR图像的副本
    // --- 以下参数在新逻辑中不再使用，但为保持接口兼容性而保留 ---
    const std::vector<cv::Point> &initial_circle_centers,
    const std::vector<double> &initial_circle_areas,
    const std::vector<int> &initial_circle_child_counts,
    std::vector<bool> &final_selected_rect_flags, // 此参数已无作用
    // ---
    bool debug_flag,
    GlobalParam &gp) {

    // --- 1. 初始化返回列表 ---
    std::vector<DetectedRectangle> final_detected_rects;

    // --- 2. 收集所有初步候选矩形 ---
    std::vector<int> candidate_indices;
    for (int i = 0; i < all_contours.size(); ++i) {
        if (is_potential_rect_contour_flags[i]) {
            candidate_indices.push_back(i);
        }
    }

    if (candidate_indices.empty()) {
        if (debug_flag) {
            std::cout << "没有初步候选矩形。" << std::endl;
        }
        return final_detected_rects; // 返回空的列表
    }

    // --- 3. 遍历所有候选，进行ROI验证 ---
    for (int original_contour_idx : candidate_indices) {
        cv::RotatedRect rot_rect = cv::minAreaRect(all_contours[original_contour_idx]);
        cv::Rect bounding_rect = rot_rect.boundingRect();

        // 确保ROI在图像边界内
        bounding_rect.x = std::max(0, bounding_rect.x);
        bounding_rect.y = std::max(0, bounding_rect.y);
        bounding_rect.width = std::min(bounding_rect.width, processed_image.cols - bounding_rect.x);
        bounding_rect.height = std::min(bounding_rect.height, processed_image.rows - bounding_rect.y);

        if (bounding_rect.width <= 0 || bounding_rect.height <= 0) {
            if (debug_flag) {
                std::cout << "候选矩形 " << original_contour_idx << " 的ROI无效。" << std::endl;
            }
            continue; // 跳过这个无效的候选
        }

        cv::Mat roi = processed_image(bounding_rect).clone();
        cv::Mat roi_gray, roi_binary;

        if (roi.channels() == 3) {
            cv::cvtColor(roi, roi_gray, cv::COLOR_BGR2GRAY);
        } else { // 如果输入的已经是灰度图或二值图
            roi_gray = roi;
        }
        
        cv::threshold(roi_gray, roi_binary, gp.thresholdValue_for_roi, 255, cv::THRESH_BINARY);
        
        std::vector<std::vector<cv::Point>> roi_contours;
        cv::findContours(roi_binary, roi_contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        // 检查ROI特征
        if (roi_contours.size() > 3) {
            if (debug_flag) {
                std::cout << "候选矩形 " << original_contour_idx << " 通过ROI验证 (内部轮廓数: " << roi_contours.size() << ")。" << std::endl;
            }
            
            // --- 4. 如果验证通过，则创建 DetectedRectangle 对象并添加到返回列表中 ---
            DetectedRectangle rect_info;
            rect_info.contour_idx = original_contour_idx;
            rect_info.rect = rot_rect;
            
            cv::Moments m = cv::moments(all_contours[original_contour_idx]);
            if (m.m00 != 0) {
                rect_info.center = cv::Point2f(static_cast<float>(m.m10 / m.m00), static_cast<float>(m.m01 / m.m00));
            }

            final_detected_rects.push_back(rect_info);
        } else {
            if (debug_flag) {
                std::cout << "候选矩形 " << original_contour_idx << " 未通过ROI特征检查 (ROI内轮廓数: " << roi_contours.size() << ")。" << std::endl;
            }
        }
    }

    // --- 5. 在函数末尾打印最终结果并返回 ---
    if (debug_flag) {
        std::cout << "[DEBUG] 即将从 refine_rectangles_roi 返回 " << final_detected_rects.size() << " 个矩形。" << std::endl;
    }

    return final_detected_rects;
}

/**
 * @brief 根据矩形中心筛选最终的两个圆形轮廓 (目标扇叶和R标)
 * @param current_keypoints
 * KeyPoints结构，包含初步识别的圆，此函数会就地修改它以包含最终选择的圆
 * @param detected_rect_centers 检测到的矩形中心点列表 (通常只使用第一个)
 * @param circle_child_counts 每个初步识别圆的子轮廓数量
 * @param debug_flag 是否开启调试模式
 */
std::vector<DetectedCircle> select_all_circles(
    const KeyPoints &initial_potential_circles,
    const std::vector<int> &initial_circle_child_counts,
    GlobalParam &gp
) {
  std::vector<DetectedCircle> all_valid_circles;
    for (size_t i = 0; i < initial_potential_circles.circleContours.size(); ++i) {
        DetectedCircle circle;
        circle.contour_idx = -1; // 注意：需要一种方法回溯到原始索引，如果需要的话
        circle.center = initial_potential_circles.circlePoints[i];
        circle.area = initial_potential_circles.circleAreas[i];
        circle.child_count = initial_circle_child_counts[i];
        circle.is_fan_candidate = false;
        circle.is_r_logo_candidate = false;

        // 判断是否为扇叶候选 (大圆，有多个子轮廓)
        if (circle.child_count >= 10 && circle.area > gp.target_circle_area_min && circle.area < gp.target_circle_area_max) {
            circle.is_fan_candidate = true;
        }

        // 判断是否为R标候选 (小圆，无父无子)
        // 注意：这里的判断需要 initial_potential_circles 包含原始 contour 和 hierarchy 信息，或者在 identify_initial_shapes 里就做好
        if (circle.child_count == 0 && circle.area > gp.R_area_min && circle.area < gp.R_area_max) {
             // 还需要检查无父轮廓的条件
            circle.is_r_logo_candidate = true;
        }
        
        if (circle.is_fan_candidate || circle.is_r_logo_candidate) {
            all_valid_circles.push_back(circle);
        }
    }
    return all_valid_circles;
}

/**
 * @brief 可视化检测到的关键点 (如果debug模式开启)
 * @param image_to_draw_on 用于绘制的图像 (通常是 processedImage 的克隆)
 * @param final_result 包含最终检测结果的KeyPoints结构
 * @param all_contours 原始图像中的所有轮廓
 * @param initial_is_rect_flags 标记了哪些轮廓是初步认定的矩形
 * @param final_selected_rect_flags 标记了最终选定的矩形轮廓
 */
void visualize_keypoints(
    cv::Mat &image_to_draw_on, // Mat会被修改用于显示
    const KeyPoints &final_result,
    const std::vector<std::vector<cv::Point>> &all_contours,
    const std::vector<bool> &initial_is_rect_flags, // 初始矩形候选标记
    const std::vector<bool> &final_selected_rect_flags) { // 最终选择的矩形标记

  // 绘制最终选择的圆形轮廓 (绿色)
  for (size_t i = 0; i < final_result.circleContours.size(); ++i) {
    cv::drawContours(image_to_draw_on, final_result.circleContours,
                     static_cast<int>(i), cv::Scalar(0, 255, 0), 2); // 绿色轮廓
    cv::circle(image_to_draw_on, final_result.circlePoints[i], 5,
               cv::Scalar(0, 255, 0), -1); // 绿色中心点
    std::string circle_info =
        "Area: " +
        std::to_string(static_cast<int>(final_result.circleAreas[i])) +
        " Circ: " +
        (final_result.circularities[i] >= 0 &&
                 final_result.circularities[i] <= 1
             ? std::to_string(final_result.circularities[i]).substr(0, 4)
             : "N/A");
    cv::putText(image_to_draw_on, circle_info,
                final_result.circlePoints[i] + cv::Point(10, 10),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 0),
                1); // 青色文字
  }

  // 绘制所有初步识别但未被最终选中的矩形轮廓 (白色)
  for (size_t i = 0; i < all_contours.size(); ++i) {
    if (initial_is_rect_flags[i] && !final_selected_rect_flags[i]) {
      cv::drawContours(image_to_draw_on, all_contours, static_cast<int>(i),
                       cv::Scalar(255, 255, 255), 1); // 白色细线条
    }
  }

  // 绘制最终筛选完毕的目标矩形轮廓 (紫色)
  for (size_t i = 0; i < all_contours.size(); ++i) {
    if (final_selected_rect_flags[i]) { // 如果这个轮廓是最终选定的矩形之一
      cv::drawContours(image_to_draw_on, all_contours, static_cast<int>(i),
                       cv::Scalar(255, 0, 255), 3); // 紫色粗线条

      cv::Point2f rect_center_to_draw;
      bool center_for_drawing_found = false;

      // 尝试从 final_result.rectCenters 中匹配正确的中心点
      // 如果只有一个最终矩形，其中心点应该就是 final_result.rectCenters[0]
      if (final_result.rectCenters.size() == 1) {
        rect_center_to_draw = final_result.rectCenters[0];
        center_for_drawing_found = true;
      } else { // 如果有多个最终矩形，需要匹配
        cv::Moments m_contour = cv::moments(all_contours[i]);
        if (m_contour.m00 != 0) {
          cv::Point2f current_contour_center(
              static_cast<float>(m_contour.m10 / m_contour.m00),
              static_cast<float>(m_contour.m01 / m_contour.m00));
          for (const auto &stored_center : final_result.rectCenters) {
            if (cv::norm(stored_center - current_contour_center) <
                1.0) { // 允许微小误差
              rect_center_to_draw = stored_center;
              center_for_drawing_found = true;
              break;
            }
          }
        }
      }
      // 如果通过上述方法未找到，作为后备，直接计算当前轮廓的中心
      if (!center_for_drawing_found) {
        cv::Moments m = cv::moments(all_contours[i]);
        if (m.m00 != 0) {
          rect_center_to_draw = cv::Point2f(static_cast<float>(m.m10 / m.m00),
                                            static_cast<float>(m.m01 / m.m00));
          center_for_drawing_found = true; // 至少我们有一个中心点来绘制
        }
      }

      if (center_for_drawing_found) {
        cv::circle(image_to_draw_on,
                   cv::Point(static_cast<int>(rect_center_to_draw.x),
                             static_cast<int>(rect_center_to_draw.y)),
                   8, cv::Scalar(255, 0, 255), -1); // 紫色中心点

        cv::RotatedRect rot_rect = cv::minAreaRect(all_contours[i]);
        float width = rot_rect.size.width;
        float height = rot_rect.size.height;
        float aspectRatio =
            (width == 0 || height == 0)
                ? 0.0f
                : ((width > height) ? (width / height) : (height / width));
        float area = static_cast<float>(cv::contourArea(all_contours[i]));

        cv::putText(image_to_draw_on, "Final Rect",
                    cv::Point(static_cast<int>(rect_center_to_draw.x),
                              static_cast<int>(rect_center_to_draw.y)) +
                        cv::Point(10, -10),
                    cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 0, 255), 2);
        std::string rect_info =
            "Ratio: " +
            (aspectRatio > 0 ? std::to_string(aspectRatio).substr(0, 4)
                             : "N/A") +
            " Area: " + std::to_string(static_cast<int>(area));
        cv::putText(image_to_draw_on, rect_info,
                    cv::Point(static_cast<int>(rect_center_to_draw.x),
                              static_cast<int>(rect_center_to_draw.y)) +
                        cv::Point(10, 20),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 0, 255), 1);
      }
    }
  }

  cv::Mat resized_image;
  // 调整图像大小以便显示 (可选)
  // cv::Size newSize(image_to_draw_on.cols / 2, image_to_draw_on.rows / 2); //
  // 如缩小一半
  cv::Size newSize(840, 620); // 与原代码一致的示例尺寸
  if (image_to_draw_on.cols > 0 && image_to_draw_on.rows > 0) { // 确保图像有效
    cv::resize(image_to_draw_on, resized_image, newSize, 0, 0,
               cv::INTER_LINEAR);
    cv::imshow("Detected Key Points (Refactored)", resized_image);
  } else {
    cv::imshow("Detected Key Points (Refactored) - Original Size",
               image_to_draw_on);
  }
  cv::waitKey(1); // 保持窗口更新
}
/**
 * @brief 识别所有潜在矩形，并识别和分类所有圆形候选
 * @param contours 输入的轮廓
 * @param hierarchy 轮廓的层级结构
 * @param is_potential_rect_contour_flags [输出] 标记哪些轮廓是潜在的矩形
 * @param gp 全局参数
 * @return std::vector<DetectedCircle> 包含所有被分类的圆形候选者的列表
 */
std::vector<DetectedCircle> identify_initial_shapes(
    const std::vector<std::vector<cv::Point>> &contours,
    const std::vector<cv::Vec4i> &hierarchy,
    std::vector<bool> &is_potential_rect_contour_flags,
    GlobalParam &gp) 
{
    std::vector<DetectedCircle> all_circle_candidates;
    is_potential_rect_contour_flags.assign(contours.size(), false);

    for (int i = 0; i < contours.size(); ++i) {
        const auto &contour = contours[i];
        double area = cv::contourArea(contour);

        if (area < gp.minContourArea) {
            continue;
        }

        // --- 1. 潜在矩形检测 ---
        if (hierarchy[i][3] == -1) { // 是顶级轮廓
            if (area > gp.rect_area_threshold) {
                cv::RotatedRect rect = cv::minAreaRect(contour);
                float width = rect.size.width;
                float height = rect.size.height;
                if (height != 0 && width != 0) {
                    float aspectRatio = (width > height) ? (width / height) : (height / width);
                    if (aspectRatio > gp.length_width_ratio_threshold) {
                        is_potential_rect_contour_flags[i] = true;
                    }
                }
            }
        }
        
        // 如果是矩形，就不再是圆形
        if (is_potential_rect_contour_flags[i]) {
            continue;
        }

        // --- 2. 潜在圆形检测与分类 ---
        double perimeter = cv::arcLength(contour, true);
        if (perimeter == 0) continue;

        double circularity = 4 * CV_PI * area / (perimeter * perimeter);
        if (circularity > (gp.circularityThreshold / 100.0)) {
            
            DetectedCircle circle;
            
            int childCount = 0;
            if (hierarchy[i][2] != -1) {
                int current_child_idx = hierarchy[i][2];
                while (current_child_idx != -1) {
                    childCount++;
                    current_child_idx = hierarchy[current_child_idx][0];
                }
            }
            circle.child_count = childCount;

            cv::Moments m = cv::moments(contour);
            if (m.m00 == 0) continue;
            circle.center = cv::Point(static_cast<int>(m.m10 / m.m00), static_cast<int>(m.m01 / m.m00));
            circle.area = area;
            circle.contour_idx = i;
            circle.is_fan_candidate = false;
            circle.is_r_logo_candidate = false;

            // 分类逻辑
            bool is_fan = (childCount >= 2 && area >= gp.target_circle_area_min && area <= gp.target_circle_area_max);
            bool is_r_logo = (hierarchy[i][3] == -1 && childCount == 0 && area >= gp.R_area_min && area <= gp.R_area_max);

            // 推荐的写法
            if (is_fan || is_r_logo) {
            // 无论它是扇叶还是R标，我们都先把 is_fan/is_r_logo 标志位设好
            if (is_fan) {
                circle.is_fan_candidate = true;
            }
            if (is_r_logo) {
                circle.is_r_logo_candidate = true;
            }

            // --- 关键：在这里统一保存轮廓 ---
            circle.contour = contour;
            
            // 最后再把填充完整的 circle 对象 push_back
            all_circle_candidates.push_back(circle);
          }
        }
    }
    return all_circle_candidates;
}

/**
 * @brief (最终修正版) 检测所有零件。接收 contours 和 hierarchy 作为输入。
 * @param contours 图像中所有轮廓
 * @param hierarchy 轮廓的层级结构
 * @param processedImage 原始 BGR 图像的副本，用于 ROI 分析和可视化
 * @param blade (不再使用)
 * @param gp 全局参数
 * @return KeyPoints 包含所有识别出的零件
 */
KeyPoints detect_key_points(
    const std::vector<std::vector<cv::Point>> &contours,
    const std::vector<cv::Vec4i> &hierarchy,
    cv::Mat &processedImage,
    WMBlade &blade,
    GlobalParam &gp) 
{
    KeyPoints final_result;

    // --- 1. 调用识别函数，获取所有零件 ---
    // (这里的 is_potential_rect_flags 是 identify_initial_shapes 的输出参数)
    std::vector<bool> is_potential_rect_flags;
    final_result.detected_circles = identify_initial_shapes(contours, hierarchy, is_potential_rect_flags, gp);

    // --- 2. 精炼矩形 ---
    std::vector<bool> dummy_flags; // 不再使用的参数
    final_result.detected_rects = refine_rectangles_roi(
        contours,
        is_potential_rect_flags,
        processedImage, // 传入的是原始 BGR 图像
        {}, {}, {}, dummy_flags,
        gp.debug,
        gp
    );
    
    // --- 3. 调试输出与可视化 ---
    if (gp.debug) {
        // a. 打印日志
        int rect_count = 0;
        for(bool flag : is_potential_rect_flags) if(flag) rect_count++;
        std::cout << "初始识别到 " << rect_count << " 个候选矩形。" << std::endl;
        
        int fan_count = 0, r_logo_count = 0;
        for(const auto& circle : final_result.detected_circles) {
            if (circle.is_fan_candidate) fan_count++;
            if (circle.is_r_logo_candidate) r_logo_count++;
        }
        std::cout << "共识别到 " << fan_count << " 个扇叶候选和 " << r_logo_count << " 个R标候选。" << std::endl;
        std::cout << "矩形精炼后，共确定 " << final_result.detected_rects.size() << " 个灯条。" << std::endl;
        
        // b. 创建可视化图像
        cv::Mat visual_image = processedImage.clone();

        // c. 绘制所有最终被分类的零件
        // 绘制所有识别到的灯条 (紫色)
        for (const auto& rect_info : final_result.detected_rects) {
            cv::Point2f vertices[4];
            rect_info.rect.points(vertices);
            for (int i = 0; i < 4; i++) {
                cv::line(visual_image, vertices[i], vertices[(i + 1) % 4], cv::Scalar(255, 0, 255), 2);
            }
        }

        // 绘制所有识别到的圆形
        for (const auto& circle_info : final_result.detected_circles) {
            if (circle_info.is_fan_candidate) {
                // 扇叶候选画绿色，并标注 'F' 和子轮廓数
                cv::drawContours(visual_image, contours, circle_info.contour_idx, cv::Scalar(0, 255, 0), 2);
                cv::putText(visual_image, "F:" + std::to_string(circle_info.child_count), 
                            circle_info.center, cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 2);
            }
            if (circle_info.is_r_logo_candidate) {
                // R标候选画青色，并标注 'R'
                cv::drawContours(visual_image, contours, circle_info.contour_idx, cv::Scalar(255, 255, 0), 2);
                cv::putText(visual_image, "R", circle_info.center, cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 0), 2);
            }
        }
        
        // d. 显示图像
        cv::imshow("1. All Detected Parts", visual_image);
        cv::waitKey(1);
    }

    return final_result;
}

DetectionResult detect(const cv::Mat &inputImage, WMBlade &blade,
                       GlobalParam &gp, int is_blue, Translator &translator) {

    DetectionResult result;
    auto start_time = high_resolution_clock::now();

    Mat final_mask = preprocess(inputImage, gp, is_blue, translator);
    if (gp.debug) {
        imshow("final_mask", final_mask);
    }

    // --- 关键修改：将 findContours 移回到 detect 函数中 ---
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    findContours(final_mask, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);
    // ----------------------------------------------------

    Mat processedImage = inputImage.clone();

    // --- 关键修改：将 contours 和 hierarchy 作为参数传递 ---
    KeyPoints all_detected_parts =
        detect_key_points(contours, hierarchy, processedImage, blade, gp);

    if (!all_detected_parts.isValid()) {
        if (gp.debug) {
            std::cout << "检测失败: 未找到足够的零件来构成一个目标。" << std::endl;
        }
        result.processedImage = processedImage; 
        result.contours = contours; // 即使失败，也可能需要轮廓信息进行调试
        auto end_time = high_resolution_clock::now();
        result.processingTime =
            duration_cast<milliseconds>(end_time - start_time).count();
        return result; 
    }

    result.all_key_points = all_detected_parts;
    result.processedImage = processedImage; 
    result.contours = contours; // <-- 现在 contours 是有效的，可以赋值

    auto end_time = high_resolution_clock::now();
    result.processingTime =
        duration_cast<milliseconds>(end_time - start_time).count();

    return result;
}



// 通过方程求解交点的方法
vector<Point> findIntersectionsByEquation(const Point &center1,
                                          const Point &center2, double radius,
                                          const RotatedRect &ellipse, Mat &pic,
                                          GlobalParam &gp, WMBlade &blade) {
  vector<Point> intersections;

      if (gp.debug) {
        std::cout << "\n--- Intersections Debug ---" << std::endl;
        std::cout << "Ellipse Center: " << ellipse.center << ", Size: " << ellipse.size << ", Angle: " << ellipse.angle << std::endl;
        std::cout << "Line defined by: " << center1 << " and " << center2 << std::endl;
    }

  // 获取椭圆参数
  Point2f ellipse_center = ellipse.center;
  Size2f size = ellipse.size;
  float angle_deg = ellipse.angle;              // 旋转角度（度）
  double angle_rad = angle_deg * CV_PI / 180.0; // 旋转角度（弧度）

  // 将椭圆缩放0.9倍 (考虑灯条粗细需要缩放让该椭圆方程能够拟合到灯条中心)
  double scale = 0.9;
  // 半长轴和半短轴缩放
  double a = (size.width / 2.0) * scale;
  double b = (size.height / 2.0) * scale;

  // 计算第一条直线的系数 A x + B y + C = 0
  double A = center2.y - center1.y;
  double B = center1.x - center2.x;
  double C = center2.x * center1.y - center1.x * center2.y;

  // 将直线方程旋转到椭圆的坐标系
  double cos_theta = cos(angle_rad);
  double sin_theta = sin(angle_rad);

  double A_rot = A * cos_theta + B * sin_theta;
  double B_rot = -A * sin_theta + B * cos_theta;
  double C_rot = C + A * ellipse_center.x + B * ellipse_center.y;

  // 在图片右侧显示方程
  if (gp.debug) {
    // 计算显示位置
    int text_x = pic.cols - 500; // 距离右边界600像素
    int text_y = pic.rows / 2;   // 垂直居中
    int line_height = 40;        // 行间距

    // 显示坐标系信息
    string coord_info = "Coordinate System:";
    string coord_info1 = "x: major axis, rotated " +
                         to_string(angle_deg).substr(0, 4) + " degrees";
    string coord_info2 = "y: minor axis, perpendicular to x";
    putText(pic, coord_info, Point(text_x, text_y - 2 * line_height),
            FONT_HERSHEY_SIMPLEX, 0.8, Scalar(255, 255, 255), 2);
    putText(pic, coord_info1, Point(text_x, text_y - line_height),
            FONT_HERSHEY_SIMPLEX, 0.8, Scalar(255, 255, 255), 2);
    putText(pic, coord_info2, Point(text_x, text_y), FONT_HERSHEY_SIMPLEX, 0.8,
            Scalar(255, 255, 255), 2);

    // 显示椭圆方程
    string ellipse_eq = "Ellipse Equation:";
    string ellipse_eq1 = "x^2/" + to_string(a * a).substr(0, 6) + " + y^2/" +
                         to_string(b * b).substr(0, 6) + " = 1";
    putText(pic, ellipse_eq, Point(text_x, text_y + line_height),
            FONT_HERSHEY_SIMPLEX, 0.8, Scalar(255, 255, 255), 2);
    putText(pic, ellipse_eq1, Point(text_x, text_y + 2 * line_height),
            FONT_HERSHEY_SIMPLEX, 0.8, Scalar(255, 255, 255), 2);

    // 显示直径方程
    string diameter_eq = "Diameter Equation:";
    string diameter_eq1 = to_string(A).substr(0, 6) + "x + " +
                          to_string(B).substr(0, 6) + "y + " +
                          to_string(C).substr(0, 6) + " = 0";
    putText(pic, diameter_eq, Point(text_x, text_y + 3 * line_height),
            FONT_HERSHEY_SIMPLEX, 0.8, Scalar(255, 255, 255), 2);
    putText(pic, diameter_eq1, Point(text_x, text_y + 4 * line_height),
            FONT_HERSHEY_SIMPLEX, 0.8, Scalar(255, 255, 255), 2);

    // 显示共轭直径方程
    string conjugate_eq = "Conjugate Diameter:";
    string conjugate_eq1;
    if (A != 0) {
      double new_slope = (b * b * B) / (a * a * A);
      conjugate_eq1 =
          to_string(new_slope).substr(0, 6) + "x - y + " +
          to_string(center1.y - new_slope * center1.x).substr(0, 6) + " = 0";
    } else {
      conjugate_eq1 = "x - " + to_string(center1.x).substr(0, 6) + " = 0";
    }
    putText(pic, conjugate_eq, Point(text_x, text_y + 5 * line_height),
            FONT_HERSHEY_SIMPLEX, 0.8, Scalar(255, 255, 255), 2);
    putText(pic, conjugate_eq1, Point(text_x, text_y + 6 * line_height),
            FONT_HERSHEY_SIMPLEX, 0.8, Scalar(255, 255, 255), 2);
  }

  // 避免除零的情况
  if (fabs(B_rot) < 1e-8) {
    cout << "直线几乎垂直" << endl;
  }

  // 计算二次方程系数
  double M = (1.0 / (a * a)) + (A_rot * A_rot) / (B_rot * B_rot * b * b);
  double N = (2.0 * A_rot * C_rot) / (B_rot * B_rot * b * b);
  double P = (C_rot * C_rot) / (B_rot * B_rot * b * b) - 1.0;

  // delta
  double discriminant = N * N - 4.0 * M * P;

  if (discriminant >= 0) {
    double sqrt_discriminant = sqrt(discriminant);
    double x1_rot = (-N + sqrt_discriminant) / (2.0 * M);
    double x2_rot = (-N - sqrt_discriminant) / (2.0 * M);

    double y1_rot = (-A_rot * x1_rot - C_rot) / B_rot;
    double y2_rot = (-A_rot * x2_rot - C_rot) / B_rot;

    double x1 = x1_rot * cos_theta - y1_rot * sin_theta + ellipse_center.x;
    double y1 = x1_rot * sin_theta + y1_rot * cos_theta + ellipse_center.y;

    double x2 = x2_rot * cos_theta - y2_rot * sin_theta + ellipse_center.x;
    double y2 = x2_rot * sin_theta + y2_rot * cos_theta + ellipse_center.y;

    Point pt1(cvRound(x1), cvRound(y1));
    Point pt2(cvRound(x2), cvRound(y2));

    if (gp.debug) {
      // 绘制矩形框
      Point2f rect_points[4];
      ellipse.points(rect_points);
      //  for (int i = 0; i < 4; i++) {
      //    line(pic, rect_points[i], rect_points[(i+1)%4], Scalar(0, 255, 255),
      //    2);
      //  }

      // 绘制扇叶椭圆
      cv::ellipse(pic, ellipse.center, Size(a / scale, b / scale), angle_deg, 0,
                  360, Scalar(255, 0, 255), 2);

      // 绘制两条直径
      line(pic, center1, center2, Scalar(255, 255, 0), 2);
      circle(pic, center1, 5, Scalar(0, 0, 255), -1);
      circle(pic, center2, 5, Scalar(0, 255, 0), -1);

      // 绘制交点
      circle(pic, pt1, 5, Scalar(255, 0, 0), -1);
      circle(pic, pt2, 5, Scalar(255, 0, 0), -1);
    }

    if (computeDistance(pt1, center2) > computeDistance(pt2, center2)) {
      intersections.emplace_back(pt1);
      blade.apex.push_back(pt1);
    } else {
      intersections.emplace_back(pt2);
      blade.apex.push_back(pt2);
    }
    circle(pic, intersections[0], 3, Scalar(0, 255, 0), -1);
  }

  double A2_rot_conj,
      B2_rot_conj; // 共轭直径在旋转坐标系下的系数 A_conj*x' + B_conj*y' = 0

  const double epsilon = 1e-9; // 用于比较浮点数是否接近0 (原用1e-8，可按需调整)

  bool conjugate_calculation_possible = true;

  if (fabs(A_rot) < epsilon && fabs(B_rot) < epsilon) {
    // A_rot 和 B_rot 都接近0 (比如 center1 和 center2 是同一点)
    // 无法定义第一条直径，因此无法计算共轭直径
    if (gp.debug) {
      cout << "警告: 计算共轭直径时，A_rot 和 B_rot "
              "同时接近0。跳过共轭直径计算。"
           << endl;
    }
    conjugate_calculation_possible = false;
  } else if (fabs(A_rot) < epsilon) {
    // 第一条直径在旋转坐标系下是水平的 (y' approx 0, 因为 B_rot*y' approx 0,
    // B_rot不为0) 其共轭直径在旋转坐标系下是垂直的 (x' = 0)
    A2_rot_conj = 1.0;
    B2_rot_conj = 0.0;
  } else if (fabs(B_rot) < epsilon) {
    // 第一条直径在旋转坐标系下是垂直的 (x' approx 0, 因为 A_rot*x' approx 0,
    // A_rot不为0) 其共轭直径在旋转坐标系下是水平的 (y' = 0)
    A2_rot_conj = 0.0;
    B2_rot_conj = 1.0; //  方向不影响直线本身, 可用 -1.0
  } else {
    // 一般情况: 第一条直径斜率 m'_1 = -A_rot / B_rot
    // 共轭直径斜率 m'_2 = -(b^2/a^2) / m'_1 = (b^2 * B_rot) / (a^2 * A_rot)
    // 方程为: y' = m'_2 * x'  =>  (b^2 * B_rot) * x' - (a^2 * A_rot) * y' = 0
    // 注意避免a或b为0的情况，尽管对于有效椭圆它们应为正
    if (fabs(a) < epsilon || fabs(b) < epsilon) {
      if (gp.debug) {
        cout << "警告: 椭圆半轴长a或b过小，无法安全计算共轭直径。" << endl;
      }
      conjugate_calculation_possible = false;
    } else {
      A2_rot_conj = b * b * B_rot;
      B2_rot_conj = -a * a * A_rot;
    }
  }

  if (conjugate_calculation_possible) {
    Point pt3, pt4; // 共轭直径的两个交点
    bool found_conjugate_intersections = false;

    if (fabs(B2_rot_conj) < epsilon) {
      // 共轭直径在旋转坐标系下是垂直的 (x' = 0), A2_rot_conj 不应为0
      // (除非A_rot,B_rot都为0)
      if (fabs(A2_rot_conj) > epsilon) { // 确保是 x'=0 而不是 0=0
        double x_rot_c = 0.0;
        // 代入椭圆方程: y'^2/b^2 = 1 => y' = +/- b
        if (b > epsilon) { // b 必须为正
          double y1_rot_c = b;
          double y2_rot_c = -b;

          pt3 = Point(cvRound(x_rot_c * cos_theta - y1_rot_c * sin_theta +
                              ellipse_center.x),
                      cvRound(x_rot_c * sin_theta + y1_rot_c * cos_theta +
                              ellipse_center.y));
          pt4 = Point(cvRound(x_rot_c * cos_theta - y2_rot_c * sin_theta +
                              ellipse_center.x),
                      cvRound(x_rot_c * sin_theta + y2_rot_c * cos_theta +
                              ellipse_center.y));
          found_conjugate_intersections = true;
        }
      }
    } else { // 一般情况，B2_rot_conj 不为0
      // 共轭直径方程 y' = -(A2_rot_conj / B2_rot_conj) * x'
      // 代入椭圆: x'^2/a^2 + (-(A2_rot_conj / B2_rot_conj) * x')^2 / b^2 = 1
      // x'^2 * [1/a^2 + A2_rot_conj^2 / (B2_rot_conj^2 * b^2)] = 1
      // 注意避免a, b, B2_rot_conj为0的情况
      if (fabs(a) < epsilon || fabs(b) < epsilon) {
        if (gp.debug)
          cout << "警告: 椭圆半轴a或b为0，无法计算共轭直径交点。" << endl;
      } else {
        double term_A_sq = A2_rot_conj * A2_rot_conj;
        double term_B_sq_b_sq = B2_rot_conj * B2_rot_conj * b * b;

        if (fabs(term_B_sq_b_sq) < epsilon) { // 防止除以0
          if (gp.debug)
            cout << "警告: 计算共轭直径交点时出现分母为0的情况 "
                    "(term_B_sq_b_sq)。"
                 << endl;
        } else {
          double M2_val = (1.0 / (a * a)) + term_A_sq / term_B_sq_b_sq;
          if (M2_val > epsilon) { // M2_val 必须为正才能开方
            double x_val_sq = 1.0 / M2_val;
            double x1_rot_c = sqrt(x_val_sq);
            double x2_rot_c = -sqrt(x_val_sq);

            double y1_rot_c = (-A2_rot_conj * x1_rot_c) / B2_rot_conj;
            double y2_rot_c = (-A2_rot_conj * x2_rot_c) / B2_rot_conj;

            pt3 = Point(cvRound(x1_rot_c * cos_theta - y1_rot_c * sin_theta +
                                ellipse_center.x),
                        cvRound(x1_rot_c * sin_theta + y1_rot_c * cos_theta +
                                ellipse_center.y));
            pt4 = Point(cvRound(x2_rot_c * cos_theta - y2_rot_c * sin_theta +
                                ellipse_center.x),
                        cvRound(x2_rot_c * sin_theta + y2_rot_c * cos_theta +
                                ellipse_center.y));
            found_conjugate_intersections = true;
          } else {
            if (gp.debug)
              cout << "警告: 计算共轭直径交点时 M2_val 非正。" << endl;
          }
        }
      }
    }

    if (found_conjugate_intersections) {
      if (gp.debug) {
        // 绘制共轭直径的两个交点
        circle(pic, pt3, 3, Scalar(255, 255, 0), -1); // 绿色表示共轭直径交点
        circle(pic, pt4, 3, Scalar(0, 255, 255), -1);
        // 可以选择绘制共轭直径本身 (pt3到pt4的连线，如果它们确实是直径的端点)
        // line(pic, pt3, pt4, Scalar(0, 0, 255), 3); // 示例：深青色线
      }

      // 使用原始代码中的排序逻辑，确保对 intersections 和 blade.apex
      // 的添加顺序一致
      Point O_sort = center2; // 参考点进行排序
      Point OP3_vec = pt3 - O_sort;
      Point OP4_vec = pt4 - O_sort;
      Point OP3N_vec(-OP3_vec.y, OP3_vec.x); // OP3_vec 旋转90度
      double dot_product_sort = OP3N_vec.x * OP4_vec.x + OP3N_vec.y * OP4_vec.y;

      if (dot_product_sort < 0) { // pt3 "在先"
        intersections.emplace_back(pt3);
        blade.apex.push_back(pt3);
        intersections.emplace_back(pt4);
        blade.apex.push_back(pt4);
      } else { // pt4 "在先"
        intersections.emplace_back(pt4);
        blade.apex.push_back(pt4);
        intersections.emplace_back(pt3);
        blade.apex.push_back(pt3);
      }
    }
  }

    if (gp.debug) {
    cv::imshow("椭圆拟合", pic);
  }

  return intersections;
}

// int main() {
//   cout << "传统算法检测" << endl;

//   bool useOffcialWindmill = true;
//   bool perspective = false;
//   bool useVideo = true;

//   if (useVideo) {
//     VideoCapture cap;
//     if (!useOffcialWindmill) {
//       cap = VideoCapture("/Users/clarencestark/RoboMaster/第四次任务/"
//                          "nanodet_rm/camera/build/output.avi");
//     } else {

//       // cap = VideoCapture(
//       // "/Users/clarencestark/RoboMaster/步兵打符-视觉组/"
//       // "传统识别算法/XJTU2025WindMill/imgs_and_videos/output.mp4");
//       // cap = VideoCapture("/Users/clarencestark/RoboMaster/第四次任务/"
//       //                    "nanodet_rm/camera/build/output222.mp4");
//       cap = VideoCapture("/Users/clarencestark/RoboMaster/第四次任务/"
//                          "nanodet_rm/camera/build/2-5-正对大符有R标(红色).avi");
//     }

//     if (!cap.isOpened()) {
//       cout << "无法打开视频文件" << endl;
//       return -1;
//     }

//     Mat frame;
//     bool paused = false;

//     // 添加帧率计算相关变量
//     double fps = 0;
//     auto last_time = high_resolution_clock::now();
//     int frame_count = 0;

//     while (true) {
//       if (!paused) {
//         if (!cap.read(frame)) {
//           cout << "视频结束或帧获取失败" << endl;
//           break;
//         }

//         // 计算帧率
//         frame_count++;
//         auto current_time = high_resolution_clock::now();
//         auto time_diff =
//             duration_cast<milliseconds>(current_time - last_time).count();

//         if (time_diff >= 1000) { // 每秒更新一次帧率
//           fps = frame_count * 1000.0 / time_diff;
//           frame_count = 0;
//           last_time = current_time;
//         }
//       }

//       DetectionResult result;
//       if (perspective) {
//         Mat transformedFrame = applyYawPerspectiveTransform(frame, 0.18);
//         WMBlade temp_blade;
//         result = detect(transformedFrame, temp_blade);
//         imshow("Original Image", frame);
//         imshow("Transformed Image", transformedFrame);
//       } else {
//         WMBlade temp_blade;
//         result = detect(frame, temp_blade);
//       }

//       // 帧率和处理时间
//       string fps_text = "FPS: " + to_string(static_cast<int>(fps));
//       string time_text =
//           "Process Time: " + to_string(result.processingTime) + "ms";
//       putText(result.processedImage, fps_text, Point(10, 30),
//               FONT_HERSHEY_SIMPLEX, 1.0, Scalar(0, 255, 0), 2);
//       putText(result.processedImage, time_text, Point(10, 70),
//               FONT_HERSHEY_SIMPLEX, 1.0, Scalar(0, 255, 0), 2);

//       // 显示结果图像
//       imshow("Processed Image", result.processedImage);

//       // 等待按键
//       char key = (char)waitKey(70);

//       // 按键控制
//       if (key == 'q' || key == 'Q') {
//         break; // 退出
//       } else if (key == ' ') {
//         paused = !paused; // 空格键切换暂停/继续
//       }
//     }

//     cap.release();
//   } else {
//     // 原有的图像处理逻辑
//     Mat frame;
//     if (!useOffcialWindmill) {
//       frame = imread("/Users/clarencestark/RoboMaster/第四次任务/nanodet_rm/"
//                      "camera/build/imgs/image52.jpg");
//     } else {
//       frame = imread("/Users/clarencestark/RoboMaster/步兵打符-视觉组/"
//                      "local_Indentify_Develop/src/test3.jpg");
//     }

//     if (frame.empty()) {
//       cout << "无法获取图像" << endl;
//       return -1;
//     }

//     WMBlade temp_blade;
//     DetectionResult result;
//     if (perspective) {
//       Mat transformedFrame = applyYawPerspectiveTransform(frame, 0.20);

//       // 显示原始图像和变换后的图像
//       imshow("Original Image", frame);
//       imshow("Transformed Image", transformedFrame);

//       // 处理变换后的图像
//       result = detect(transformedFrame, temp_blade);
//     } else {
//       result = detect(frame, temp_blade);
//     }

//     // 显示结果
//     cout << "处理时间: " << result.processingTime << " ms" << endl;
//     cout << "检测到 " << result.circlePoints.size() << " 个圆心" << endl;
//     cout << "检测到 " << result.intersections.size() << " 个交点" << endl;

//     // 显示结果图像
//     imshow("Processed Image", result.processedImage);
//     waitKey(0);
//   }

//   return 0;
// }
