// Copyright (c) 2022 ChenJun
// Licensed under the MIT License.

// OpenCV
#include <opencv2/core.hpp>
#include <opencv2/core/base.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/imgproc.hpp>

// STD
#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>
#include <filesystem>

#include "detector.hpp"
#include "opencv2/core/hal/interface.h"
// #include <light_corner.hpp>
#include "opencv2/highgui.hpp"

Detector::Detector(GlobalParam &gp)
{
    // int binary_thres = binary_threshold;
    // int detect_color = color;
    this->gp = &gp;
    int color = gp.color;
    this->detect_color = color;
    double min_ratio,
        max_ratio,
        max_angle_l,
        min_light_ratio,
        min_small_center_distance,
        max_small_center_distance,
        min_large_center_distance,
        max_large_center_distance,
        max_angle_a,
        num_threshold;
    min_ratio = gp.min_ratio;
    max_ratio = gp.max_ratio;
    max_angle_l = gp.max_angle_l;
    min_light_ratio = gp.min_light_ratio;
    min_small_center_distance = gp.min_small_center_distance;
    max_small_center_distance = gp.max_small_center_distance;
    min_large_center_distance = gp.min_large_center_distance;
    max_large_center_distance = gp.max_large_center_distance;
    max_angle_a = gp.max_angle_a;
    num_threshold = gp.num_threshold;
    this->red_threshold = gp.red_threshold;
    this->blue_threshold = gp.blue_threshold;
    binary_thres = color == RED ? this->red_threshold : this->blue_threshold;
    this->l = {
        .min_ratio = min_ratio,
        .max_ratio = max_ratio,
        .max_angle = max_angle_l};

    this->a = {
        .min_light_ratio = 0.7,
        .min_small_center_distance = min_small_center_distance,
        .max_small_center_distance = max_small_center_distance,
        .min_large_center_distance = min_large_center_distance,
        .max_large_center_distance = max_large_center_distance,
        .max_angle = max_angle_a};

    // Init classifier
    auto model_path = "../model/lenet.onnx";
    auto label_path = "../model/label.txt";
    std::vector<std::string> ignore_classes =
        std::vector<std::string>{"negative"};
    this->classifier =
        std::make_unique<NumberClassifier>(model_path, label_path, num_threshold, ignore_classes);
}
 
std::vector<UnsolvedArmor> Detector::detect(cv::Mat &input, const int color)
{
    this->binary_thres = color == RED ? this->red_threshold : this->blue_threshold;
    this->detect_color = color;
    cv::Mat gray_img;
    cv::cvtColor(input, gray_img, cv::COLOR_RGB2GRAY);
    cv::Mat binary_img;
    cv::threshold(gray_img, binary_img, binary_thres, 255, cv::THRESH_BINARY);
    using namespace cv;
#ifdef DEBUGCOLOR
    // cv::imshow("gray_img", gray_img);
    cv::Mat tmp;
    cv::resize(binary_img, tmp, cv::Size(binary_img.size[1] * gp->resize, binary_img.size[0] * gp->resize));
    cv::imshow("binary", tmp);
#endif
#ifdef DILATED
    if (this->detect_color == 1)
    {
        auto kernel = cv::getStructuringElement(cv::MORPH_CROSS, cv::Size(3, 3));
        cv::dilate(binary_img, binary_img, kernel, cv::Point(-1, -1), 2);
#ifdef DEBUGCOLOR
        // cv::imshow("binary__", binary_img);
#endif
    }
    else
    {
        auto kernel = cv::getStructuringElement(cv::MORPH_CROSS, cv::Size(3, 3));
        cv::dilate(binary_img, binary_img, kernel, cv::Point(-1, -1), 1);
    }
#endif
    lights_ = findLights(input, binary_img);
#ifdef DEBUGMODE
    // for (auto light : lights_)
    // {
    //     cv::rectangle(input, light.boundingRect2f(), cv::Scalar(255, 255, 255), 1);
    // }
#endif
    armors_ = matchLights(lights_);
    if (!armors_.empty())
    {
        classifier->extractNumbers(input, armors_, this->detect_color);
        classifier->classify(armors_);
#ifdef CORNER
        for(auto &armor : armors_)
        {
            LightCornerCorrector corner_corrector;
            corner_corrector.correctCorners(armor, gray_img);
        }
#endif
    }
    return armors_;
}

cv::Mat Detector::preprocessImage(const cv::Mat &rgb_img)
{
    cv::Mat gray_img;
    cv::cvtColor(rgb_img, gray_img, cv::COLOR_RGB2GRAY);
    cv::Mat binary_img;
    cv::threshold(gray_img, binary_img, binary_thres, 255, cv::THRESH_BINARY);
    return binary_img;
}

std::vector<Light> Detector::findLights(const cv::Mat &rbg_img, const cv::Mat &binary_img)
{
    using std::vector;
    vector<vector<cv::Point>> contours;
    vector<cv::Vec4i> hierarchy;
#ifndef FYT
    cv::findContours(binary_img, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
#else
    cv::findContours(binary_img, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
#endif
    vector<Light> lights;
    for (const auto &contour : contours)
    {
        if (contour.size() < 6)
            continue;

        auto r_rect = cv::minAreaRect(contour);
        auto light = Light(r_rect);

        if (isLight(light))
        {
            auto rect = light.boundingRect();
            if ( // Avoid assertion failed
                0 <= rect.x && 0 <= rect.width && rect.x + rect.width <= rbg_img.cols && 0 <= rect.y &&
                0 <= rect.height && rect.y + rect.height <= rbg_img.rows)
            {
                int sum_r = 0, sum_b = 0;
                auto roi = rbg_img(rect);
                auto roi_binary = binary_img(rect);
                std::vector<cv::Point2f> binary_points;
                // Iterate through the ROI
                for (int i = 0; i < roi.rows; i++)
                {
                    for (int j = 0; j < roi.cols; j++)
                    {
                        if (cv::pointPolygonTest(contour, cv::Point2f(j + rect.x, i + rect.y), false) >= 0)
                        {
                            sum_r += roi.at<cv::Vec3b>(i, j)[0];
                            sum_b += roi.at<cv::Vec3b>(i, j)[2];
                        }
                    }
                }
                // Sum of red pixels > sum of blue pixels ?
                if (std::abs(sum_r - sum_b) / static_cast<double>(contour.size()) > color_diff_threshold)
                {
                    light.color = sum_r > sum_b ? RED : BLUE;
                }
                lights.emplace_back(light);
            }
        }
    }
    std::sort(lights.begin(), lights.end(), 
    [](const Light &l1, const Light &l2) {
        return l1.center.x < l2.center.x;
    });

    return lights;
}

bool Detector::isLight(const Light &light)
{
    // The ratio of light (short side / long side)
    float ratio = light.width / light.length;
    bool ratio_ok = l.min_ratio < ratio && ratio < l.max_ratio;

    bool angle_ok = light.tilt_angle < l.max_angle;
    bool size_ok = light.length * light.width < 12800 and light.length > 10;
    bool is_light = ratio_ok && angle_ok && size_ok;

    return is_light;
}

std::vector<UnsolvedArmor> Detector::matchLights(const std::vector<Light> &lights)
{
    std::vector<UnsolvedArmor> armors;

    // Loop all the pairing of lights
    for (auto light_1 = lights.begin(); light_1 != lights.end(); light_1++)
    {   
        if( light_1->color != detect_color)
            continue;
        for (auto light_2 = light_1 + 1; light_2 != lights.end(); light_2++)
        {
            if (light_2->color != detect_color)
                continue;
            if (containLight(light_1 - lights.begin(), light_2 - lights.begin(), lights))
            {   
                continue;
            }
            auto type = isArmor(*light_1, *light_2);
            if (type != ArmorType::INVALID)
            {
                auto armor = UnsolvedArmor(*light_1, *light_2);
                armor.type = type;
                armors.emplace_back(armor);
            }
        }
    }

    return armors;
}

// Check if there is another light in the boundingRect formed by the 2 lights
bool Detector::containLight(const int i, const int j, const std::vector<Light> &lights) noexcept {
  const Light &light_1 = lights.at(i), light_2 = lights.at(j);
  auto points = std::vector<cv::Point2f>{light_1.top, light_1.bottom, light_2.top, light_2.bottom};
  auto bounding_rect = cv::boundingRect(points);
  double avg_length = (light_1.length + light_2.length) / 2.0;
  double avg_width = (light_1.width + light_2.width) / 2.0;
  // Only check lights in between
  for (int k = i + 1; k < j; k++) {
    const Light &test_light = lights.at(k);

    // 防止数字干扰
    if (test_light.width > 2 * avg_width) {
      continue;
    }
    // 防止红点准星或弹丸干扰
    if (test_light.length < 0.5 * avg_length) {
      continue;
    }

    if (bounding_rect.contains(test_light.top) || bounding_rect.contains(test_light.bottom) ||
        bounding_rect.contains(test_light.center)) {
      return true;
    }
  }
  return false;
}

ArmorType Detector::isArmor(const Light &light_1, const Light &light_2)
{
    // Ratio of the length of 2 lights (short side / long side)
    float light_length_ratio = light_1.length < light_2.length ? light_1.length / light_2.length
                                                               : light_2.length / light_1.length;
    bool light_ratio_ok = light_length_ratio > a.min_light_ratio;
    // //std::cout<<"light_length_ratio: "<<light_length_ratio<<std::endl;
    // Distance between the center of 2 lights (unit : light length)
    float avg_light_length = (light_1.length + light_2.length) / 2;
    float center_distance = cv::norm(light_1.center - light_2.center) / avg_light_length;
    bool center_distance_ok = (a.min_small_center_distance <= center_distance &&
                               center_distance < a.max_small_center_distance) ||
                              (a.min_large_center_distance <= center_distance &&
                               center_distance < a.max_large_center_distance);

    // Angle of light center connection
    // //std::cout<<"center_distance: "<<center_distance<<std::endl;
    cv::Point2f diff = light_1.center - light_2.center;
    float angle = std::abs(std::atan(diff.y / diff.x)) / CV_PI * 180;
    bool angle_ok = angle < a.max_angle;

    // //std::cout<<"angle: "<<angle<<std::endl;
    bool is_armor = light_ratio_ok && center_distance_ok && angle_ok;

    // Judge armor type
    ArmorType type;
    if (is_armor)
    {
        type = center_distance > a.min_large_center_distance ? ArmorType::LARGE : ArmorType::SMALL;
    }
    else
    {
        type = ArmorType::INVALID;
    }

    return type;
}

cv::Mat Detector::getAllNumbersImage()
{
    if (armors_.empty())
    {
        return cv::Mat(cv::Size(20, 28), CV_8UC1);
    }
    else
    {
        std::vector<cv::Mat> number_imgs;
        number_imgs.reserve(armors_.size());
        for (auto &armor : armors_)
        {
            number_imgs.emplace_back(armor.number_img);
        }
        cv::Mat all_num_img;
        cv::vconcat(number_imgs, all_num_img);
        return all_num_img;
    }
}

void Detector::drawResults(cv::Mat &img)
{
    // Draw Lights
    for (const auto &light : lights_)
    {
        //cv::circle(img, light.top, 3, cv::Scalar(255, 255, 255), 1);
        //cv::circle(img, light.bottom, 3, cv::Scalar(255, 255, 255), 1);
        auto line_color = light.color == RED ? cv::Scalar(255, 255, 0) : cv::Scalar(255, 0, 255);
        cv::line(img, light.top, light.bottom, line_color, 1);
    }

    // Draw armors
    for (const auto &armor : armors_)
    {
        cv::line(img, armor.left_light.top, armor.right_light.bottom, cv::Scalar(0, 255, 0), 2);
        cv::line(img, armor.left_light.bottom, armor.right_light.top, cv::Scalar(0, 255, 0), 2);
    }

    // Show numbers and confidence
    for (const auto &armor : armors_)
    {
        cv::putText(
            img, armor.classfication_result, armor.left_light.top, cv::FONT_HERSHEY_SIMPLEX, 0.8,
            cv::Scalar(0, 255, 255), 2);
    }
}