#include "globalParam.hpp"
#include <glog/logging.h>
#include <filesystem>
#include <iostream>
void GlobalParam::initGlobalParam(const int color)
{   
    cv::FileStorage fs;
    this ->color = color;

    // 打开CameraConfig配置文件
    if(!fs.open("../config/CameraConfig.yaml", cv::FileStorage::READ)){
        printf("CameraConfig.yaml not found!\n");
        exit(1);
    }
    fs["cam_index"] >> cam_index;
    fs["enable_auto_exp"] >> enable_auto_exp;
    fs["energy_exp_time"] >> energy_exp_time;
    // fs["armor_exp_time"] >> armor_exp_time;
    fs["blue_exp_time"] >> blue_exp_time;
    fs["red_exp_time"] >> red_exp_time;
    fs["r_balance"] >> r_balance;
    fs["g_balance"] >> g_balance;
    fs["b_balance"] >> b_balance;
    fs["e_r_balance"] >> e_r_balance;
    fs["e_g_balance"] >> e_g_balance;
    fs["e_b_balance"] >> e_b_balance;
    fs["enable_auto_gain"] >> enable_auto_gain;
    fs["gain"] >> gain;
    fs["gamma_value"] >> gamma_value;
    fs["trigger_activation"] >> trigger_activation;
    fs["frame_rate"] >> frame_rate;
    fs["enable_trigger"] >> enable_trigger;
    fs["trigger_source"] >> trigger_source;
    fs["cx"] >> cx;  // cx
    fs["cy"] >> cy;  // cy
    fs["fx"] >> fx;  // fx
    fs["fy"] >> fy;  // fy
    fs["k1"] >> k1;
    fs["k2"] >> k2;
    fs["k3"] >> k3;
    fs["p1"] >> p1;
    fs["p2"] >> p2;
    fs["vector_x"] >> vector_x;
    fs["vector_y"] >> vector_y;
    fs["vector_z"] >> vector_z;
    fs.release();
    
    // 打开AimautoConfig配置文件
    if(!fs.open("../config/AimautoConfig.yaml", cv::FileStorage::READ)){
        printf("AimautoConfig.yaml not found!\n");
        exit(1);
    }
    fs["cost_threshold"] >> cost_threshold;
    fs["max_lost_frame"] >> max_lost_frame;
    // 卡尔曼滤波相关参数
    fs["s2qxyz"] >> s2qxyz;              // 位置转移噪声
    fs["s2qyaw"] >> s2qyaw;              // 角度转移噪声
    fs["s2qr"] >> s2qr;                  // 半径转移噪声
    fs["r_xy_factor"] >> r_xy_factor;     
    fs["r_z"] >> r_z;     
    fs["r_yaw"] >> r_yaw;
    fs["s2p0xyr"] >> s2p0xyr;
    fs["s2p0yaw"] >> s2p0yaw;
    fs["r_initial"] >> r_initial;
    fs["small_armor_a"] >> small_armor_a; // 小装甲板的长
    fs["small_armor_b"] >> small_armor_b; // 小装甲板的宽
    fs["big_armor_a"] >> big_armor_a;     // 大装甲板的长
    fs["big_armor_b"] >> big_armor_b;     // 大装甲板的宽
    fs["resize"] >> resize;
    fs.release();

    // 打开DetectionConfig配置文件
    if(!fs.open("../config/DetectionConfig.yaml", cv::FileStorage::READ)){
        printf("DetectionConfig.yaml not found!\n");
        exit(1);
    }
    fs["min_ratio"] >> min_ratio;
    fs["max_ratio"] >> max_ratio;
    fs["max_angle_l"] >> max_angle_l;
    fs["min_light_ratio"] >> min_light_ratio;
    fs["min_small_center_distance"] >> min_small_center_distance;
    fs["max_small_center_distance"] >> max_small_center_distance;
    fs["min_large_center_distance"] >> min_large_center_distance;
    fs["max_large_center_distance"] >> max_large_center_distance;
    fs["max_angle_a"] >> max_angle_a;
    fs["num_threshold"] >> num_threshold;
    fs["blue_threshold"] >> blue_threshold;
    fs["red_threshold"] >> red_threshold;
    fs["grad_max"] >> grad_max;
    fs["grad_min"] >> grad_min;
    fs.release();

    if (!fs.open("../config/WindmillConfig.yaml", cv::FileStorage::READ)) {
        printf("WindmillConfig.yaml not found!\n");
        exit(1);
      }
      fs["circularityThreshold"] >> circularityThreshold;
      fs["medianBlurSize"] >> medianBlurSize;
      fs["medianBlurSize_1"] >> medianBlurSize_1;
      fs["debug"] >> debug;
      fs["dilationSize"] >> dilationSize;
      fs["dilationSize_1"] >> dilationSize_1;
      fs["erosionSize"] >> erosionSize;
      fs["erosionSize_1"] >> erosionSize_1;
      fs["thresholdValue"] >> thresholdValue;
      fs["thresholdValue_1"] >> thresholdValue_1;
      fs["thresholdValueBlue"] >> thresholdValueBlue;
      fs["thresholdValueBlue_1"] >> thresholdValueBlue_1;
      fs["thresholdValue_for_roi"] >> thresholdValue_for_roi;
      fs["rect_area_threshold"] >> rect_area_threshold;
      fs["circle_area_threshold"] >> circle_area_threshold;
      fs["length_width_ratio_threshold"] >> length_width_ratio_threshold;
      fs["minContourArea"] >> minContourArea;
      fs["target_circle_area_min"] >> target_circle_area_min;
      fs["target_circle_area_max"] >> target_circle_area_max;
      fs["R_area_min"] >> R_area_min;
      fs["R_area_max"] >> R_area_max;
    
      fs["list_size"] >> list_size;
      fs["d_Radius"] >> d_Radius;
      fs["d_P1P3"] >> d_P1P3;
      fs["d_RP2"] >> d_RP2;
    
      fs["gap"] >> gap;
      fs["gap_control"] >> gap_control;
    
      fs["tx_cam2cloud"] >> tx_cam2cloud;
      fs["tx_cam2cloud_1"] >> tx_cam2cloud_1;
      fs["ty_cam2cloud"] >> ty_cam2cloud;
      fs["ty_cam2cloud_1"] >> ty_cam2cloud_1;
      fs["tz_cam2cloud"] >> tz_cam2cloud;
      fs["tz_cam2cloud_1"] >> tz_cam2cloud_1;
    
      fs["delta_t"] >> delta_t;
    
      fs.release();
    
    // LOG_IF(INFO, switch_INFO) << "initGlobalParam Successful";
}

void GlobalParam::saveGlobalParam()
{
    cv::FileStorage fs;

    // 打开CameraConfig配置文件以写入参数
    if(!fs.open("../config/CameraConfig.yaml", cv::FileStorage::WRITE)){
        printf("CameraConfig.yaml not found!\n");
        exit(1);
    }
    fs << "cam_index" << cam_index;
    fs << "enable_auto_exp" << enable_auto_exp;
    fs << "energy_exp_time" << energy_exp_time;
    // fs << "armor_exp_time" << armor_exp_time;
    fs << "blue_exp_time" << blue_exp_time;
    fs << "red_exp_time" << red_exp_time;
    fs << "r_balance" << r_balance;
    fs << "g_balance" << g_balance;
    fs << "b_balance" << b_balance;
    fs << "enable_auto_gain" << enable_auto_gain;
    fs << "gain" << gain;
    fs << "gamma_value" << gamma_value;
    fs << "trigger_activation" << trigger_activation;
    fs << "frame_rate" << frame_rate;
    fs << "enable_trigger" << enable_trigger;
    fs << "trigger_source" << trigger_source;
    fs << "cx" << cx;  // cx
    fs << "cy" << cy;  // cy
    fs << "fx" << fx;  // fx
    fs << "fy" << fy;  // fy
    fs << "k1" << k1;
    fs << "k2" << k2;
    fs << "k3" << k3;
    fs << "p1" << p1;
    fs << "p2" << p2;
    fs << "vector_x" << vector_x;
    fs << "vector_y" << vector_y;
    fs << "vector_z" << vector_z;
    fs.release();

    // 打开AimautoConfig配置文件以写入参数
    if(!fs.open("../config/AimautoConfig.yaml", cv::FileStorage::WRITE)){
        printf("AimautoConfig.yaml not found!\n");
        exit(1);
    }
    fs << "cost_threshold" << cost_threshold;
    fs << "max_lost_frame" << max_lost_frame;
    // 卡尔曼滤波相关参数
    fs << "s2qxyz" << s2qxyz;                // 位置转移噪声
    fs << "s2qyaw" << s2qyaw;                // 角度转移噪声
    fs << "s2qr" << s2qr;                    // 半径转移噪声
    fs << "r_xy_factor" << r_xy_factor;     
    fs << "r_z" << r_z;    
    fs << "r_yaw" << r_yaw;    
    fs << "s2p0xyr" << s2p0xyr;
    fs << "s2p0yaw" << s2p0yaw;
    fs << "r_initial" << r_initial;
    fs << "small_armor_a" << small_armor_a; // 小装甲板的长
    fs << "small_armor_b" << small_armor_b; // 小装甲板的宽
    fs << "big_armor_a" << big_armor_a;     // 大装甲板的长
    fs << "big_armor_b" << big_armor_b;     // 大装甲板的宽
    fs << "resize" << resize;
    fs.release();              

    // 打开DetectionConfig配置文件以写入参数
    if(!fs.open("../config/DetectionConfig.yaml", cv::FileStorage::WRITE)){
        printf("DetectionConfig.yaml not found!\n");
        exit(1);
    }
    fs << "min_ratio" << min_ratio;
    fs << "max_ratio" << max_ratio;
    fs << "max_angle_l" << max_angle_l;
    fs << "min_light_ratio" << min_light_ratio;
    fs << "min_small_center_distance" << min_small_center_distance;
    fs << "max_small_center_distance" << max_small_center_distance;
    fs << "min_large_center_distance" << min_large_center_distance;
    fs << "max_large_center_distance" << max_large_center_distance;
    fs << "max_angle_a" << max_angle_a;
    fs << "num_threshold" << num_threshold;
    fs << "blue_threshold" << blue_threshold;
    fs << "red_threshold" << red_threshold;
    fs << "grad_max" << grad_max;
    fs << "grad_min" << grad_min;
    fs.release();

    if (!fs.open("../config/WindmillConfig.yaml", cv::FileStorage::READ)) {
        printf("WindmillConfig.yaml not found!\n");
        exit(1);
      }
      fs << "circularityThreshold" << circularityThreshold;
      fs << "medianBlurSize" << medianBlurSize;
      fs << "medianBlurSize_1" << medianBlurSize_1;
      fs << "dilationSize" << dilationSize;
      fs << "dilationSize_1" << dilationSize_1;
      fs << "erosionSize" << erosionSize;
      fs << "erosionSize_1" << erosionSize_1;
      fs << "thresholdValue" << thresholdValue;
      fs << "thresholdValue_1" << thresholdValue_1;
      fs << "thresholdValueBlue" << thresholdValueBlue;
      fs << "thresholdValueBlue_1" << thresholdValueBlue_1;
      fs << "thresholdValue_for_roi" << thresholdValue_for_roi;
      fs << "rect_area_threshold" << rect_area_threshold;
      fs << "circle_area_threshold" << circle_area_threshold;
      fs << "length_width_ratio_threshold" << length_width_ratio_threshold;
      fs << "minContourArea" << minContourArea;
      fs << "target_circle_area_min" << target_circle_area_min;
      fs << "target_circle_area_max" << target_circle_area_max;
      fs << "R_area_min" << R_area_min;
      fs << "R_area_max" << R_area_max;
    
      fs << "list_size" << list_size;
      fs << "d_Radius" << d_Radius;
      fs << "d_P1P3" << d_P1P3;
      fs << "d_RP2" << d_RP2;
    
      fs << "gap" << gap;
      fs << "gap_control" << gap_control;
    
      fs << "tx_cam2cloud" << tx_cam2cloud;
      fs << "tx_cam2cloud_1" << tx_cam2cloud_1;
      fs << "ty_cam2cloud" << ty_cam2cloud;
      fs << "ty_cam2cloud_1" << ty_cam2cloud_1;
      fs << "tz_cam2cloud" << tz_cam2cloud;
      fs << "tz_cam2cloud_1" << tz_cam2cloud_1;
    
      fs << "delta_t" << delta_t;
      fs.release();
    

    // LOG_IF(INFO, switch_INFO) << "saveGlobalParam Successful";
}