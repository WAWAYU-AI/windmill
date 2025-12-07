/**
 * @file WMIdentify_UI.cpp
 * @author 高宁 (3406402603@qq.com)
 * @brief 因为调试过程中的UI系统与本体过于无关，而且因为架构比较无脑，很占地方，所以将其与本身的WMIdentify.cpp分开
 * @version 0.1
 * @date 2023-02-20
 *
 * @copyright Copyright (c) 2023
 *
 */
#include <UIManager.hpp>

UIManager::UIManager(GlobalParam &gp, int max_row)
{
    this->gp = &gp;
    this->page = 1;
    this->row = 1;
    param_list = {
      {&gp.blue_exp_time, "blue_exp_time", '+', 10, 100},
      {&gp.red_exp_time, "red_exp_time", '+', 10, 100},
      {&gp.blue_threshold, "blue_threshold", '+', 1, 10},
      {&gp.red_threshold, "red_threshold", '+', 1, 10},
      {&gp.cost_threshold, "cost_threshold", '+', 10, 100},
      {&gp.max_lost_frame, "max_lost_frame", '+', 1, 5},
      {&gp.s2qxyz, "s2qxyz", '*', 1.1, 2},
      {&gp.s2qr, "s2qr", '*', 1.1, 2},
      {&gp.s2qyaw, "s2qyaw", '*', 1.1, 2},
      {&gp.s2p0xyr, "s2p0xyr", '*', 1.1, 2},
      {&gp.s2p0yaw, "s2p0yaw", '*', 1.1, 2},
      {&gp.r_xy_factor, "r_xy_factor", '*', 1.1, 2},
      {&gp.r_z, "r_z", '*', 1.1, 2},
      {&gp.r_yaw, "r_yaw", '*', 1.1, 2},
      {&gp.r_initial, "r_initial", '+', 10, 100},
      {&gp.small_armor_a, "small_armor_a", '+', 0.1, 10},
      {&gp.small_armor_b, "small_armor_b", '+', 0.1, 10},
      {&gp.big_armor_a, "big_armor_a", '+', 0.1, 10},
      {&gp.big_armor_b, "big_armor_b", '+', 0.1, 10},
    };
    this->max_row = max_row;
    this->max_page = ((int)param_list.size() - 1) / max_row + 1;

}

UIManager::~UIManager()
{
}

void UIManager::receive_pic(cv::Mat img)
{
    this->img = img;
}

void UIManager::windowsManager(int key, int &debug_t)
{
    tag = 0;
    if (key == 'w' || key == 'W')   //上
    {
        if (row > 1)
            row--;
    }
    if (key == 's' || key == 'S')   //下
    {
        if (row < max_row)
            row++;
        if ((page - 1) * max_row + row - 1 >= (int)param_list.size())
            row = (int)param_list.size() % max_row;
    }
    if (key == 'a' || key == 'A')   //左
    {
        if (page > 1)
            page--;
    }
    if (key == 'd' || key == 'D')   //右
    {
        if (page < max_page)
            page++;
    }
    if (key == 'o' || key == 'O')   
    {
        debug_t *= 2;
    }
    if (key == 'l' || key == 'L')
    {
        if (debug_t > 1)
            debug_t /= 2;
        else
            debug_t = 1;
    }
    if (key == 'u' || key == 'U')   //少量增加
    {
        tag = 1;
    }
    if (key == 'j' || key == 'J')   //少量减少
    {
        tag = 3;
    }
    if (key == 'i' || key == 'I')   //大量增加
    {
        tag = 2;
    }
    if (key == 'k' || key == 'K')   //大量减少
    {
        tag = 4;
    }
    cv::putText(this->img, "Page:" + std::to_string(page) + "  Left:A  Right:D", cv::Point(1200, 20), 2, 0.5, cv::Scalar(255, 255, 0));
    cv::putText(this->img, "Row:" + std::to_string(row) + "  Up:W  Down:S", cv::Point(1200, 45), 2, 0.5, cv::Scalar(255, 255, 0));
    
    int st = (page - 1) * max_row;
    int ed = std::min(st + max_row, (int)param_list.size());
    for (int i = st; i < ed; i++){
        auto &param = param_list[i];
        int id = i - st + 1;
        std::visit([param, id, this](auto &value) {
            if (id == row){
                cv::putText(this->img, param.name + ": " + std::to_string(*value), cv::Point(20, 20 + 30 * (id - 1)), 2, 0.8, cv::Scalar(255, 0, 0));
                if (tag == 1){
                    if (param.type == '+')
                        *value += param.delta_small;
                    if (param.type == '*')
                        *value *= param.delta_small;
                }
                if (tag == 2){
                    if (param.type == '+')
                        *value += param.delta_big;
                    if (param.type == '*')
                        *value *= param.delta_big;
                }
                if (tag == 3){
                    if (param.type == '+')
                        *value -= param.delta_small;
                    if (param.type == '*')
                        *value /= param.delta_small;
                }
                if (tag == 4){
                    if (param.type == '+')
                        *value -= param.delta_big;
                    if (param.type == '*')
                        *value /= param.delta_big;
                }
            }else{
                cv::putText(this->img, param.name + ": " + std::to_string(*value), cv::Point(20, 20 + 30 * (id - 1)), 2, 0.8, cv::Scalar(0, 0, 255));
            } 
        }, param.param);
    }

    if (key == 'm' || key == 'M'){   //保存参数
        cv::Mat img = cv::Mat::zeros(50, 400, CV_8UC3);
        cv::putText(img, "Save? Press M to confirm", cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 255, 255), 2, cv::LINE_AA);
        cv::imshow("Save?", img);
        key = cv::waitKey();
        if(key == 'm' || key == 'M'){
            gp->saveGlobalParam();
            printf("Save success\n");
        }
        cv::destroyWindow("Save?");
    }
    if (key == '\n' || key == '\r'){    //修改参数
        std::string input;
        while(true){
            cv::Mat img = cv::Mat::zeros(50, 400, CV_8UC3);
            cv::putText(img, input, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 255, 255), 2, cv::LINE_AA);
            cv::imshow("input", img);
            key = cv::waitKey();
            if(key == '\n' || key == '\r'){
                break;
            }
            if((key >= '0' && key <= '9') || key == '.' || key == '-'){
                input += (char)key;
            }
            if(key == 8){
                if (input.empty()) continue;
                input.pop_back();
            }
        }
        try{
            double value = std::stod(input);
            auto &param = param_list[(page - 1) * max_row + row - 1];
            std::visit([&value](auto &param_value) {
                *param_value = value;
            }, param.param);
        }catch(...){
            printf("Invalid input\n");
        }
        cv::destroyWindow("input");
    }
}