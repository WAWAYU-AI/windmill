#if !defined(__UIMANAGER_HPP)
#define __UIMANAGER_HPP
#include "globalParam.hpp"
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <deque>
#include <glog/logging.h>
#include <iostream>
#include <opencv2/video/video.hpp>
#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <opencv2/imgproc/types_c.h>
#include <variant>

class UIManager
{
private:
    cv::Mat img;
    int row;
    int page;
    int tag;
    int max_row;
    int max_page;
    GlobalParam *gp;
    struct UIElement
    {
        using Ptr = std::variant<int*, float*, double*>;
        Ptr param;
        std::string name;
        char type;
        double delta_small;
        double delta_big;
    };
    std::vector<UIElement> param_list;
public:
    /**
     * @brief WMIdentify的构造函数，初始化一些参数
     *
     * @param gp 全局参数结构体，通过引用输入
     */
    UIManager(GlobalParam &gp, int max_row = 5);
    /**
     * @brief WMIdentify的析构函数，一般来说程序会自行销毁那些需要销毁的东西
     *
     */
    ~UIManager();
    /**
     * @brief 输入图像的接口
     *
     * @param input_img 输入的图像
     */
    void receive_pic(cv::Mat);
    /**
     * @brief 调试过程中的UI系统，方便动态调试
     *
     * @param gp 全局参数结构体，通过引用输入，会改变其中的参数大小
     * @param page 当前的UI的页数
     * @param row 当前的UI的行数
     * @param tag 当前进行的操作，0为不操作，1为增加，2为大量增加，3为减少，4为大量减少
     */
    void windowsManager(int key,int &debug_t);
};

#endif // __UIMANAGER_HPP