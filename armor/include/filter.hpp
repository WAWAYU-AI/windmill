#include <iostream>
#include <vector>
#include <deque>
#include <cmath>
#include <utility> // for std::pair

// class Filter {
// private:
//     std::deque<std::pair<double, double>> window; // {time, value}
//     double timeWindow; // 时间窗口长度(秒)
    
// public:
//     Filter(double windowSize=0.2) : timeWindow(windowSize) {}
    
//     double filter(double x, double dt) {
//         // 添加新样本
//         double t=0;
//         if (!window.empty()) t = window.back().first + dt;
//         window.emplace_back(t, x);
        
//         // 移除超出时间窗口的旧样本
//         while (!window.empty() && (t - window.front().first) > timeWindow) {
//             window.pop_front();
//         }
        
//         // 计算加权平均
//         double sum = 0.0;
//         double weightSum = 0.0;
//         double oldestTime = window.front().first;
        
//         for (const auto& sample : window) {
//             double timeSinceOldest = sample.first - oldestTime;
//             // 线性权重：越新的样本权重越大
//             double weight = timeSinceOldest + 0.1; // 加0.1避免零权重
//             sum += sample.second * weight;
//             weightSum += weight;
//         }
        
//         return sum / weightSum;
//     }
    
//     void reset() {
//         window.clear();
//     }
// };

class Filter {
    private:
        double cutoffFreq;  // 截止频率(Hz)
        double y_prev;      // 上一个输出值
    
    public:
        Filter(double cutoff = 0.1) 
            : cutoffFreq(cutoff), y_prev(0.0){}
    
        // 滤波函数
        // x: 当前采样值
        // t: 当前时间(秒)
        double filter(double x, double dt) {
            if (y_prev == 0.0) { // 第一次调用
                y_prev = x;
                return x;
            }
    
            double RC = 1.0 / (2 * M_PI * cutoffFreq);
            double alpha = dt / (RC + dt);
            
            y_prev = alpha * x + (1 - alpha) * y_prev;
            
            return y_prev;
        }
    
        void reset() {
            y_prev = 0.0;
        }
    };