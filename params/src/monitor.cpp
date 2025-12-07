// #include "monitor.hpp"
// #include "opencv2/core/mat.hpp"
// #include "opencv2/imgproc.hpp"
// #include <deque>
// #include <opencv2/core/types.hpp>
// #include <string>

// bool polynomial_curve_fit(std::vector<cv::Point> &key_point, int n, cv::Mat &A)
// {
//     // Number of key points
//     int N = key_point.size();

//     // 构造矩阵X
//     cv::Mat X = cv::Mat::zeros(n + 1, n + 1, CV_64FC1);
//     for (int i = 0; i < n + 1; i++)
//     {
//         for (int j = 0; j < n + 1; j++)
//         {
//             for (int k = 0; k < N; k++)
//             {
//                 X.at<double>(i, j) = X.at<double>(i, j) +
//                                      std::pow((double)key_point[k].x, i + j);
//             }
//         }
//     }

//     // 构造矩阵Y
//     cv::Mat Y = cv::Mat::zeros(n + 1, 1, CV_64FC1);
//     for (int i = 0; i < n + 1; i++)
//     {
//         for (int k = 0; k < N; k++)
//         {
//             Y.at<double>(i, 0) = Y.at<double>(i, 0) +
//                                  std::pow((double)key_point[k].x, i) * key_point[k].y;
//         }
//     }

//     A = cv::Mat::zeros(n + 1, 1, CV_64FC1);
//     // 求解矩阵A
//     cv::solve(X, Y, A, cv::DECOMP_LU);
//     return true;
// }
// bool polynomial_curve_fit(std::deque<cv::Point> &key_point, int n, cv::Mat &A)
// {
//     // Number of key points
//     int N = key_point.size();

//     // 构造矩阵X
//     cv::Mat X = cv::Mat::zeros(n + 1, n + 1, CV_64FC1);
//     for (int i = 0; i < n + 1; i++)
//     {
//         for (int j = 0; j < n + 1; j++)
//         {
//             for (int k = 0; k < N; k++)
//             {
//                 X.at<double>(i, j) = X.at<double>(i, j) +
//                                      std::pow((double)key_point[k].x, i + j);
//             }
//         }
//     }

//     // 构造矩阵Y
//     cv::Mat Y = cv::Mat::zeros(n + 1, 1, CV_64FC1);
//     for (int i = 0; i < n + 1; i++)
//     {
//         for (int k = 0; k < N; k++)
//         {
//             Y.at<double>(i, 0) = Y.at<double>(i, 0) +
//                                  std::pow((double)key_point[k].x, i) * key_point[k].y;
//         }
//     }

//     A = cv::Mat::zeros(n + 1, 1, CV_64FC1);
//     // 求解矩阵A
//     cv::solve(X, Y, A, cv::DECOMP_LU);
//     return true;
// }
// void draw(std::deque<cv::Point3f> points3d, std::deque<double> time, int flag, float t_time, float t_points)
// {
//     // 创建用于绘制的深蓝色背景图像
//     cv::Mat image = cv::Mat::zeros(960, 1280, CV_8UC3);
//     image.setTo(cv::Scalar(100, 0, 0));

//     cv::line(image, cv::Point(0, 480), cv::Point(1280, 480), cv::Scalar(255, 255, 255), 1);
//     std::vector<cv::Point> points;
//     int len = 70;
//     if (flag == 1)
//     {
//         for (int i = points3d.size() - len; i < points3d.size(); i++)
//         {
//             points.push_back(cv::Point((time[i] - time[points3d.size() - len]) * t_time, 480 - points3d[i].x * t_points));
//         }
//     }
//     else if (flag == 2)
//     {
//         for (int i = points3d.size() - len; i < points3d.size(); i++)
//         {
//             points.push_back(cv::Point((time[i] - time[points3d.size() - len]) * t_time, 480 - points3d[i].y * t_points));
//         }
//     }
//     else if (flag == 3)
//     {
//         for (int i = points3d.size() - len; i < points3d.size(); i++)
//         {
//             points.push_back(cv::Point((time[i] - time[points3d.size() - len]) * t_time, 480 - points3d[i].z * t_points));
//         }
//     }
//     else if (flag == 4)
//     {
//         for (int i = points3d.size() - len; i < points3d.size(); i++)
//         {
//             points.push_back(cv::Point(800 - points3d[i].x * t_points, 700 - points3d[i].y * t_points));
//         }
//     }
//     else
//     {
//         return;
//     }
//     // 将拟合点绘制到空白图上
//     for (int i = 0; i < points.size(); i++)
//     {
//         if (points[i].y == 480)
//             cv::circle(image, points[i], 5, cv::Scalar(0, 0, 255), -1, 8, 0);
//         else
//             cv::circle(image, points[i], 5, cv::Scalar(193, 182, 255), -1, 8, 0);
//     }
//     if (flag == 1)
//     {
//         cv::putText(image, std::to_string(points3d[points3d.size() - 1].x), points[points.size() - 1], cv::FONT_HERSHEY_PLAIN, 2, cv::Scalar(255, 100, 0));
//     }
//     else if (flag == 2)
//     {
//         cv::putText(image, std::to_string(points3d[points3d.size() - 1].y), points[points.size() - 1], cv::FONT_HERSHEY_PLAIN, 2, cv::Scalar(255, 100, 0));
//     }
//     else if (flag == 3)
//     {
//         cv::putText(image, std::to_string(points3d[points3d.size() - 1].z), points[points.size() - 1], cv::FONT_HERSHEY_PLAIN, 2, cv::Scalar(255, 100, 0));
//     }
//     cv::circle(image, points[points.size() - 1], 5, cv::Scalar(255, 0, 0), -1, 8, 0);
//     // 绘制折线
//     if (flag != 4)
//         cv::polylines(image, points, false, cv::Scalar(0, 255, 0), 1, 8, 0);
//     if (flag == 4)
//     {
//         cv::circle(image, cv::Point2f(800, 700), 5, cv::Scalar(255, 255, 255), -1);
//         cv::line(image, cv::Point2f(800, 700), points[points.size() - 1], cv::Scalar(255, 255, 255), 1);
//     }
//     cv::imshow("image" + std::to_string(flag), image);
//     return;
// }

// void drawRobotState(const Translator &ts, double times, double dt, double r)
// {
//     cv::Mat image = cv::Mat::zeros(960, 1280, CV_8UC3);
//     image.setTo(cv::Scalar(211, 211, 211));
//     double x_c = ts.message.x_c;
//     x_c = 3000.0; // fixed
//     double y_c = ts.message.y_c;
//     // double y_c = 100.0;
//     double x_a = x_c - r * 1000 * cos(ts.message.yaw_a);
//     double y_a = y_c + r * 1000 * sin(ts.message.yaw_a);
//     cv::circle(image, cv::Point(480 - x_c * times, 640 - y_c * times), 5, cv::Scalar(193, 182, 255), -1, 8, 0);
//     cv::circle(image, cv::Point(480 - x_a * times, 640 - y_a * times), 2, cv::Scalar(0, 0, 255), -1, 8, 0);
//     cv::line(image, cv::Point(480 - x_c * times, 640 - y_c * times), cv::Point(480 - x_a * times, 640 - y_a * times), cv::Scalar(0, 0, 255), 1);
//     cv::circle(image, cv::Point(480 - (2 * x_c - x_a) * times, 640 - (2 * y_c - y_a) * times), 2, cv::Scalar(0, 0, 255), -1, 8, 0);
//     cv::line(image, cv::Point(480 - x_c * times, 640 - y_c * times), cv::Point(480 - (2 * x_c - x_a) * times, 640 - (2 * y_c - y_a) * times), cv::Scalar(0, 0, 255), 1);
//     cv::circle(image, cv::Point(480 - (x_c + y_c - y_a) * times, 640 - (y_c - x_c + x_a) * times), 2, cv::Scalar(0, 0, 255), -1, 8, 0);
//     cv::line(image, cv::Point(480 - x_c * times, 640 - y_c * times), cv::Point(480 - (x_c + y_c - y_a) * times, 640 - (y_c - x_c + x_a) * times), cv::Scalar(0, 0, 255), 1);
//     cv::circle(image, cv::Point(480 - (x_c - y_c + y_a) * times, 640 - (y_c + x_c - x_a) * times), 2, cv::Scalar(0, 0, 255), -1, 8, 0);
//     cv::line(image, cv::Point(480 - x_c * times, 640 - y_c * times), cv::Point(480 - (x_c - y_c + y_a) * times, 640 - (y_c + x_c - x_a) * times), cv::Scalar(0, 0, 255), 1);

//     double pred_x_c = x_c + ts.message.vx_c * dt;
//     double pred_y_c = y_c + ts.message.vy_c * dt;
//     double pred_yaw = ts.message.yaw_a + ts.message.vyaw_a * dt;
//     double pred_x_a = pred_x_c - r * 1000 * cos(pred_yaw);
//     double pred_y_a = pred_y_c + r * 1000 * sin(pred_yaw);
//     cv::circle(image, cv::Point(480 - pred_x_c * times, 640 - pred_y_c * times), 5, cv::Scalar(193, 182, 255), -1, 8, 0);
//     cv::circle(image, cv::Point(480 - pred_x_a * times, 640 - pred_y_a * times), 2, cv::Scalar(0, 215, 255), -1, 8, 0);
//     cv::line(image, cv::Point(480 - pred_x_c * times, 640 - pred_y_c * times), cv::Point(480 - pred_x_a * times, 640 - pred_y_a * times), cv::Scalar(255, 0, 0), 1);
//     cv::circle(image, cv::Point(480 - (2 * pred_x_c - pred_x_a) * times, 640 - (2 * pred_y_c - pred_y_a) * times), 2, cv::Scalar(0, 215, 255), -1, 8, 0);
//     cv::line(image, cv::Point(480 - pred_x_c * times, 640 - pred_y_c * times), cv::Point(480 - (2 * pred_x_c - pred_x_a) * times, 640 - (2 * pred_y_c - pred_y_a) * times), cv::Scalar(255, 0, 0), 1);
//     cv::circle(image, cv::Point(480 - (pred_x_c + pred_y_c - pred_y_a) * times, 640 - (pred_y_c - pred_x_c + pred_x_a) * times), 2, cv::Scalar(0, 215, 255), -1, 8, 0);
//     cv::line(image, cv::Point(480 - pred_x_c * times, 640 - pred_y_c * times), cv::Point(480 - (pred_x_c + pred_y_c - pred_y_a) * times, 640 - (pred_y_c - pred_x_c + pred_x_a) * times), cv::Scalar(255, 0, 0), 1);
//     cv::circle(image, cv::Point(480 - (pred_x_c - pred_y_c + pred_y_a) * times, 640 - (pred_y_c + pred_x_c - pred_x_a) * times), 2, cv::Scalar(0, 215, 255), -1, 8, 0);
//     cv::line(image, cv::Point(480 - pred_x_c * times, 640 - pred_y_c * times), cv::Point(480 - (pred_x_c - pred_y_c + pred_y_a) * times, 640 - (pred_y_c + pred_x_c - pred_x_a) * times), cv::Scalar(255, 0, 0), 1);
//     cv::line(image, cv::Point2f(480 - ts.message.x_a * times, 640 - ts.message.y_a * times), cv::Point2f(480, 640), cv::Scalar(255, 255, 255), 1);
//     cv::imshow("image", image);
// }

// void drawStationState(const Translator &ts, double times, double dt, double r)
// {
//     cv::Mat image = cv::Mat::zeros(960, 1280, CV_8UC3);
//     image.setTo(cv::Scalar(211, 211, 211));
//     double x_c = ts.message.x_c;
//     double y_c = ts.message.y_c;
//     double x_a = x_c - r * 1000 * cos(ts.message.yaw_a);
//     double y_a = y_c + r * 1000 * sin(ts.message.yaw_a);
//     cv::circle(image, cv::Point(480 - x_c * times, 640 - y_c * times), 5, cv::Scalar(193, 182, 255), -1, 8, 0);
//     cv::circle(image, cv::Point(480 - x_a * times, 640 - y_a * times), 2, cv::Scalar(0, 0, 255), -1, 8, 0);
//     cv::line(image, cv::Point(480 - x_c * times, 640 - y_c * times), cv::Point(480 - x_a * times, 640 - y_a * times), cv::Scalar(0, 0, 255), 1);
//     cv::circle(image, cv::Point(480, 640) - cv::Point(x_c + cos(2 * M_PI / 3) * (x_a - x_c) - sin(2 * M_PI / 3) * (y_a - y_c), y_c + sin(2 * M_PI / 3) * (x_a - x_c) + cos(2 * M_PI / 3) * (y_a - y_c)) * times, 2, cv::Scalar(0, 0, 255), -1, 8, 0);
//     cv::line(image, cv::Point(480 - x_c * times, 640 - y_c * times), cv::Point(480, 640) - cv::Point(x_c + cos(2 * M_PI / 3) * (x_a - x_c) - sin(2 * M_PI / 3) * (y_a - y_c), y_c + sin(2 * M_PI / 3) * (x_a - x_c) + cos(2 * M_PI / 3) * (y_a - y_c)) * times, cv::Scalar(0, 0, 255), 1);
//     cv::circle(image, cv::Point(480, 640) - cv::Point(x_c + cos(2 * M_PI / 3) * (x_a - x_c) + sin(2 * M_PI / 3) * (y_a - y_c), y_c - sin(2 * M_PI / 3) * (x_a - x_c) + cos(2 * M_PI / 3) * (y_a - y_c)) * times, 2, cv::Scalar(0, 0, 255), -1, 8, 0);
//     cv::line(image, cv::Point(480 - x_c * times, 640 - y_c * times), cv::Point(480, 640) - cv::Point(x_c + cos(2 * M_PI / 3) * (x_a - x_c) + sin(2 * M_PI / 3) * (y_a - y_c), y_c - sin(2 * M_PI / 3) * (x_a - x_c) + cos(2 * M_PI / 3) * (y_a - y_c)) * times, cv::Scalar(0, 0, 255), 1);

//     double pred_x_c = x_c + ts.message.vx_c * dt;
//     double pred_y_c = y_c + ts.message.vy_c * dt;
//     double pred_yaw = ts.message.yaw_a + ts.message.vyaw_a * dt;
//     double pred_x_a = pred_x_c - r * 1000 * cos(pred_yaw);
//     double pred_y_a = pred_y_c + r * 1000 * sin(pred_yaw);
//     cv::circle(image, cv::Point(480 - pred_x_c * times, 640 - pred_y_c * times), 5, cv::Scalar(193, 182, 255), -1, 8, 0);

//     cv::circle(image, cv::Point(480 - pred_x_a * times, 640 - pred_y_a * times), 2, cv::Scalar(0, 0, 255), -1, 8, 0);

//     cv::line(image, cv::Point(480 - pred_x_c * times, 640 - pred_y_c * times), cv::Point(480 - pred_x_a * times, 640 - pred_y_a * times), cv::Scalar(0, 0, 255), 1);

//     cv::circle(image, cv::Point(480, 640) - cv::Point(pred_x_c + cos(2 * M_PI / 3) * (pred_x_a - pred_x_c) - sin(2 * M_PI / 3) * (pred_y_a - pred_y_c), pred_y_c + sin(2 * M_PI / 3) * (pred_x_a - pred_x_c) + cos(2 * M_PI / 3) * (pred_y_a - pred_y_c)) * times, 2, cv::Scalar(0, 0, 255), -1, 8, 0);

//     cv::line(image, cv::Point(480 - pred_x_c * times, 640 - pred_y_c * times), cv::Point(480, 640) - cv::Point(pred_x_c + cos(2 * M_PI / 3) * (pred_x_a - pred_x_c) - sin(2 * M_PI / 3) * (pred_y_a - pred_y_c), pred_y_c + sin(2 * M_PI / 3) * (pred_x_a - pred_x_c) + cos(2 * M_PI / 3) * (pred_y_a - pred_y_c)) * times, cv::Scalar(0, 0, 255), 1);

//     cv::circle(image, cv::Point(480, 640) - cv::Point(pred_x_c + cos(2 * M_PI / 3) * (pred_x_a - pred_x_c) + sin(2 * M_PI / 3) * (pred_y_a - pred_y_c), pred_y_c - sin(2 * M_PI / 3) * (pred_x_a - pred_x_c) + cos(2 * M_PI / 3) * (pred_y_a - pred_y_c)) * times, 2, cv::Scalar(0, 0, 255), -1, 8, 0);

//     cv::line(image, cv::Point(480 - pred_x_c * times, 640 - pred_y_c * times), cv::Point(480, 640) - cv::Point(pred_x_c + cos(2 * M_PI / 3) * (pred_x_a - pred_x_c) + sin(2 * M_PI / 3) * (pred_y_a - pred_y_c), pred_y_c - sin(2 * M_PI / 3) * (pred_x_a - pred_x_c) + cos(2 * M_PI / 3) * (pred_y_a - pred_y_c)) * times, cv::Scalar(0, 0, 255), 1);
//     cv::imshow("image", image);
// }

// void drawStat(std::deque<cv::Point3f> &points3d, std::deque<double> &times, Translator &translator)
// {
// #ifdef DETAILEDINFO
//     points3d.push_back(cv::Point3f(translator.message.x_c, translator.message.y_c, translator.message.yaw_a));

//     if (points3d.size() >= 70)
//     {
//         draw(points3d, times, 1, 1000, 0.03);
//         draw(points3d, times, 2, 1000, 1.0);
//         draw(points3d, times, 3, 1000, 100.0);
//         draw(points3d, times, 4, 600, 0.1);
//         points3d.pop_front();
//         times.pop_front();
//     }
// #endif
// }

// void resizePic(cv::Mat &pic)
// {
//     int midx = pic.size[1] / 2;
//     int midy = pic.size[0];
//     cv::Point startPoint(midx, 0);
//     cv::Point endPoint(midx, midy);
//     cv::line(pic, startPoint, endPoint, cv::Scalar(255, 0, 255), 2);
//     cv::resize(pic, pic, cv::Size(720, 540));
// }
