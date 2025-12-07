#ifndef _WMINFERENCE_HPP
#define _WMINFERENCE_HPP
#include <iostream>
#include <iterator>
#include <memory>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <opencv2/opencv.hpp>

#include "globalParam.hpp"
#include "globalText.hpp"

#include <algorithm>
#include <cstdint>
// #include <ngraph/type/float16.hpp>
#include <opencv2/core/cvdef.h>
#include <opencv2/core/hal/interface.h>
#include <opencv2/videoio.hpp>
// #include <openvino/core/type/float16.hpp>
#include <openvino/openvino.hpp>
#include <string>
#include <vector>
class WMDetector {
public:
  WMDetector();
  ~WMDetector();
  bool detect(cv::Mat &src, std::vector<WMBlade> &objects);

private:
  // ov::Core core;
  address addr;
  ov::Core core;
  ov::CompiledModel compiled_model;
  ov::InferRequest infer_request;
};

#endif //_WMINFERENCE_HPP
