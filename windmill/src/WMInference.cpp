// #include "WMInference.hpp"
// int NUM_CLASSES_ = 1;
// int NUM_KEYPOINTS_ = 5;
// float IMG_WIDTH_ = 640.;
// float IMG_HEIGHT_ = 640.;
// cv::Mat letterbox_WM(const cv::Mat &source) {
//   int col = source.cols;
//   int row = source.rows;
//   int _max = MAX(col, row);
//   cv::Mat result = cv::Mat::zeros(_max, _max, CV_8UC3);
//   source.copyTo(result(cv::Rect(0, 0, col, row)));
//   // result.convertTo(result, CV_32F);
//   return result;
// }

// WMDetector::WMDetector()
//     : compiled_model(core.compile_model(
//           addr.model_address + "best.xml", "CPU",
//           ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY))),
//       infer_request(compiled_model.create_infer_request()) {}
// WMDetector::~WMDetector() {}
// bool WMDetector::detect(cv::Mat &img, std::vector<WMBlade> &objects) {
//   if (img.empty()) {
//     std::cout << "empty input" << std::endl;
//     return false;
//   }
//   cv::Mat letterbox_img = letterbox_WM(img);
//   float scale = letterbox_img.size[0] / IMG_WIDTH_;

//   cv::Mat blob = cv::dnn::blobFromImage(letterbox_img, 1.0 / 255.0,
//                                         cv::Size(IMG_WIDTH_, IMG_HEIGHT_),
//                                         cv::Scalar(), true);

//   auto input_port = compiled_model.input();
//   ov::Tensor input_tensor(input_port.get_element_type(), input_port.get_shape(),
//                           blob.ptr(0));
//   infer_request.set_input_tensor(input_tensor);

//   std::chrono::milliseconds start_ms =
//       std::chrono::duration_cast<std::chrono::milliseconds>(
//           std::chrono::system_clock::now().time_since_epoch());

//   infer_request.infer();

//   std::chrono::milliseconds end_ms =
//       std::chrono::duration_cast<std::chrono::milliseconds>(
//           std::chrono::system_clock::now().time_since_epoch());
//   float total_time = (end_ms - start_ms).count() / 1000.0;

//   auto output = infer_request.get_output_tensor(0);
//   auto output_shape = output.get_shape();
//   float *data = output.data<float>();

//   // std::cout << "输出张量形状: [";
//   // for (auto dim : output_shape) {
//   //   std::cout << dim << ", ";
//   // }
//   // std::cout << "]" << std::endl;

//   // std::cout << "输出数据前10个值: ";
//   // for (int i = 0; i < 10; i++) {
//   //   std::cout << data[i] << " ";
//   // }
//   // std::cout << std::endl;

//   cv::Mat output_buffer(output_shape[1], output_shape[2], CV_32F, data);
//   std::cout << "转换后矩阵大小: " << output_buffer.size() << std::endl;
//   transpose(output_buffer, output_buffer);
//   std::cout << "转置后矩阵大小: " << output_buffer.size() << std::endl;

//   float score_threshold = 0.2;
//   float nms_threshold = 0.2;

//   std::vector<int> labels;
//   std::vector<int> class_ids;
//   std::vector<float> class_scores;
//   std::vector<cv::Rect> boxes;
//   std::vector<std::vector<cv::Point2f>> objects_keypoints;
//   objects.clear();

//   for (int i = 0; i < output_buffer.rows; i++) {
//     auto row_ptr = output_buffer.row(i).ptr<float>();

//     auto bboxes_ptr = row_ptr;
//     auto score = row_ptr[4];
//     auto kps_ptr = row_ptr + 5; // 关键点数据指针（从index 5开始）

//     if (score > score_threshold) {
//       class_scores.push_back(score);
//       class_ids.push_back(0);

//       float cx = *bboxes_ptr++;
//       float cy = *bboxes_ptr++;
//       float w = *bboxes_ptr++;
//       float h = *bboxes_ptr++;

//       int left = int((cx - 0.5 * w) * scale);
//       int top = int((cy - 0.5 * h) * scale);
//       int width = int(w * scale);
//       int height = int(h * scale);

//       std::vector<cv::Point2f> keypoints;
//       for (int i = 0; i < NUM_KEYPOINTS_; i++) {
//         float x = *(kps_ptr + i * 3) * scale;     // 每组的第1个值是x
//         float y = *(kps_ptr + i * 3 + 1) * scale; // 每组的第2个值是y
//         float kp_conf = *(kps_ptr + i * 3 + 2); // 每组的第3个值是置信度

//         if (kp_conf > 0.1) {
//           keypoints.push_back(cv::Point2f(x, y));
//         } else {
//           keypoints.push_back(cv::Point2f(-1, -1));
//         }
//       }

//       boxes.push_back(cv::Rect(left, top, width, height));
//       objects_keypoints.push_back(keypoints);
//     }
//   }

//   std::vector<int> indices;
//   cv::dnn::NMSBoxes(boxes, class_scores, score_threshold, nms_threshold,
//                     indices);

//   objects.clear();
//   for (auto &i : indices) {
//     WMBlade obj;
//     obj.cls = class_ids[i];
//     obj.prob = class_scores[i];
//     obj.color = obj.cls / 3;
//     obj.cls %= 3;
//     for (int j = 0; j < NUM_KEYPOINTS_; j++) {
//       obj.apex[j] = objects_keypoints[i][j];
//     }
//     objects.push_back(obj);
//   }
//   return true;
// }
