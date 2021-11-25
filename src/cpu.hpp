#ifndef CPU_HPP
 #define CPU_HPP

#include <opencv2/opencv.hpp>

cv::Mat get_grayscale(const std::string img_path);
void get_sobel(cv::Mat img, char type, cv::Mat &dst);
void average_pooling(cv::Mat src, cv::Mat &dst, int pool_size);
void post_processing(cv::Mat src, cv::Mat &dst, int postproc_size);
void threshold(cv::Mat src, cv::Mat &dst);
void process_cpu(const std::string img_path);

#endif