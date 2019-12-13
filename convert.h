
#ifndef CONVERT_H
#define CONVERT_H
#include <vector>
//#include <opencv2/opencv.hpp>

//#include "demo/libImage/image.hpp"
//#include "extras/sift/demo_lib_sift.h"

#include "algorithm.h"



typedef std::pair<size_t, size_t> Pair;

//int Convert_image(const cv::Mat& In, cv::Mat& imag);//convert only gray scale image of opencv

int Convert_detectors(const  std::vector<cv::KeyPoint>& feat1, std::vector<keypoint>& F1);//convert openCV detectors to KVLD suitable detectors
int Convert_detectors(const  std::vector<keypoint>& F1, std::vector<cv::KeyPoint>& feat1);

int Convert_matches(const std::vector<cv::DMatch>& matches, std::vector<Pair>& matchesPair);
int Convert_pairs2matches(const std::vector<Pair>& matchesPair,
	std::vector<cv::DMatch>& matches);

int read_detectors(const std::string& filename, std::vector<keypoint>& feat);//reading openCV style detectors


int read_matches(const std::string& filename, std::vector<cv::DMatch>& matches);//reading openCV style matches

#endif //CONVERT_H