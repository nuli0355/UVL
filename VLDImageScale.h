#pragma once

#include<opencv2/opencv.hpp>

class VLDImageScale
{
public:
	VLDImageScale();
	~VLDImageScale();

	std::vector<cv::Mat> images;
	std::vector<cv::Mat> angles;
	std::vector<cv::Mat> magnitudes;
	std::vector<double> ratios;
	double radius_size;
	double step;
	int nOctaves;

	//ImageScale();
	void createScale(const cv::Mat& I, double r = 5);
	int getIndex(const double r)const;

private:
	void GradAndNorm(const cv::Mat& I, cv::Mat& angle, cv::Mat& m);
};

