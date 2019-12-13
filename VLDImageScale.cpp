#include "VLDImageScale.h"
#include "algorithm.h"


VLDImageScale::VLDImageScale()
{
}


VLDImageScale::~VLDImageScale()
{
}


void VLDImageScale::createScale(const cv::Mat& Image, double r)
{
	cv::Mat I;
	Image.convertTo(I, CV_32FC1);
	//IntegralImages inter(I);
	radius_size = r;
	step = sqrt(2.0);
	int size = std::max(I.cols, I.rows);

	nOctaves = int(log(size / r) / log(2.0)) + 1;
	images.resize(nOctaves);
	angles.resize(nOctaves);
	magnitudes.resize(nOctaves);
	ratios.resize(nOctaves);

	images[0] = I.clone();
	//计算最底层图像的梯度值和梯度方向
	GradAndNorm(I, angles[0], magnitudes[0]);
	ratios[0] = 1;//底层图像的尺度比例是1

	cv::parallel_for_(cv::Range(0, nOctaves), [&](const cv::Range& range)-> void
	{
		for (int i = range.start; i < range.end; i++)
		{
			//for (int i = 1; i < number; i++)
			//{
			double ratio = 1 * pow(step, i);//每次缩小√2
			cv::Mat I2;
			cv::resize(I, I2, cv::Size(int(I.cols / ratio), int(I.rows / ratio)));

			GradAndNorm(I2, angles[i], magnitudes[i]);
			ratios[i] = ratio;
			images[i] = I2.clone();
		}
	});
}

void VLDImageScale::GradAndNorm(const cv::Mat& I, cv::Mat& angle, cv::Mat& m)
{
	angle = cv::Mat::zeros(I.size(), CV_32FC1);
	m = cv::Mat::zeros(I.size(), CV_32FC1);

	for (int x = 1; x < I.cols - 1; x++)
		for (int y = 1; y < I.rows - 1; y++)
		{
			float gx = I.at<float>(y, x + 1) - I.at<float>(y, x - 1);
			float gy = I.at<float>(y + 1, x) - I.at<float>(y - 1, x);

			if (!anglefrom(gx, gy, angle.at<float>(y, x)))
				angle.at<float>(y, x) = -1;
			m.at<float>(y, x) = sqrt(gx*gx + gy * gy);
		}
}

int VLDImageScale::getIndex(const double r)const
{
	const double step = sqrt(2.0);
	//radius_size=5
	if (r <= radius_size) return 0;
	else
	{
		double range_low = radius_size;
		int index = 0;
		while (r > range_low*step)
		{
			index++;
			range_low *= step;
		}
		return std::min(int(angles.size() - 1), index);
	}
}