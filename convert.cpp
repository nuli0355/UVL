

#include "convert.h"

//int Convert_image(const cv::Mat& In, Image<float> & imag)//convert only gray scale image of opencv
//{
//	unsigned char* pixelPtr = (unsigned char*)In.data;
//	imag = Image<float>(In.cols, In.rows);
//	int cn = In.channels();
//	if (cn == 1)//gray scale
//	{
//		for (int i = 0; i < In.rows; ++i)
//		{
//			for (int j = 0; j < In.cols; ++j)
//			{
//				imag(i, j) = pixelPtr[i*In.cols + j];
//			}
//		}
//	}
//	else
//	{
//		for (int i = 0; i < In.rows; ++i)
//		{
//			for (int j = 0; j < In.cols; ++j)
//			{
//				//Gray = R*0.299 + G*0.587 + B*0.114ʹ��RGBת�Ҷȹ�ʽ
//				imag(i, j) = (float(pixelPtr[(i*In.cols + j)*cn + 0]) * 29 + float(pixelPtr[(i*In.cols + j)*cn + 1]) * 150 + float(pixelPtr[(i*In.cols + j)*cn + 2]) * 77) / 255;
//			}
//		}
//
//	}
//	return 0;
//}

int Convert_detectors(const  std::vector<cv::KeyPoint>& feat1,
	std::vector<keypoint>& F1) 
{
	F1.clear();
	for (std::vector<cv::KeyPoint>::const_iterator it = feat1.begin(); it != feat1.end(); it++)
	{
		keypoint key;
		key.x = it->pt.x + 0.5;// opencv 2.4.8 mark the first pixel as (0,0) which should be (0.5,0.5)  precisely
		key.y = it->pt.y + 0.5;// opencv 2.4.8 mark the first pixel as (0,0) which should be (0.5,0.5)  precisely
		key.angle = (it->angle)*PI / 180;// opencv inverse the rotation in lower version of 2.4
		key.scale = it->size / 2;
		F1.push_back(key);
	}
	return 0;
}

int Convert_detectors(const  std::vector<keypoint>& F1, 
	std::vector<cv::KeyPoint>& feat1)
{
	feat1.clear();
	for (std::vector<keypoint>::const_iterator it = F1.begin(); it != F1.end(); it++) {
		cv::KeyPoint key;
		key.pt.x = it->x - 0.5;// opencv 2.4.8 mark the first pixel as (0,0) which should be (0.5,0.5)  precisely
		key.pt.y = it->y - 0.5;// opencv 2.4.8 mark the first pixel as (0,0) which should be (0.5,0.5)  precisely
		key.angle = (it->angle) * 180 / PI;// opencv inverse the rotation in lower version of 2.4
		key.size = it->scale * 2;
		feat1.push_back(key);
	}
	return 0;
}

int Convert_matches(const std::vector<cv::DMatch>& matches, 
	std::vector<Pair>& matchesPair)
{
	for (std::vector<cv::DMatch>::const_iterator it = matches.begin(); it != matches.end(); it++)
		matchesPair.push_back(Pair(it->queryIdx, it->trainIdx));
	return 0;
}

int Convert_pairs2matches(const std::vector<Pair>& matchesPair,
	std::vector<cv::DMatch>& matches)
{
	for (std::vector<Pair>::const_iterator it = matchesPair.begin(); it != matchesPair.end(); it++)
	{
		cv::DMatch d;
		d.queryIdx = it->first;
		d.trainIdx = it->second;
		matches.push_back(d);
	}
	return 0;
}

int read_detectors(const std::string& filename, std::vector<keypoint>& feat) {
	std::ifstream file(filename.c_str());
	if (!file.is_open()) {
		std::cout << "error in reading detector files" << std::endl;
		return -1;
	}
	int size;
	file >> size;
	for (int i = 0; i < size; i++) {

		keypoint key;
		readDetector(file, key);
		feat.push_back(key);
	}
}

int read_matches(const std::string& filename, std::vector<cv::DMatch>& matches) {
	std::ifstream file(filename.c_str());
	if (!file.is_open()) {
		std::cout << "error in reading matches files" << std::endl;
		return -1;
	}
	int size;
	file >> size;
	for (int i = 0; i < size; i++) {
		int  l, r;
		file >> l >> r;
		cv::DMatch m(l, r, 0);
		matches.push_back(m);
	}
}
