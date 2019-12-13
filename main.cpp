
#include <iostream>
#include <fstream>
#include <algorithm>
#include <memory>
#include "kvld.h"
#include "convert.h"
#include "UVLDetector.h"

int main()
{
	cv::Mat image1, image2, image1color, image2color;
	cv::Mat concat;//for visualization
	image1 = cv::imread("...\\7.JPG", 0);
	image2 = cv::imread("...\\8.JPG", 0);
	image1color = cv::imread("...\\7.JPG", 1);
	image2color = cv::imread("...\\8.JPG", 1);

	std::vector<keypoint> F1, F2;
	std::vector<cv::KeyPoint> feat1, feat2;
	cv::Mat descriptor1, descriptor2;

	VLDImageScale vldis1;
	vldis1.createScale(image1);
	VLDImageScale vldis2;
	vldis2.createScale(image2);

	UVLDetector uvld;
	uvld.UVLcompute(image1, vldis1, feat1, descriptor1);
	uvld.UVLcompute(image2, vldis2, feat2, descriptor2);

	//cv::Mat outimg_1, outimg_2;
	//cv::Mat img1 = image1.clone();
	//cv::drawKeypoints(img1, feat1, outimg_1, cv::Scalar(255, 0, 255));
	//cv::Mat img2 = image2.clone();
	//cv::drawKeypoints(img2, feat2, outimg_2, cv::Scalar(255, 0, 255));

	std::vector<cv::DMatch> matches;
	cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::MatcherType::BRUTEFORCE);
	matcher->match(descriptor1, descriptor2, matches);
	std::cout << "sift:: match: " << matches.size() << " matches" << std::endl;

	Convert_detectors(feat1, F1);//we only need detectors for KVLD
	Convert_detectors(feat2, F2);//we only need detectors for KVLD
	std::cout << "sift:: 1st image: " << F1.size() << " keypoints" << std::endl;
	std::cout << "sift:: 2nd image: " << F2.size() << " keypoints" << std::endl;

	cv::Mat If1, If2;
	image1.convertTo(If1, CV_32FC1);
	image2.convertTo(If2, CV_32FC1);

	std::vector<Pair> matchesPair;
	Convert_matches(matches, matchesPair);
	std::cout << "K-VLD starts with " << matches.size() << " matches" << std::endl;
	std::vector<Pair> matchesFiltered;
	std::vector<double> vec_score;

	cv::Mat E = cv::Mat::ones(matches.size(), matches.size(), CV_32FC1)*(-1);
	std::vector<bool> valide(matches.size(), true);

	size_t it_num = 0;
	KvldParameters kvldparameters;//initial parameters of KVLD
	float myinlierRate = KVLD(If1, If2, vldis1, vldis2, F1, F2,
		matchesPair, matchesFiltered, vec_score, E, valide, kvldparameters);

	std::cout << "K-VLD filter ends with " << myinlierRate << " inlier Rate" << std::endl;
	std::cout << "K-VLD filter ends with " << matchesFiltered.size() << " selected matches" << std::endl;

	cv::vconcat(image1color, image2color, concat);
	for (std::vector<Pair >::const_iterator ptr = matchesFiltered.begin(); ptr != matchesFiltered.end(); ++ptr)
	{
		size_t i = ptr->first;
		size_t j = ptr->second;
		cv::KeyPoint start = feat1[i];
		cv::KeyPoint end = feat2[j];

		cv::line(concat, start.pt, end.pt + cv::Point2f(0, image1.rows), cv::Scalar(0, 255, 0), 2);
	}

	for (int it1 = 0; it1 < matchesPair.size() - 1; it1++)
	{
		for (int it2 = it1 + 1; it2 < matchesPair.size(); it2++)
		{
			if (valide[it1] && valide[it2] && E.at<float>(it1, it2) >= 0)
			{
				cv::KeyPoint l1 = feat1[matchesPair[it1].first];
				cv::KeyPoint l2 = feat1[matchesPair[it2].first];
				cv::KeyPoint r1 = feat2[matchesPair[it1].second];
				cv::KeyPoint r2 = feat2[matchesPair[it2].second];
				cv::line(concat, l1.pt, l2.pt, cv::Scalar(0, 0, 255), 2);
				cv::line(concat, r1.pt + cv::Point2f(0, image1.rows), r2.pt + cv::Point2f(0, image1.rows), cv::Scalar(0, 0, 255), 2);
			}
		}
	}


	return 0;
}