#pragma once

#include "sift.h"
#include "VLDImageScale.h"


struct UVLDkeypoint
{
	cv::KeyPoint kpt;
	float competency = 0;
};

struct CompetencyGreaterThanThreshold
{
	CompetencyGreaterThanThreshold(float _value) :
		value(_value)
	{
	}
	inline bool operator()(const UVLDkeypoint& kpt_with_E) const
	{
		return kpt_with_E.competency >= value;
	}
	float value;
};

struct MKptCompetencyGreater
{
	inline bool operator()(const UVLDkeypoint& kp1, const UVLDkeypoint& kp2) const
	{
		return kp1.competency > kp2.competency;
	}
};

class UVLDetector
{
public:
	UVLDetector();
	~UVLDetector();

	//=================主函数入口==========================
	void UVLcompute(cv::Mat img, VLDImageScale vldimagescale,
		std::vector<cv::KeyPoint>& kpt, cv::Mat& descriptor/*,int dp*/);

	//KVLD，获取最大邻域范围
	float getRange(const cv::Mat& I, int a, const float p, const float ratio);

	void fullGraphInfo(VLDImageScale vldimagescale, std::vector<UVLDkeypoint>& ukpt);

	void kptCelled(cv::Mat gray_img, std::vector<UVLDkeypoint> ukpt,
		std::vector<cv::KeyPoint>& final_kpts);

	void find_kpt_in_cell(std::vector<UVLDkeypoint> ukpt,
		int r1, int c1, int r2, int c2,
		std::vector<UVLDkeypoint>& ukpt_in_cell);

	void convertKpt2Ukpt(std::vector<cv::KeyPoint> initial_kpts,
		std::vector<UVLDkeypoint>& ukpt);

	void convertUKpt2kpt(std::vector<UVLDkeypoint> ukpt,
		std::vector<cv::KeyPoint>& kptout);

	float entropy(cv::Mat img);

	float entropy_line(cv::Mat img, cv::Mat mask);

	void select_kpt_by_C(std::vector<UVLDkeypoint> KptCell, int Ncell, 
		std::vector<cv::KeyPoint>& kptout);

	void competencyDist(std::vector<float> ptDist, float& stdevDistance);

	void erase_used_pt(std::vector<UVLDkeypoint>& ukpt,
		std::vector<cv::KeyPoint> used_kpts);

	void KptLayered(cv::Mat gray_img, std::vector<cv::KeyPoint> initial_kpts,
		std::vector<cv::KeyPoint>& layered_kpts);

	void find_kpt_use_nOctave(std::vector<cv::KeyPoint> kpt_in, int o,
		std::vector<cv::KeyPoint>& kpt_out);

	void detectSIFT(cv::Mat img, std::vector<cv::KeyPoint>& kpt_out);

	const int nWindow = 100;//格网大小
	const int Sradius = 200;//因为初始检测的点的数量特别稀少，因此人为设定搜索半径
private:
	const float PI = 4.0 * atan(1.0f);//3.1415926
	int Pnumber = 0;//检测点的数目
	SIFT_Impl sift;
	int min_dist = 10;
};

