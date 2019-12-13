#include "UVLDetector.h"
#include <numeric>

bool sort_by(float a, float b) { return a > b; }//�Ӵ�С����
bool sortcompetency(UVLDkeypoint a, UVLDkeypoint b)
{
	return a.competency > b.competency;
}//�Ӵ�С����


UVLDetector::UVLDetector()
{
}

UVLDetector::~UVLDetector()
{
}


void UVLDetector::UVLcompute(cv::Mat img, VLDImageScale vldimagescale,
	std::vector<cv::KeyPoint>& kpt, cv::Mat& descriptor/*,int dp*/)
{
	cv::Mat gray_img;
	if (img.channels() == 3 || img.channels() == 4)
		cvtColor(img, gray_img, cv::COLOR_BGR2GRAY);
	else
		gray_img = img.clone();
	Pnumber = img.cols*img.rows* 0.0005;//0.05%
	//Pnumber = Pnumber > 200 ? Pnumber : 200;
	//Pnumber = dp;
	kpt.clear();
	std::vector<cv::KeyPoint> initial_kpts;
	std::vector<UVLDkeypoint> ukpt;
	//detectSIFT(gray_img, initial_kpts);
	sift.create(15000, 3, 0, 10);
	sift.detectAndCompute(gray_img, Mat(), initial_kpts, descriptor);

	cv::Mat img2 = gray_img.clone(), outimg_2;
	cv::drawKeypoints(img2, initial_kpts, outimg_2, cv::Scalar(0, 255, 255));

	UVLDetector::convertKpt2Ukpt(initial_kpts, ukpt);
	std::vector<cv::KeyPoint>().swap(initial_kpts);

	//nWindow = (img.cols*0.08) > 100 ? (img.cols*0.08) : 100;

	//=============���������������ȫ����������Ϣ=======================
	//==================================================================
	std::cout << "Compute full graph..." << std::endl;
	fullGraphInfo(vldimagescale, ukpt);

	//========================�ֿ��������������================================
	//==================================================================
	std::cout << "Keypoint celled..." << std::endl;
	kptCelled(gray_img, ukpt, kpt);

	//==================������������=========================================
	//==================================================================
	sift.detectAndCompute(gray_img, Mat(), kpt, descriptor, true, true);

}

float UVLDetector::getRange(const cv::Mat& I, int a, const float p, const float ratio)
{
	float range = ratio * sqrt(float(3 * I.cols*I.rows) / (p*a*PI));
	std::cout << "range =" << range << std::endl;
	return range;
}

void UVLDetector::fullGraphInfo(VLDImageScale vldimagescale,
	std::vector<UVLDkeypoint>& ukpt)
{
	int N = ukpt.size();
	int dimension = 10;

	cv::Mat Z(N, N, CV_32FC3, cv::Scalar(-1, -1, -1));
	cv::parallel_for_(cv::Range(0, N), [&](const cv::Range& range)-> void
	{
		for (int it1 = range.start; it1 < range.end; it1++)
		{
			//for (int it1 = 0; it1 < N; it1++)
			//{
			for (int it2 = it1 + 1; it2 < N; it2++)
			{
				float dy = float(ukpt[it2].kpt.pt.y - ukpt[it1].kpt.pt.y);
				float dx = float(ukpt[it2].kpt.pt.x - ukpt[it1].kpt.pt.x);
				float dist = sqrt(dy * dy + dx * dx);
				//������ľ���̫С��̫��ʱ�����ų�
				if (dist < min_dist || dist > Sradius)
					continue;

				//������Ӧֵ**********************
				float tResponse = 0.5*(ukpt[it1].kpt.response + ukpt[it2].kpt.response);
				Z.at<cv::Vec3f>(it1, it2)[0] = tResponse;

				//�������ֵ*************************
				Z.at<cv::Vec3f>(it1, it2)[1] = dist;

				//������ֵ���������ϴ�**********************
				//Ϊ����߼���Ч�ʣ�ֻȡһ���ֵ������ֵ
				{
					float eSum = 0;
					float r = std::max(dist / float(dimension + 1), 2.0f);//at least 2
					//ȡ��Ӧͼ��
					int scale_index = vldimagescale.getIndex(r);//����Բ�Ĵ�С��ȡͼ��߶ȼ���
					double ratio = vldimagescale.ratios[scale_index];
					cv::Mat img = vldimagescale.images[scale_index];
					//cv::Mat dst;
					//cv::normalize(img, dst, 0, 255, cv::NORM_MINMAX);
					//dst.convertTo(dst, CV_8UC1);
					//���������ŵ���Ӧ�߶ȼ���
					float minx = std::min(ukpt[it1].kpt.pt.x, ukpt[it2].kpt.pt.x) / float(ratio);
					float maxx = std::max(ukpt[it1].kpt.pt.x, ukpt[it2].kpt.pt.x) / float(ratio);
					float miny = std::min(ukpt[it1].kpt.pt.y, ukpt[it2].kpt.pt.y) / float(ratio);
					float maxy = std::max(ukpt[it1].kpt.pt.y, ukpt[it2].kpt.pt.y) / float(ratio);
					minx = (minx - (r + 1)) < 0 ? 0 : (minx - (r + 1));
					maxx = (maxx + (r + 1)) > img.cols ? img.cols : (maxx + (r + 1));
					miny = (miny - (r + 1)) < 0 ? 0 : (miny - (r + 1));
					maxy = (maxy + (r + 1)) > img.rows ? img.rows : (maxy + (r + 1));

					cv::Mat imgROI = img(cv::Range(miny, maxy), cv::Range(minx, maxx));

					for (int i = 0; i < dimension; i += 3)
					{
						//��i��Բ������
						float xi = float(ukpt[it1].kpt.pt.x + float(i + 1) / (dimension + 1)*(dx));
						float yi = float(ukpt[it1].kpt.pt.y + float(i + 1) / (dimension + 1)*(dy));
						yi /= float(ratio);
						xi /= float(ratio);

						//����Mask
						cv::Mat mask = cv::Mat::zeros(imgROI.size(), CV_8UC1);
						cv::circle(mask, cv::Point(xi - minx, yi - miny), r + 1, 255, -1);
						eSum += entropy_line(imgROI, mask);
					}
					Z.at<cv::Vec3f>(it1, it2)[2] = eSum / 4;
				}
			}
		}
	});

	//ֱ�Ӽ���ÿ������걸�ԣ�
	std::vector<std::vector<float>> ptResponse(N);//����ÿ�������Ӧֵ
	std::vector<std::vector<float>> ptEntropy(N);//����ÿ�������
	std::vector<std::vector<float>> ptDistance(N);//����ÿ����ľ���
	for (int k = 0; k < N; k++)
	{
		cv::Mat Erow = Z.rowRange(k, k + 1);
		cv::Mat Ecol = Z.colRange(k, k + 1);
		for (int i = 0; i < N; i++)
		{
			if (Erow.at<cv::Vec3f>(0, i)[0] != -1)
				ptResponse[k].push_back(Erow.at<cv::Vec3f>(0, i)[0]);
			if (Erow.at<cv::Vec3f>(0, i)[1] != -1)
				ptDistance[k].push_back(Erow.at<cv::Vec3f>(0, i)[1]);
			if (Erow.at<cv::Vec3f>(0, i)[2] != -1)
				ptEntropy[k].push_back(Erow.at<cv::Vec3f>(0, i)[2]);
			if (Ecol.at<cv::Vec3f>(i, 0)[0] != -1)
				ptResponse[k].push_back(Ecol.at<cv::Vec3f>(i, 0)[0]);
			if (Ecol.at<cv::Vec3f>(i, 0)[1] != -1)
				ptDistance[k].push_back(Ecol.at<cv::Vec3f>(i, 0)[1]);
			if (Ecol.at<cv::Vec3f>(i, 0)[2] != -1)
				ptEntropy[k].push_back(Ecol.at<cv::Vec3f>(i, 0)[2]);
		}
	}
	//===========����ÿ������걸�Բ���===============================
	std::cout << "Compute competency..." << std::endl;
	//std::vector<float> competency_Dis(N);//����ÿ������걸�Է���
	std::vector<float> competency_E(N);//����ÿ������걸�Է���
	std::vector<float> competency_Res(N);//����ÿ������걸�Է���
	cv::parallel_for_(cv::Range(0, N), [&](const cv::Range& range)-> void
	{
		for (int i = range.start; i < range.end; i++)
		{
			//for (int i = 0; i < kN; i++)
			//{
			float meanResponse = 0, meanEntropy = 0;
			if (ptResponse[i].size() == 0)
			{
				std::cout << "connection size = 0" << std::endl;
				//competency_Dis[i] = 0;
				competency_E[i] = 0;
				competency_Res[i] = 0;
			}
			else if (ptResponse[i].size() < 3 && ptResponse[i].size() != 0)
			{
				double sumResponse = std::accumulate(std::begin(ptResponse[i]), std::end(ptResponse[i]), 0.0);
				meanResponse = sumResponse / ptResponse[i].size();
				double sumEntropy = std::accumulate(std::begin(ptEntropy[i]), std::end(ptEntropy[i]), 0.0);
				meanEntropy = sumEntropy / ptEntropy[i].size();
				competency_E[i] = meanEntropy;
				competency_Res[i] = meanResponse;
			}
			else
			{
				//=========ptResponse===========================
				{
					std::vector<float> response;
					std::sort(ptResponse[i].begin(), ptResponse[i].end(), sort_by);
					int RN = ptResponse[i].size()*0.3;//ȡǰ30%
					RN = RN < 3 ? 3 : RN;
					response.insert(response.end(), ptResponse[i].begin(), ptResponse[i].begin() + RN);
					double sumResponse = std::accumulate(std::begin(response), std::end(response), 0.0);
					meanResponse = sumResponse / response.size();
					if (sumResponse != sumResponse)
						std::cout << "sumResponse nan" << std::endl;

				}
				//=========ptEntropy===========================
				{
					std::vector<float> entropy;
					std::sort(ptEntropy[i].begin(), ptEntropy[i].end(), sort_by);
					int EN = ptEntropy[i].size()*0.3;//ȡǰ30%
					EN = EN < 3 ? 3 : EN;
					entropy.insert(entropy.end(), ptEntropy[i].begin(), ptEntropy[i].begin() + EN);
					double sumEntropy = std::accumulate(std::begin(entropy), std::end(entropy), 0.0);
					meanEntropy = sumEntropy / entropy.size();
					if (sumEntropy != sumEntropy)
						std::cout << "sumEntropy nan" << std::endl;
				}
				//=========ptDistance===========================
				//competency_Dis[i] = meanDistance;//��Сֵ��ƽ��ֵԽ��Խ�ã�˵��������֮�����Զ��
				competency_E[i] = meanEntropy;
				competency_Res[i] = meanResponse;
			}
		}
	});
	//����֮����� ptResponse, ptEntropy, ptDistance��������
	std::vector<std::vector<float>>().swap(ptResponse);
	std::vector<std::vector<float>>().swap(ptEntropy);
	std::vector<std::vector<float>>().swap(ptDistance);
	////����Ҫ�Ծ�����д���ʹ��������
	//int minValue = *std::min_element(competency_Dis.begin(), competency_Dis.end());
	//cv::parallel_for_(cv::Range(0, N), [&](const cv::Range& range)-> void
	//{
	//	for (int i = range.start; i < range.end; i++)
	//	{
	//		competency_Dis[i] -= minValue;
	//	}
	//});
	//========================��Ȩ�ط����걸��====================================
	//==================================================================
	std::cout << "Weighting competency..." << std::endl;
	//����ʹ��rank���ϲ����ݣ�ʹ�ø��ʵĻ���С�����6λ�����Ȼᶪʧ
	double sum_Res = std::accumulate(std::begin(competency_Res), std::end(competency_Res), 0.0);
	//double sum_Dis = std::accumulate(std::begin(competency_Dis), std::end(competency_Dis), 0.0);
	double sum_E = std::accumulate(std::begin(competency_E), std::end(competency_E), 0.0);
	cv::parallel_for_(cv::Range(0, N), [&](const cv::Range& range)-> void
	{
		for (int i = range.start; i < range.end; i++)
		{
			//��Ȩ�ط���
			float /*Wdis = 0.1, */We = 0.6;
			float C = We * (competency_E[i] / sum_E) +
				(1 - We)*(competency_Res[i] / sum_Res);
			ukpt[i].competency = C;
		}
	});
}

float UVLDetector::entropy_line(cv::Mat img, cv::Mat mask)
{
	//����ֱ��ͼͳ��ÿ�����صĸ���
	//��ֵͳ��ʱ��bin�ļ��������Ϊ1�������Ļ�һЩ�Ҷȱ仯��С��ͼ����ֵ
	//Ҳ��ܴ󣬱���ͼƬ�Ҷ�ֵȫ�����ǣ�170��171��
	int histSize = 255;	// �趨bin�ļ��Ϊ5
	 // �趨ȡֵ��Χ ( R,G,B) )
	float range[] = { 0, 255 };
	const float* histRange = { range };
	bool uniform = true;//ÿ��bin���Ǿ��ȵ�
	bool accumulate = false;//����ǰ���hist
	cv::Mat hist;
	// ����ֱ��ͼ:
	cv::calcHist(&img, 1, 0, mask, hist, 1, &histSize, &histRange, uniform, accumulate);
	//�������
	cv::Mat hist_prob = cv::Mat::zeros(hist.size(), CV_32FC1);
	double histsum = cv::sum(hist).val[0];
	double result = 0;
	if (histsum < 1)//���е㶼Ϊ255��������Ϊ0
		return result;
	else
	{
		hist_prob = hist / histsum;
		// ����ͼ����Ϣ��
		for (int i = 0; i < hist_prob.rows; i++)
		{
			float temp = hist_prob.at<float>(i, 0);
			if (temp != 0.0)
				result = result - temp * (log(temp) / log(2.0));
		}
		return result;
	}
}

void UVLDetector::kptCelled(cv::Mat gray_img, std::vector<UVLDkeypoint> ukpt,
	std::vector<cv::KeyPoint>& final_kpts)
{
	final_kpts.clear();
	int N = ukpt.size();
	std::vector<float> Ec, Cc, Nc;
	std::vector<std::vector<UVLDkeypoint>> KptCell;
	for (int i = 0; i < gray_img.rows; i += nWindow)
	{
		for (int j = 0; j < gray_img.cols; j += nWindow)
		{
			//���ָ�����Χ
			int r1 = i;
			int c1 = j;
			int r2 = (i + nWindow) > gray_img.rows ? gray_img.rows : (i + nWindow);
			int c2 = (j + nWindow) > gray_img.cols ? gray_img.cols : (j + nWindow);
			//ȡʵ�������ڵ�������
			std::vector<UVLDkeypoint> ukpt_in_Cell;
			find_kpt_in_cell(ukpt, r1, c1, r2, c2, ukpt_in_Cell);

			float CcSum = 0;
			std::for_each(std::begin(ukpt_in_Cell), std::end(ukpt_in_Cell), [&](const UVLDkeypoint d)
			{
				CcSum += d.competency;
			});
			int N_in_cell = ukpt_in_Cell.size() != 0 ? ukpt_in_Cell.size() : 1;
			Cc.push_back(CcSum);
			Nc.push_back(float(ukpt_in_Cell.size()) / float(ukpt.size()));
			cv::Mat imgROI = gray_img(cv::Range(r1, r2), cv::Range(c1, c2));
			Ec.push_back(entropy(imgROI));
			KptCell.push_back(ukpt_in_Cell);
		}
	}

	//����ÿ�������ĸ���
	double sum_Ec = std::accumulate(Ec.begin(), Ec.end(), 0.0);
	float We = 0.5, Wc = 0.3;

	float SS = 0;
	float NN = 0;
	std::vector<float> FE(Ec.size()), FC(Ec.size()), FN(Ec.size());
	for (int i = 0; i < Ec.size(); i++)
	{
		//competency����ͱ�ʾ�˸��ʣ����Բ���Ҫ�ټ������
		FE[i] = We * (Ec[i] / sum_Ec);
		FC[i] = Wc * Cc[i];
		FN[i] = (1 - We - Wc)*Nc[i];

		float Fcell = FE[i] + FC[i] + FN[i];
		SS += Fcell;
		NN += Nc[i];
		if (KptCell[i].size() < 1)continue;
		int Ncell = int(Fcell * Pnumber + 0.5);
		if (Ncell < 1)continue;
		//���յ���걸��ɸѡ
		std::vector<cv::KeyPoint> kptout;
		if (KptCell[i].size() > Ncell)
		{
			select_kpt_by_C(KptCell[i], Ncell, kptout);
		}
		else
		{
			convertUKpt2kpt(KptCell[i], kptout);
		}
		//��ɸѡ֮��Ľ������
		//std::cout << "$$$$$$$$$..." << kptout.size()<< std::endl;
		final_kpts.insert(final_kpts.end(), kptout.begin(), kptout.end());
		//ɾ����ʹ�ù��ĵ�
		erase_used_pt(ukpt, kptout);
	}
	//std::cout << "sum of F=" << SS << std::endl;
	//std::cout << "sum of C=" << NN << std::endl;

	////=====================��ʾÿһ������ĸ���================================
	//int Nrows = std::ceil(double(gray_img.rows) / double(nWindow));
	//int Ncols = std::ceil(double(gray_img.cols) / double(nWindow));
	//cv::Mat img1, img2, img3;
	//img1.create(Nrows, Ncols, CV_64FC1);
	//img2.create(Nrows, Ncols, CV_64FC1);
	//img3.create(Nrows, Ncols, CV_64FC1);
	//int fi = 0;
	//for (int i = 0; i < Nrows; i ++)
	//{
	//	for (int j = 0; j < Ncols; j ++)
	//	{
	//		img1.at<double>(i, j) = FE[fi];
	//		img2.at<double>(i, j) = FC[fi];
	//		img3.at<double>(i, j) = FN[fi];
	//		//���ָ�����Χ
	//		//int r1 = i;
	//		//int c1 = j;
	//		//int r2 = (i + nWindow) > gray_img.rows ? gray_img.rows : (i + nWindow);
	//		//int c2 = (j + nWindow) > gray_img.cols ? gray_img.cols : (j + nWindow);
	//		//���û����ı�����ز���
	//		//std::string text;
	//		//std::stringstream test;
	//		//test << FE[fi] << std::endl << FC[fi] << std::endl << FN[fi];
	//		//std::string text = test.str();
	//		//std::string text = std::to_string(/*FE[fi] + FC[fi] +*/ FN[fi]);
	//		//int font_face = cv::FONT_HERSHEY_COMPLEX;
	//		//double font_scale = 0.5;
	//		//int thickness = 1;
	//		//int baseline;
	//		//cv::Point origin;
	//		//origin.x = c1;
	//		//origin.y = r1 + (r2 - r1) / 2;
	//		//cv::putText(display_img, text, origin, font_face, font_scale,
	//		//	cv::Scalar(0, 255, 255), thickness, 8, 0);
	//		fi++;
	//	}
	//}


	//�õ��ĵ��п���С��Ԥ��ֵ
	if (final_kpts.size() < Pnumber)
	{
		int Unumber = Pnumber - final_kpts.size();
		int Snumber = Unumber < ukpt.size() ? Unumber : ukpt.size();

		std::nth_element(ukpt.begin(), ukpt.begin() + Snumber - 1,
			ukpt.end(), MKptCompetencyGreater());

		int i = 0;
		while (true)
		{
			i++;
			if (i > Snumber - 1)
				break;
			final_kpts.push_back(ukpt[i].kpt);
			if (final_kpts.size() > Pnumber)
				break;
		}
	}

	std::cout << "Detected keypoint size=" << final_kpts.size() << std::endl;
}

void UVLDetector::find_kpt_in_cell(std::vector<UVLDkeypoint> ukpt,
	int r1, int c1, int r2, int c2,
	std::vector<UVLDkeypoint>& ukpt_in_cell)
{
	ukpt_in_cell.clear();
	for (int i = 0; i < ukpt.size(); i++)
	{
		float x = ukpt[i].kpt.pt.x;
		float y = ukpt[i].kpt.pt.y;
		if ((x >= c1 && x < c2) && (y >= r1 && y < r2))
		{
			ukpt_in_cell.push_back(ukpt[i]);
		}
	}
}

float UVLDetector::entropy(cv::Mat img)
{
	float eSum = 0;
	int cN = 0;
	int histSize = 255;	// �趨bin�ļ��Ϊ5 
	float range[] = { 0, 255 };// �趨ȡֵ��Χ
	const float* histRange = { range };
	bool uniform = true;//ÿ��bin���Ǿ��ȵ�
	bool accumulate = false;//����ǰ���hist
	cv::Mat hist;
	// ����ֱ��ͼ:
	cv::calcHist(&img, 1, 0, cv::Mat(), hist, 1, &histSize, &histRange, uniform, accumulate);
	//�������
	cv::Mat hist_prob = cv::Mat::zeros(hist.size(), CV_32FC1);
	double histsum = cv::sum(hist).val[0];
	double result = 0;
	if (histsum < 1)//���е㶼Ϊ255��������Ϊ0
		return result;
	else
	{
		hist_prob = hist / histsum;
		// ����ͼ����Ϣ��
		for (int i = 0; i < hist_prob.rows; i++)
		{
			float temp = hist_prob.at<float>(i, 0);
			if (temp != 0.0)
				result = result - temp * (log(temp) / log(2.0));
		}
		return result;
	}

	//for (int i = 10; i < img.rows; i += 10)
	//{
	//	for (int j = 10; j < img.cols; j += 10)
	//	{
	//		cv::Mat mask = cv::Mat::zeros(img.size(), CV_8UC1);
	//		cv::circle(mask, cv::Point(j, i), 10, 255, -1);
	//		cv::Mat hist;
	//		// ����ֱ��ͼ:
	//		cv::calcHist(&img, 1, 0, mask, hist, 1, &histSize, &histRange, uniform, accumulate);
	//		//�������
	//		cv::Mat hist_prob = cv::Mat::zeros(hist.size(), CV_32FC1);
	//		hist_prob = hist / (cv::sum(hist).val[0]);
	//		double result = 0;
	//		// ����ͼ����Ϣ��
	//		for (int i = 0; i < hist_prob.rows; i++)
	//		{
	//			float temp = hist_prob.at<float>(i, 0);
	//			if (temp != 0.0)
	//				result = result - temp * (log(temp) / log(2.0));
	//		}
	//		eSum += result;
	//		cN++;
	//	}
	//}
	//eSum /= cN;
	//return eSum;
}

void UVLDetector::select_kpt_by_C(std::vector<UVLDkeypoint> KptCell, int Ncell,
	std::vector<cv::KeyPoint>& kptout)
{
	kptout.clear();
	std::sort(KptCell.begin(), KptCell.end(), sortcompetency);
	float ambiguous_EAD = KptCell[Ncell - 1].competency;
	std::vector<UVLDkeypoint>::const_iterator new_end =
		std::partition(KptCell.begin() + Ncell, KptCell.end(),
			CompetencyGreaterThanThreshold(ambiguous_EAD));
	KptCell.resize(new_end - KptCell.begin());
	for (int i = 0; i < KptCell.size(); i++)
		kptout.push_back(KptCell[i].kpt);
}

void UVLDetector::convertUKpt2kpt(std::vector<UVLDkeypoint> ukpt,
	std::vector<cv::KeyPoint>& kptout)
{
	kptout.clear();
	for (int i = 0; i < ukpt.size(); i++)
		kptout.push_back(ukpt[i].kpt);
}

void UVLDetector::convertKpt2Ukpt(std::vector<cv::KeyPoint> initial_kpts,
	std::vector<UVLDkeypoint>& ukpt)
{
	ukpt.clear();
	ukpt.resize(initial_kpts.size());
	cv::parallel_for_(cv::Range(0, initial_kpts.size()), [&](const cv::Range& range)-> void
	{
		for (int i = range.start; i < range.end; i++)
		{
			ukpt[i].kpt = initial_kpts[i];
		}
	});
}

void UVLDetector::competencyDist(std::vector<float> ptDist, float& stdevDistance)
{
	//�󼫲���Ϊ������ȷֲ������б�׼
	float Wscale_dist = 0.25;//����Ӧ�÷ֲ�����
	int numD[4] = { 0 };
	stdevDistance = 0;
	for (int ds = 0; ds < 4; ds++)
	{
		int minD = 0, maxD = 0;
		if (ds == 3)
		{
			minD = 10 + ds * (Sradius / 4);
			maxD = Sradius;
		}
		else
		{
			minD = 10 + ds * (Sradius / 4);
			maxD = 10 + (ds + 1)*(Sradius / 4);
		}
		for (int j = 0; j < ptDist.size(); j++)
		{
			if (ptDist[j] > minD&&ptDist[j] <= maxD)
				numD[ds]++;
		}
	}
	int maxValue = *std::max_element(numD, numD + 4);
	int minValue = *std::min_element(numD, numD + 4);

	//��Сֵ�����ֵ����Ϊ�˷����жϽ��
	stdevDistance = minValue - maxValue;
}

void UVLDetector::erase_used_pt(std::vector<UVLDkeypoint>& ukpt,
	std::vector<cv::KeyPoint> used_kpts)
{
	for (auto p1 = ukpt.begin(); p1 != ukpt.end();)
	{
		int num = 0;
		for (auto p2 = used_kpts.begin(); p2 != used_kpts.end(); p2++)
		{
			//���ꡢ�Ƕ���ͬ
			if ((p1->kpt.pt.x == p2->pt.x && p1->kpt.pt.y == p2->pt.y)
				&& p1->kpt.angle == p2->angle)
			{
				num = 1;
				break;//��ǰ��������֤����Ч��
			}
		}
		if (num == 1)
		{
			p1 = ukpt.erase(p1);
		}
		else
		{
			p1++;
		}
	}
}


void UVLDetector::KptLayered(cv::Mat gray_img, std::vector<cv::KeyPoint> initial_kpts,
	std::vector<cv::KeyPoint>& layered_kpts)
{
	layered_kpts.clear();
	float sigma = 1.6;
	cv::Mat base_img = sift.createInitialImage(gray_img, true, sigma);
	int nOctaves = cvRound(std::log((double)std::min(base_img.cols, base_img.rows)) /
		std::log(2.) - 2) - (-1);
	std::cout << "nOctaves=" << nOctaves << std::endl;

	double Zo = 0;
	for (int o = 0; o < nOctaves; o++)
	{
		Zo += std::pow(2, o);
	}
	for (int o = 0; o < nOctaves; o++)
	{
		double So = sigma * std::pow(2, o);//��*2��
		double fo = std::pow(2, nOctaves - 1) / Zo;
		double Fo = fo / std::pow(2, o);//ÿһ��ĸ���
		int No = int(Fo * Pnumber + 0.5);//����ĵ����ĿӦΪ������Ŀ�������Ǽ����Ŀ
		std::vector <cv::KeyPoint> kpt_o;
		//�ҵ��ò�����е�
		find_kpt_use_nOctave(initial_kpts, o, kpt_o);
		int N4 = 4 * No;
		int No_choose = kpt_o.size() > N4 ? N4 : kpt_o.size();
		if (No_choose > 0)//����Ӧֵ�Ĵ�Сɸѡ���ʵĵ�
			KeyPointsFilter::retainBest(kpt_o, No_choose);

		layered_kpts.insert(layered_kpts.end(), kpt_o.begin(), kpt_o.end());
	}
}

void UVLDetector::find_kpt_use_nOctave(std::vector<cv::KeyPoint> kpt_in, int o,
	std::vector<cv::KeyPoint>& kpt_out)
{
	kpt_out.clear();
	for (std::vector<cv::KeyPoint>::const_iterator it = kpt_in.begin();
		it != kpt_in.end(); it++)
	{
		int octave, layer;
		float scale;
		//*************����ע�⣬��⵽��octave�Ǵ�-1��ʼ
		sift.unpackOctave(*it, octave, layer, scale);
		if (o == (octave - (-1)))
			kpt_out.push_back(*it);
	}
}

void UVLDetector::detectSIFT(cv::Mat img, std::vector<cv::KeyPoint>& kpt_out)
{
	kpt_out.clear();
	int ngrid = 100;
	int nstep = 80;
	int kptnum = 45;//ÿ�������м��40����
	cv::Mat descriptor;
	std::vector<cv::KeyPoint> nkpt;
	sift.create(kptnum, 3, 0);
	for (int i = 0; i < img.rows; i += nstep)
	{
		for (int j = 0; j < img.cols; j += nstep)
		{
			//���ָ�����Χ
			int r1 = i;
			int c1 = j;
			int r2 = (i + ngrid) > img.rows ? img.rows : (i + ngrid);
			int c2 = (j + ngrid) > img.cols ? img.cols : (j + ngrid);
			cv::Mat imgROI = img(cv::Range(r1, r2), cv::Range(c1, c2));
			sift.detectAndCompute(imgROI, cv::Mat(), nkpt, descriptor);
			//��⵽�ĵ���Ҫת����ԭʼ����ϵ
			for (int k = 0; k < nkpt.size(); k++)
			{
				nkpt[k].pt.x += c1;
				nkpt[k].pt.y += r1;
			}
			kpt_out.insert(kpt_out.end(), nkpt.begin(), nkpt.end());
		}
	}
	//��ȡ��ͬ�ĵ�
	cv::KeyPointsFilter::removeDuplicated(kpt_out);
}
