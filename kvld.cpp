
#include "kvld.h"
#include <functional>
#include <numeric>
#include <map>

//构造函数
template<typename T>
VLD::VLD(const VLDImageScale& series, T const& P1, T const& P2) : contrast(0.0)
{
	//P1和P2是同一张图片上的两个邻域点
	//============== initializing============//
	principleAngle.fill(0);//每条线有10个圆
	descriptor.fill(0);//每条线有10X8个描述向量
	weight.fill(0);//每条线有10个主方向

	begin_point[0] = P1.x;
	begin_point[1] = P1.y;
	end_point[0] = P2.x;
	end_point[1] = P2.y;

	float dy = float(end_point[1] - begin_point[1]), dx = float(end_point[0] - begin_point[0]);
	distance = sqrt(dy*dy + dx * dx);

	if (distance == 0)
		std::cerr << "Two SIFT points have the same coordinate" << std::endl;

	float radius = std::max(distance / float(dimension + 1), 2.0f);//at least 2

	//获取两个点的连线的方向
	double mainAngle = get_orientation();//absolute angle

	int image_index = series.getIndex(radius);//根据圆的大小获取图像尺度级别

	const cv::Mat & ang = series.angles[image_index].clone();
	const cv::Mat & m = series.magnitudes[image_index].clone();
	double ratio = series.ratios[image_index];

	// std::cout<<std::endl<<"index of image "<<radius<<" "<<image_index<<" "<<ratio<<std::endl;

	int w = m.cols, h = m.rows;
	float r = float(radius / ratio);//series.radius_size;
	float sigma2 = r * r;

	//======Computing the descriptors=====//
	for (int i = 0; i < dimension; i++)
	{
		double statistic[binNum];//24
		std::fill_n(statistic, binNum, 0.0);
		//第i个圆的坐标
		float xi = float(begin_point[0] + float(i + 1) / (dimension + 1)*(dx));
		float yi = float(begin_point[1] + float(i + 1) / (dimension + 1)*(dy));
		yi /= float(ratio);//缩放到对应尺度级别
		xi /= float(ratio);
		//在以半径为r的范围内计算描述符
		for (int y = int(yi - r); y <= int(yi + r + 0.5); y++)
		{
			for (int x = int(xi - r); x <= int(xi + r + 0.5); x++)
			{
				//计算该点到圆心的距离
				float d = point_distance(xi, yi, float(x), float(y));
				//确保点在范围内
				if (d <= r && inside(w, h, x, y, 1))
				{
					//================angle and magnitude==========================//
					double angle;
					if (ang.at<float>(y, x) >= 0)//ang是梯度方向图
						angle = ang.at<float>(y, x) - mainAngle;//relative angle
					else angle = 0.0;

					//cout<<angle<<endl;
					while (angle < 0)
						angle += 2 * PI;
					while (angle >= 2 * PI)
						angle -= 2 * PI;

					//===============principle angle==============================//
					int index = int(angle*binNum / (2 * PI) + 0.5);

					double Gweight = exp(-d * d / 4.5 / sigma2)*(m.at<float>(y, x));
					// std::cout<<"in number "<<image_index<<" "<<x<<" "<<y<<" "<<m(y,x)<<std::endl;
					if (index < binNum)
						statistic[index] += Gweight;
					else // possible since the 0.5
						statistic[0] += Gweight;

					//==============the descriptor===============================//
					int index2 = int(angle*subdirection / (2 * PI) + 0.5);
					assert(index2 >= 0 && index2 <= subdirection);

					if (index2 < subdirection)
						descriptor[subdirection*i + index2] += Gweight;
					else descriptor[subdirection*i] += Gweight;// possible since the 0.5
				}
			}
		}
		//=====================find the biggest angle of ith SIFT==================//
		int index, second_index;
		//确定weight的主方向梯度值，以及index所在
		max(statistic, weight[i], binNum, index, second_index);
		principleAngle[i] = index;
	}

	normalize_weight(descriptor);

	contrast = std::accumulate(weight.begin(), weight.end(), 0.0);
	contrast /= distance / ratio;
	normalize_weight(weight);
}

float KVLD(const cv::Mat& I1, const cv::Mat& I2,
	VLDImageScale Chaine1, VLDImageScale Chaine2,
	std::vector<keypoint>& F1, std::vector<keypoint>& F2,
	const std::vector<Pair>& matches, std::vector<Pair>& matchesFiltered,
	std::vector<double>& score, cv::Mat& E,
	std::vector<bool>& valide, KvldParameters& kvldParameters)
{
	matchesFiltered.clear();
	score.clear();

	std::cout << "Image scale-space complete..." << std::endl;

	float range1 = getRange(I1, std::min(F1.size(), matches.size()), kvldParameters.inlierRate, kvldParameters.rang_ratio);
	float range2 = getRange(I2, std::min(F2.size(), matches.size()), kvldParameters.inlierRate, kvldParameters.rang_ratio);
	kvldParameters.range = range1;

	////================distance map construction, for use of selecting neighbors===============//
	//std::cout<<"computing distance maps"<<std::endl;
	//libNumerics::matrix<float> dist1=libNumerics::matrix<float>::zeros(F1.size(), F1.size());
	//libNumerics::matrix<float> dist2=libNumerics::matrix<float>::zeros(F2.size(), F2.size());
	//  for (int a1=0; a1<F1.size();++a1)
	//    for (int a2=0; a2<F1.size();++a2)
	//      dist1(a1,a2)=point_distance(F1[a1],F1[a2]);
	//  for (int b1=0; b1<F2.size();++b1)
	//    for (int b2=0; b2<F2.size();++b2)
	//      dist2(b1,b2)=point_distance(F2[b1],F2[b2]);

	size_t size = matches.size();
	//valide=size
	fill(valide.begin(), valide.end(), true);
	std::vector<double> scoretable(size, 0);
	std::vector<size_t> result(size, 0);

	//============main iteration for match verification==========//
	std::cout << "main iteration" << std::endl;
	bool change = true, initial = true;

	while (change)
	{
		std::cout << "iteration..." << std::endl;
		//std::cout << ".";
		change = false;

		fill(scoretable.begin(), scoretable.end(), 0.0);
		fill(result.begin(), result.end(), 0);
		//========substep 1: search for each match its neighbors and===============
		//verify if they are gvld-consistent 
		cv::parallel_for_(cv::Range(0, size), [&](const cv::Range& range)-> void
		{
			for (int it1 = range.start; it1 < range.end; it1++)
			{
				//for (int it1 = 0; it1 < size - 1; it1++) //按匹配数目循环
				//{
				if (valide[it1]) //valide初始值全为真
				{
					size_t a1 = matches[it1].first, b1 = matches[it1].second;
					//暴力搜索
					for (int it2 = it1 + 1; it2 < size; it2++)
					{
						if (valide[it2])
						{
							size_t a2 = matches[it2].first, b2 = matches[it2].second;
							float dist1 = point_distance(F1[a1], F1[a2]);
							float dist2 = point_distance(F2[b1], F2[b2]);
							//既大于最小邻域半径，又大于最大范围半径的点才能作为
							//符合条件的结果（同一邻域）
							if (dist1 > min_dist && dist2 > min_dist
								&& (dist1 < range1 || dist2 < range2))
							{
								if (E.at<float>(it1, it2) == -1) //update E if unknow
								{
									E.at<float>(it1, it2) = -2; E.at<float>(it2, it1) = -2;
									if (consistent(F1[a1], F1[a2], F2[b1], F2[b2]) < distance_thres)
									{
										//if (!kvldParameters.geometry || consistent(F1[a1], F1[a2], F2[b1], F2[b2]) < distance_thres)
										//{
										VLD vld1(Chaine1, F1[a1], F1[a2]);
										VLD vld2(Chaine2, F2[b1], F2[b2]);
										//vld1.test();
										//计算两条直线的描述向量的差异
										double error = vld1.difference(vld2);
										//std::cout<<std::endl<<it1<<" "<<it2<<" "<<dist1(a1,a2)<<" "<< dist2(b1,b2)<<" "<<error<<std::endl;
										//jude=0.35
										if (error < juge)
										{
											//相对值的结果是一样的
											E.at<float>(it1, it2) = (float)error;
											E.at<float>(it2, it1) = (float)error;
											//std::cout<<E(it2,it1)<<std::endl;
										}
									}
								}

								if (E.at<float>(it1, it2) >= 0)
								{
									result[it1] += 1;
									result[it2] += 1;
									scoretable[it1] += double(E.at<float>(it1, it2));
									scoretable[it2] += double(E.at<float>(it1, it2));
									//max_connection=20
									//一个顶点包含最多20个邻域点
									//if (result[it1] >= max_connection)
									//	break;
								}
							}
						}
					}
				}
			}
		});
		//========substep 2: remove false matches by K gvld-consistency criteria ============//
		for (int it = 0; it < size; it++)
		{
			//K=3，邻域点不能少于3个
			if (valide[it] && result[it] < kvldParameters.K)
			{
				valide[it] = false; change = true;
			}
		}
		//========substep 3: remove multiple matches to a same point by keeping 
		//the one with the best average gvld-consistency score ============//
		if (uniqueMatch)
		{
			for (int it1 = 0; it1 < size - 1; it1++)
			{
				if (valide[it1])
				{
					size_t a1 = matches[it1].first, b1 = matches[it1].second;

					for (int it2 = it1 + 1; it2 < size; it2++)
						if (valide[it2])
						{
							size_t a2 = matches[it2].first, b2 = matches[it2].second;

							if (a1 == a2 || b1 == b2
								|| (F1[a1].x == F1[a2].x && F1[a1].y == F1[a2].y && (F2[b1].x != F2[b2].x || F2[b1].y != F2[b2].y))
								|| ((F1[a1].x != F1[a2].x || F1[a1].y != F1[a2].y) && F2[b1].x == F2[b2].x && F2[b1].y == F2[b2].y)
								)
							{
								//cardinal comparison谁的基数大/分数高，选谁
								if (result[it1] > result[it2]) {
									valide[it2] = false; change = true;
								}
								else if (result[it1] < result[it2]) {
									valide[it1] = false; change = true;

								}
								else if (result[it1] == result[it2]) {
									//score comparison
									if (scoretable[it1] > scoretable[it2]) {
										valide[it1] = false; change = true;
									}
									else if (scoretable[it1] < scoretable[it2]) {
										valide[it2] = false; change = true;
									}
								}
							}
						}
				}
			}
		}
		//========substep 4: if geometric verification is set,===============
		//re-score matches by geometric-consistency, and remove poorly scored ones ============================//
		if (kvldParameters.geometry)
		{
			scoretable.resize(size, 0);
			std::vector<bool> switching(size, false);

			for (int it1 = 0; it1 < size; it1++)
			{
				if (valide[it1])
				{
					size_t a1 = matches[it1].first, b1 = matches[it1].second;
					float index = 0.0f;
					int good_index = 0;
					for (int it2 = 0; it2 < size; it2++)
					{
						if (it1 != it2 && valide[it2])
						{
							size_t a2 = matches[it2].first, b2 = matches[it2].second;
							float dist1 = point_distance(F1[a1], F1[a2]);
							float dist2 = point_distance(F2[b1], F2[b2]);
							if ((dist1 < range1 || dist2 < range2)
								&& (dist1 > min_dist && dist2 > min_dist))
							{
								float d = consistent(F1[a1], F1[a2], F2[b1], F2[b2]);
								scoretable[it1] += d;
								index += 1;
								if (d < distance_thres)
									good_index++;
							}
						}
					}
					scoretable[it1] /= index;
					//满足几何约束的邻域点的数目不能少于所有邻域点数目的30%
					//特征点与其所有邻域点几何差异的平均值应该小于1.2
					if (good_index<0.3f*float(index) && scoretable[it1]>1.2)
					{
						switching[it1] = true; change = true;
					}
				}
			}
			for (int it1 = 0; it1 < size; it1++)
			{
				if (switching[it1])
					valide[it1] = false;
			}
		}
	}
	//while循环结束
	std::cout << std::endl;

	//=============== generating output list ===================//
	for (int it = 0; it < size; it++)
	{
		if (valide[it])
		{
			matchesFiltered.push_back(matches[it]);
			score.push_back(scoretable[it]);
		}
	}
	return float(matchesFiltered.size()) / matches.size();
}

void writeResult(const std::string output, const std::vector<keypoint>& F1,
	const std::vector<keypoint>& F2, const std::vector<Pair>& matches,
	const std::vector<Pair>& matchesFiltered, const std::vector<double>& score)
{
	//========features
	std::ofstream feature1((output + "Detectors1.txt"));
	if (!feature1.is_open())
		std::cout << "error while writing Features1.txt" << std::endl;

	feature1 << F1.size() << std::endl;
	for (std::vector<keypoint>::const_iterator it = F1.begin(); it != F1.end(); it++) {
		writeDetector(feature1, (*it));
	}
	feature1.close();

	std::ofstream feature2((output + "Detectors2.txt"));
	if (!feature2.is_open())
		std::cout << "error while writing Features2.txt" << std::endl;
	feature2 << F2.size() << std::endl;
	for (std::vector<keypoint>::const_iterator it = F2.begin(); it != F2.end(); it++) {
		writeDetector(feature2, (*it));
	}
	feature2.close();

	//========matches
	std::ofstream initialmatches((output + "initial_matches.txt"));
	if (!initialmatches.is_open())
		std::cout << "error while writing initial_matches.txt" << std::endl;
	initialmatches << matches.size() << std::endl;
	for (std::vector<Pair>::const_iterator it = matches.begin(); it != matches.end(); it++) {
		initialmatches << it->first << " " << it->second << std::endl;

	}
	initialmatches.close();

	//==========kvld filtered matches
	std::ofstream filteredmatches((output + "kvld_matches.txt"));
	if (!filteredmatches.is_open())
		std::cout << "error while writing kvld_filtered_matches.txt" << std::endl;

	filteredmatches << matchesFiltered.size() << std::endl;
	for (std::vector<Pair>::const_iterator it = matchesFiltered.begin(); it != matchesFiltered.end(); it++) {
		filteredmatches << it->first << " " << it->second << std::endl;

	}
	filteredmatches.close();

	//====== KVLD score of matches
	std::ofstream kvldScore((output + "kvld_matches_score.txt"));
	if (!kvldScore.is_open())
		std::cout << "error while writing kvld_matches_score.txt" << std::endl;

	for (std::vector<double>::const_iterator it = score.begin(); it != score.end(); it++) {
		kvldScore << *it << std::endl;
	}
	kvldScore.close();
}
