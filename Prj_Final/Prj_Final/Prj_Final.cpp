// Prj_Final.cpp : 定义控制台应用程序的入口点。
//

#include <iostream>  
#include <string>  
#include "opencv2/highgui/highgui.hpp"  
#include <opencv2/imgproc/imgproc.hpp> 
#include "opencv2/core/core.hpp"  
#include<stdlib.h>
#include <opencv2/stitching/stitcher.hpp>
#include<opencv2/core/core.hpp>

using namespace std;
using namespace cv;

string IMAGE_PATH_PREFIX = "./image/";

bool try_use_gpu = false;
vector<Mat> imgs;
string result_name = IMAGE_PATH_PREFIX + "result.jpg";


int main()
{
	Mat img = imread(IMAGE_PATH_PREFIX + "boat1.jpg");
	imgs.push_back(img);
	img = imread(IMAGE_PATH_PREFIX + "boat2.jpg");
	imgs.push_back(img);
	//img = imread(IMAGE_PATH_PREFIX + "boat3.jpg");
	//imgs.push_back(img);
	//img = imread(IMAGE_PATH_PREFIX + "boat3.jpg");
	//imgs.push_back(img);
	//img = imread(IMAGE_PATH_PREFIX + "boat4.jpg");
	//imgs.push_back(img);
	//img = imread(IMAGE_PATH_PREFIX + "boat5.jpg");
	//imgs.push_back(img);
	//img = imread(IMAGE_PATH_PREFIX + "boat6.jpg");
	//imgs.push_back(img);
	for (int i = 0; i <imgs.size(); i++)
	{
		imshow("picture", imgs[i]);
	}
	//灰度图片转换
	vector<Mat> cvt_imgs;
	for (int i = 0; i < imgs.size(); i++)
	{
		Mat temp;
		cvtColor(imgs[0], temp, CV_RGB2GRAY);
		cvt_imgs.push_back(temp);
	}

	//提取特征点
	OrbFeatureDetector orbFeatureDetector(1000);//参考opencvorb实现，调整精度，值越小，点越少越精准
	vector< vector<KeyPoint>> keyPoints;
	for (int i = 0; i < cvt_imgs.size(); i++)
	{
		vector<KeyPoint> keyPoint;
		orbFeatureDetector.detect(cvt_imgs[i], keyPoint);
		keyPoints.push_back(keyPoint);
	}

	//特征点描述，为特征点匹配做准备
	OrbDescriptorExtractor orbDescriptor;
	vector<Mat> imagesDesc;
	for (int i = 0; i < cvt_imgs.size(); i++)
	{
		Mat imageDesc;
		orbDescriptor.compute(cvt_imgs[i], keyPoints[i], imageDesc);
		imagesDesc.push_back(imageDesc);
	}
	flann::Index flannIndex(imagesDesc[0], flann::LshIndexParams(12, 20, 2), cvflann::FLANN_DIST_HAMMING);
	vector<DMatch> GoodMatchPoints;
	Mat matchIndex(imagesDesc[1].rows, 2, CV_32SC1), matchDistance(imagesDesc[1].rows, 2, CV_32FC1);
	flannIndex.knnSearch(imagesDesc[1], matchIndex, matchDistance, 2, flann::SearchParams());
	

	//采用Lowe's 算法选取优秀匹配点
	for (int i = 0; i < matchDistance.rows; i++)
	{
		if (matchDistance.at<float>(i, 0) < 0.6*matchDistance.at<float>(i, 1))
		{
			DMatch dmatches(i, matchIndex.at<int>(i, 0), matchDistance.at<float>(i, 0));
			GoodMatchPoints.push_back(dmatches);
		}
	}
	Mat first_match;
	drawMatches(imgs[1], keyPoints[1], imgs[0], keyPoints[0], GoodMatchPoints, first_match);
	imshow("first_match", first_match);
	string match_result = IMAGE_PATH_PREFIX + "match_result.jpg";
	imwrite(match_result, first_match);
	waitKey();
	Mat pano;//拼接结果图片
	ORB orb;
	Stitcher stitcher = Stitcher::createDefault(true);
	Stitcher::Status status = stitcher.stitch(imgs, pano);

	if (status != Stitcher::OK)
	{
		cout << "Can't stitch images, error code = " << int(status) << endl;
		return -1;
	}

	imwrite(result_name, pano);
	waitKey();
}

