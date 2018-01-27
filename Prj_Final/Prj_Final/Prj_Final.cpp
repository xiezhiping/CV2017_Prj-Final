// Prj_Final.cpp : 定义控制台应用程序的入口点。
//

#include <iostream>  
#include <string>  
#include "opencv2/highgui/highgui.hpp"  
#include <opencv2/imgproc/imgproc.hpp> 
#include "opencv2/core/core.hpp"  
#include<stdlib.h>
#include <opencv2/stitching/stitcher.hpp>
#include"opencv2/nonfree/nonfree.hpp"
#include"opencv2/legacy/legacy.hpp"

using namespace std;
using namespace cv;
typedef struct
{
	Point2f left_top;
	Point2f left_bottom;
	Point2f right_top;
	Point2f right_bottom;
}four_corners_t;

void OptimizeSeam(Mat &img1, Mat &trans, Mat& dst);

four_corners_t corners;
string IMAGE_PATH_PREFIX = "./image/";
bool try_use_gpu = false;
vector<Mat> imgs;
string result_name = IMAGE_PATH_PREFIX + "result.jpg";

//计算图片四个角的值
void CalcCorners(const Mat& H, const Mat& src)
{
	double v2[] = { 0, 0, 1 };//左上角
	double v1[3];//变换后的坐标值
	Mat V2 = Mat(3, 1, CV_64FC1, v2);  //列向量
	Mat V1 = Mat(3, 1, CV_64FC1, v1);  //列向量

	V1 = H * V2;
	//左上角(0,0,1)
	cout << "V2: " << V2 << endl;
	cout << "V1: " << V1 << endl;
	corners.left_top.x = v1[0] / v1[2];
	corners.left_top.y = v1[1] / v1[2];

	//左下角(0,src.rows,1)
	v2[0] = 0;
	v2[1] = src.rows;
	v2[2] = 1;
	V2 = Mat(3, 1, CV_64FC1, v2);  //列向量
	V1 = Mat(3, 1, CV_64FC1, v1);  //列向量
	V1 = H * V2;
	corners.left_bottom.x = v1[0] / v1[2];
	corners.left_bottom.y = v1[1] / v1[2];

	//右上角(src.cols,0,1)
	v2[0] = src.cols;
	v2[1] = 0;
	v2[2] = 1;
	V2 = Mat(3, 1, CV_64FC1, v2);  //列向量
	V1 = Mat(3, 1, CV_64FC1, v1);  //列向量
	V1 = H * V2;
	corners.right_top.x = v1[0] / v1[2];
	corners.right_top.y = v1[1] / v1[2];

	//右下角(src.cols,src.rows,1)
	v2[0] = src.cols;
	v2[1] = src.rows;
	v2[2] = 1;
	V2 = Mat(3, 1, CV_64FC1, v2);  //列向量
	V1 = Mat(3, 1, CV_64FC1, v1);  //列向量
	V1 = H * V2;
	corners.right_bottom.x = v1[0] / v1[2];
	corners.right_bottom.y = v1[1] / v1[2];
}

int main()
{
	Mat img = imread(IMAGE_PATH_PREFIX + "orb2.jpg");
	imgs.push_back(img);
	img = imread(IMAGE_PATH_PREFIX + "orb1.jpg");
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
	//灰度图片转换
	vector<Mat> cvt_imgs;
	for (int i = 0; i < imgs.size(); i++)
	{
		Mat temp;
		cvtColor(imgs[i], temp, CV_RGB2GRAY);
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
	//将goodmatch点集进行转换
	vector<vector<Point2f>> Points2f;
	vector<Point2f> imagePoint1, imagePoint2;
	Points2f.push_back(imagePoint1);
	Points2f.push_back(imagePoint2);
	vector<KeyPoint> keyPoint1, keyPoint2;
	keyPoint1=keyPoints[0];
	keyPoint2 = keyPoints[1];
		for (int i = 0; i < GoodMatchPoints.size(); i++)
		{
			Points2f[1].push_back(keyPoint2[GoodMatchPoints[i].queryIdx].pt);
			Points2f[0].push_back(keyPoint1[GoodMatchPoints[i].trainIdx].pt);
		}
	//获取图像1到图像2 的投射矩阵，尺寸为3*3
	Mat homo = findHomography(Points2f[0], Points2f[1], CV_RANSAC);//需要legacy.hpp头文件
	cout << "变换矩阵为：\n" << homo << endl << endl;//输出映射矩阵

	//计算配准图的四个坐标顶点
	CalcCorners(homo, imgs[0]);

	//图像匹配
	vector<Mat> imagesTransform;
	Mat imageTransform1, imageTransform2;
	warpPerspective(imgs[0], imageTransform1, homo, Size(MAX(corners.right_top.x, corners.right_bottom.x), imgs[1].rows));
	imagesTransform.push_back(imageTransform1);
	imagesTransform.push_back(imageTransform2);
	imshow("直接经过透视矩阵变换", imageTransform1);
	imwrite(IMAGE_PATH_PREFIX + "orb_transform_result.jpg", imageTransform1);

	//图像拷贝
	int dst_width = imageTransform1.cols;
	int dst_height = imgs[1].rows;
	Mat dst(dst_height, dst_width, CV_8UC3);
	dst.setTo(0);
	imageTransform1.copyTo(dst(Rect(0, 0, imageTransform1.cols, imageTransform1.rows)));
	imgs[1].copyTo(dst(Rect(0, 0, imgs[1].cols, imgs[1].rows)));
	imshow("copy_dst", dst);
	imwrite(IMAGE_PATH_PREFIX + "copy_dst.jpg", dst);
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
//优化两图的连接处，使得拼接自然
void OptimizeSeam(Mat& img1, Mat& trans, Mat& dst)
{
	int start = MIN(corners.left_top.x, corners.left_bottom.x);//开始位置，即重叠区域的左边界  

	double processWidth = img1.cols - start;//重叠区域的宽度  
	int rows = dst.rows;
	int cols = img1.cols; //注意，是列数*通道数
	double alpha = 1;//img1中像素的权重  
	for (int i = 0; i < rows; i++)
	{
		uchar* p = img1.ptr<uchar>(i);  //获取第i行的首地址
		uchar* t = trans.ptr<uchar>(i);
		uchar* d = dst.ptr<uchar>(i);
		for (int j = start; j < cols; j++)
		{
			//如果遇到图像trans中无像素的黑点，则完全拷贝img1中的数据
			if (t[j * 3] == 0 && t[j * 3 + 1] == 0 && t[j * 3 + 2] == 0)
			{
				alpha = 1;
			}
			else
			{
				//img1中像素的权重，与当前处理点距重叠区域左边界的距离成正比，实验证明，这种方法确实好  
				alpha = (processWidth - (j - start)) / processWidth;
			}

			d[j * 3] = p[j * 3] * alpha + t[j * 3] * (1 - alpha);
			d[j * 3 + 1] = p[j * 3 + 1] * alpha + t[j * 3 + 1] * (1 - alpha);
			d[j * 3 + 2] = p[j * 3 + 2] * alpha + t[j * 3 + 2] * (1 - alpha);

		}
	}

}

