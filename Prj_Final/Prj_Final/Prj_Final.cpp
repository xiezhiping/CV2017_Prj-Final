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
	img = imread(IMAGE_PATH_PREFIX + "boat3.jpg");
	imgs.push_back(img);
	img = imread(IMAGE_PATH_PREFIX + "boat3.jpg");
	imgs.push_back(img);
	img = imread(IMAGE_PATH_PREFIX + "boat4.jpg");
	imgs.push_back(img);
	img = imread(IMAGE_PATH_PREFIX + "boat5.jpg");
	imgs.push_back(img);
	img = imread(IMAGE_PATH_PREFIX + "boat6.jpg");
	imgs.push_back(img);

	Mat pano;//拼接结果图片
	//Stitcher stitcher = Stitcher::createDefault(try_use_gpu);
	Stitcher stitcher = Stitcher::createDefault(true);
	Stitcher::Status status = stitcher.stitch(imgs, pano);

	if (status != Stitcher::OK)
	{
		cout << "Can't stitch images, error code = " << int(status) << endl;
		return -1;
	}

	imwrite(result_name, pano);
}

