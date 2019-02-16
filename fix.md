修缮后的代码：

```
/**
* 图像全景拼接
@author 谢之平
*/
#include <iostream>  
#include <string>  
#include "opencv2/highgui/highgui.hpp"  
#include <opencv2/imgproc/imgproc.hpp> 
#include "opencv2/core/core.hpp"  
#include<stdlib.h>
#include <opencv2/stitching/stitcher.hpp>
#include"opencv2/nonfree/nonfree.hpp"
#include"opencv2/legacy/legacy.hpp"
#include <io.h>  

using namespace std;
using namespace cv;

typedef struct
{
	Point2f left_top;
	Point2f left_bottom;
	Point2f right_top;
	Point2f right_bottom;
}four_corners_t;

typedef struct
{
	int src_pic_num; // 图片序号
	int dst_pic_num;
	float src_pic_x; // 平均x坐标
	float dst_pic_x;
	vector<DMatch> good_match_points;
}good_macth_pair;


void OptimizeSeam(Mat &img1, Mat &trans, Mat& dst);
four_corners_t corners;
string IMAGE_PATH_PREFIX = "./image/";
bool try_use_gpu = false;
vector<Mat> imgs;
string result_name = IMAGE_PATH_PREFIX + "result.jpg";
vector<good_macth_pair> good_macth_pairs;
int book[100] = { 0 }; // 用于标记图片是否被剔除的数组
/**
* 计算图片四个角的值的方法
*/
void CalcCorners(const Mat& H, const Mat& src)
{
	double v2[] = { 0, 0, 1 };//左上角
	double v1[3];//变换后的坐标值
	Mat V2 = Mat(3, 1, CV_64FC1, v2);  //列向量
	Mat V1 = Mat(3, 1, CV_64FC1, v1);  //列向量

	V1 = H * V2;
	//左上角(0,0,1)
	cout << "\nHomography: " << H << endl;
	cout << "\n列向量V2: " << V2 << endl;
	cout << "\n列向量V1: " << V1 << endl;
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

/**
* 读取指定文件夹下所有图片的方法
*/
int readImg()
{
	cout << "\n*******开始读入图片*******" << endl;
	int i = 1;
	const char path[100] = "./test/*.jpg";
	struct _finddata_t fileinfo;
	intptr_t handle; // 这里不能用long
	handle = _findfirst(path, &fileinfo);
	if (!handle)
	{
		cout << "输入的路径有错误" << endl;
		return -1;
	}
	else {
		string name = fileinfo.name;
		cout << "\n第" << i << "张图片：" << fileinfo.name << endl;
		Mat img = imread("./test/" + name);
		// 尺寸调整  
		resize(img, img, Size(400, 300), 0, 0, INTER_LINEAR);
		imgs.push_back(img);
		i++;
		while (!_findnext(handle, &fileinfo))
		{
			cout << "\n第" << i << "张图片：" << fileinfo.name << endl;
			string name = fileinfo.name;
			Mat img = imread("./test/" + name);
			// 尺寸调整  
			resize(img, img, Size(400, 300), 0, 0, INTER_LINEAR);
			imgs.push_back(img);
			i++;
		}
	}
	if (_findclose(handle) == 0) cout << "\n文件句柄成功关闭" << endl;  // 不要忘了关闭句柄  
	else cout << "\n文件句柄关闭失败..." << endl;
	return 0;
}

/**
* 根据特征点描述结果进行图片分组，存储并显示被剔除的图片
*/
vector<set<int>> group(vector<Mat>& imagesDesc, vector<vector<KeyPoint>>& keyPoints)
{
	vector<set<int>> groups; // 存储图片分组情况
	for (int i = 0; i < imgs.size(); i++)
	{
		for (int j = i + 1; j < imgs.size(); j++)
		{
			flann::Index flannIndex(imagesDesc[i], flann::LshIndexParams(12, 20, 2), cvflann::FLANN_DIST_HAMMING);
			vector<DMatch> GoodMatchPoints;
			Mat matchIndex(imagesDesc[j].rows, 2, CV_32SC1), matchDistance(imagesDesc[j].rows, 2, CV_32FC1);
			flannIndex.knnSearch(imagesDesc[j], matchIndex, matchDistance, 2, flann::SearchParams());
			// 交换图片顺序基本不影响特征点对的计算
			// 采用Lowe's 算法选取优秀匹配点
			for (int k = 0; k < matchDistance.rows; k++)
			{
				if (matchDistance.at<float>(k, 0) < 0.6*matchDistance.at<float>(k, 1))
				{
					DMatch dmatches(k, matchIndex.at<int>(k, 0), matchDistance.at<float>(k, 0));
					GoodMatchPoints.push_back(dmatches);
				}
			}
			good_macth_pair gp; // 记录优秀匹配点对，减少重复计算
			gp.src_pic_num = i;
			gp.dst_pic_num = j;
			gp.src_pic_x = 0; // 初始化
			gp.dst_pic_x = 0;
			float src_pic_x_sum = 0;
			float dst_pic_x_sum = 0;
			gp.good_match_points = GoodMatchPoints;
			for (int k = 0; k < GoodMatchPoints.size(); k++)
			{
				DMatch dmatch = GoodMatchPoints[k];
				src_pic_x_sum += keyPoints[i][dmatch.trainIdx].pt.x;
				dst_pic_x_sum += keyPoints[j][dmatch.queryIdx].pt.x;
			}
			if (GoodMatchPoints.size() != 0){
				gp.src_pic_x = src_pic_x_sum / GoodMatchPoints.size();
				gp.dst_pic_x = dst_pic_x_sum / GoodMatchPoints.size();
			}
			good_macth_pairs.push_back(gp);
			
			
			// 优秀匹配点数量大于10的图片分为一组
			if (GoodMatchPoints.size() > 10)
			{
				int insertFlag = 0;
				for (int k = 0; k < groups.size(); k++)
				{
					if (groups[k].count(i) > 0) // 如果已为i分组
					{
						groups[k].insert(j);
						insertFlag = 1;
						break;
					}
					else if (groups[k].count(j) > 0) // 如果已为j分组
					{
						groups[k].insert(i);
						insertFlag = 1;
						break;
					}
				}
				if (insertFlag == 0) // 尚未为i或j分组，创建分组
				{
					set<int> group;
					group.insert(i);
					group.insert(j);
					groups.push_back(group);
				}
			}
		}
	}

	for (set<int> group : groups)
	{
		cout << "\n分组:\n";
		for (int pic : group)
		{
			cout << pic << " ";
			if (pic >= 0 && book[pic] == 0) {
				book[pic] = 1; // 将已经在分组中的图片标记为1，最后为0且小标小于imgs数量的图片则为被剔除的图片
			}
		}
		cout << endl;
	}
	return groups;
}

int vectorSearch(vector<int>& v, int num)
{
	vector <int>::iterator iElement = find(v.begin(),
		v.end(), num);
	if (iElement != v.end())
	{
		int nPosition = distance(v.begin(), iElement);
		return nPosition;
	}
	return -1;
}


/**
* 根据图片映射的坐标信息，对图片进行左右排序
*/
vector<vector<int>> order(vector<set<int>>& groups, vector<good_macth_pair>& good_macth_pairs)
{
	vector<vector<int>> ordered_groups;
	for (int i = 0; i < groups.size();i++) 
	{
		cout << "\n图片顺序生成中:\n";
		int size = groups[i].size();

		set<int> group = groups[i];
		map<int, int> ordered_pic;
		set<int>::iterator iter;
		for (iter = group.begin(); iter != group.end(); iter++)
		{
			int pic1_num = *iter;
			int pic2_num = -1;
			int min_big_x = imgs[pic1_num].cols;
			for (good_macth_pair gp : good_macth_pairs)
			{
				if (gp.src_pic_num == pic1_num && group.count(gp.dst_pic_num) > 0) // 第j张图与该组中的另一张图比较
				{
					if (gp.dst_pic_x > gp.src_pic_x)
					{
						if (gp.dst_pic_x < min_big_x)
						{
							pic2_num = gp.dst_pic_num;
							min_big_x = gp.dst_pic_x;
			
						}
					}
				}
				else if (gp.dst_pic_num == pic1_num &&  group.count(gp.src_pic_num) > 0)
				{
					if (gp.dst_pic_x < gp.src_pic_x)
					{
						if (gp.src_pic_x < min_big_x)
						{
							pic2_num = gp.src_pic_num;
							min_big_x = gp.src_pic_x;
						}
					}
				}
			}
			ordered_pic[pic1_num] = pic2_num;
			cout << pic1_num << " " << pic2_num << endl;
		}

		vector<int> o_group;
		cout << "\n分组排序后:\n";

		for (iter = group.begin(); iter != group.end(); iter++)
		{
			int pic = *iter;
			if (vectorSearch(o_group, pic) == -1) // 只要搜不到，就加在最前面。然后把一连串的图像序号都加进来
			{
				int index = 0;
				o_group.insert(o_group.begin(), pic);
				
				
				while (ordered_pic[pic] != -1 && vectorSearch(o_group, ordered_pic[pic]) == -1)
				{
					index += 1;
					o_group.insert(o_group.begin()+index, ordered_pic[pic]);
					pic = ordered_pic[pic];
					
				}
			}

		}
		for (int t : o_group)
		{

			cout << t << " ";
		}
		cout << endl;
		ordered_groups.push_back(o_group);
	}
	return ordered_groups;
}
/*
* 函数功能：输入两张具有重叠的图片，合成并优化生成一张图片返回
*/
Mat Stitched(Mat img1, Mat img2)
{
	vector<Mat> s_imgs;
	//便于debug
	Mat img = img1;
	s_imgs.push_back(img);
	img = img2;
	s_imgs.push_back(img);
	
	//灰度图片转换
	vector<Mat> cvt_s_imgs;
	for (int i = 0; i < s_imgs.size(); i++)
	{
		Mat temp;
		cvtColor(s_imgs[i], temp, CV_RGB2GRAY);
		cvt_s_imgs.push_back(temp);
	}

	//提取特征点
	OrbFeatureDetector orbFeatureDetector(1000);//参考opencvorb实现，调整精度，值越小，点越少越精准
	vector< vector<KeyPoint>> keyPoints;
	for (int i = 0; i < cvt_s_imgs.size(); i++)
	{
		vector<KeyPoint> keyPoint;
		orbFeatureDetector.detect(cvt_s_imgs[i], keyPoint);
		keyPoints.push_back(keyPoint);
	}

	//特征点描述，为特征点匹配做准备
	OrbDescriptorExtractor orbDescriptor;
	vector<Mat> imagesDesc;
	for (int i = 0; i < cvt_s_imgs.size(); i++)
	{
		Mat imageDesc;
		orbDescriptor.compute(cvt_s_imgs[i], keyPoints[i], imageDesc);
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
	keyPoint1 = keyPoints[0];
	keyPoint2 = keyPoints[1];
	for (int i = 0; i < GoodMatchPoints.size(); i++)
	{
		Points2f[1].push_back(keyPoint2[GoodMatchPoints[i].queryIdx].pt);
		Points2f[0].push_back(keyPoint1[GoodMatchPoints[i].trainIdx].pt);
	}
	Mat match;
	drawMatches(img1, keyPoint2, img2, keyPoint1, GoodMatchPoints, match);
	imshow("match", match);
	waitKey();
	//获取图像1到图像2 的投射矩阵，尺寸为3*3
	Mat homo = findHomography(Points2f[0], Points2f[1], CV_RANSAC);//需要legacy.hpp头文件

	//计算配准图的四个坐标顶点
	CalcCorners(homo, s_imgs[0]);

	//图像匹配
	vector<Mat> imagesTransform;
	Mat imageTransform1, imageTransform2;
	warpPerspective(s_imgs[0], imageTransform1, homo, Size(MAX(corners.right_top.x, corners.right_bottom.x), s_imgs[1].rows));
	imagesTransform.push_back(imageTransform1);
	imagesTransform.push_back(imageTransform2);
	//imshow("直接经过透视矩阵变换", imageTransform1);

	//图像拷贝
	int dst_width = imageTransform1.cols;
	int dst_height = s_imgs[1].rows;
	Mat dst(dst_height, dst_width, CV_8UC3);
	dst.setTo(0);
	imageTransform1.copyTo(dst(Rect(0, 0, imageTransform1.cols, imageTransform1.rows)));
	s_imgs[1].copyTo(dst(Rect(0, 0, s_imgs[1].cols, s_imgs[1].rows)));

	//优化拼接边缘
	OptimizeSeam(s_imgs[1], imageTransform1, dst);
	//imshow("边缘优化结果", dst);
	// waitKey();
	return dst;
}

int main()
{
	cout << CV_VERSION << endl;
	/**
	* 第一步： 从磁盘读入图片
	*/
	readImg();
	/**
	* 第二步： 预处理将图片进行灰度转换
	*/
	vector<Mat> cvt_imgs;
	for (int i = 0; i < imgs.size(); i++)
	{
		Mat temp;
		cvtColor(imgs[i], temp, CV_RGB2GRAY);
		cvt_imgs.push_back(temp);
	}
	
	/**
	* 第三步： 提取特征点
	*/
	OrbFeatureDetector orbFeatureDetector(1000); // 参考opencvorb实现，调整精度，值越小，点越少越精准
	vector<vector<KeyPoint>> keyPoints;
	for (int i = 0; i < cvt_imgs.size(); i++)
	{
		vector<KeyPoint> keyPoint;
		orbFeatureDetector.detect(cvt_imgs[i], keyPoint);
		keyPoints.push_back(keyPoint);
	}

	/**
	* 第四步：特征点描述，为特征点匹配做准备
	*/
	OrbDescriptorExtractor orbDescriptor;
	vector<Mat> imagesDesc;
	for (int i = 0; i < cvt_imgs.size(); i++)
	{
		Mat imageDesc;
		orbDescriptor.compute(cvt_imgs[i], keyPoints[i], imageDesc);
		imagesDesc.push_back(imageDesc);
	}

	/**
	* 第五步： 根据特征点将一组无序图片进行分组
	*/
	vector<set<int>> groups = group(imagesDesc, keyPoints);
	
	/**
	* 第六步：排序，按照好的匹配里的x坐标的平均值的大小来区分图片的左右顺序
	*/
	vector<vector<int>> ordered_groups = order(groups, good_macth_pairs); 
	/**
	* 第七步： 根据上面排好的顺序，依次拼接图片
	*/
	for (int i = 0; i < ordered_groups.size(); i++)
	{
		cout << "\n*******正在拼接第" << i + 1 << "幅全景图*******" << endl;
		vector<int> temp_ordered = ordered_groups[i];
		// 对一组图片进行拼接
		int temp_index = temp_ordered[0];
		Mat img1 = imgs[temp_index];
		for (int j = 1; j < temp_ordered.size(); j++)
		{
			int index = temp_ordered[j];
			img1 = Stitched(img1, imgs[index]);

		}
		cout << "\n第" << i << "幅全景图拼接完成" << endl;
		imshow("全景图", img1);
		waitKey();
		stringstream stream;
		string str;
		stream << i;
		stream >> str;
		string filename = "./result/全景图"+ str + ".jpg";
		if (i > 0) {
			imwrite(filename, img1);
		}
	}
	/**
	* 第八步：找到被提出的图片并显示、存储
	*/
	for (int i = 0; i < imgs.size(); i++) {
		if (book[i] == 0) {
			stringstream ss;
			string name;
			ss << i;
			ss >> name;
			string filename = "./result/exImg" + name + ".jpg";
			imwrite(filename, imgs[i]);
			imshow("被剔除的图片", imgs[i]);
			waitKey();
		}
	}
}

//消除并优化接缝视觉差
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

```