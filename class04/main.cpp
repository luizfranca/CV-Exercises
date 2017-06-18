/*
 * main.cpp
 *
 *  Created on: Jun 18, 2017
 *      Author: luizdaniel
 */


#include <iostream>
#include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;

Mat src;
Mat src_gray;
int thresh = 100;
int angle;
Point2f center;
RNG rng(12345);

void thresh_callback1(int, void*) {
	Mat binary;
	vector< vector<Point> > contours;
	vector<Vec4i> hierarchy;

	threshold(src_gray, binary, thresh, 255, THRESH_BINARY);

	findContours(binary.clone(), contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE);

	Mat drawing = Mat::zeros(binary.size(), CV_8UC3);
	for (int i = 0; i < contours.size(); ++i) {
		Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));

		drawContours(drawing, contours, i, color);
	}

	imshow("Binary", binary);
	imshow("Contours", drawing);
}

void example1() {
	src = imread("Lenna.png");
	cvtColor(src, src_gray, CV_BGR2GRAY);

	namedWindow("Binary", CV_WINDOW_AUTOSIZE);

	createTrackbar("Thresh:", "Binary", &thresh, 255, thresh_callback1);

	thresh_callback1(0, 0);

	waitKey(0);
}

void example2() {
	Mat image = imread("Lenna.png");

	vector<Mat> pyramid;
	buildPyramid(image, pyramid, 4);

	for (int i = 0; i < pyramid.size(); ++i) {
		imshow("Level " + to_string(i), pyramid[i]);
	}

	waitKey();
}

void thresh_callback2(int, void*) {
	Mat rot = getRotationMatrix2D(center, angle, 1.0);

	Mat dst;

	warpAffine(src, dst, rot, src.size());

	imshow("Rotation", dst);
}

void onMouse1(int eventType, int x, int y, int, void*) {
	if (eventType == CV_EVENT_LBUTTONDOWN) {
		center.x = x;
		center.y = y;

		thresh_callback2(0, 0);
	}
}

void example3() {
	src = imread("Lenna.png");

	center.x = src.cols / 2;
	center.y = src.rows / 2;

	imshow("Source", src);

	createTrackbar("Angle:", "Source", &angle, 360, thresh_callback2);

	setMouseCallback("Source", onMouse1);

	thresh_callback2(0, 0);

	waitKey(0);
}

int main() {

//	example1();

//	example2();

	example3();

	return 0;
}

