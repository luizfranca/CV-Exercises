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

void example1() {

	Mat image = imread("Lenna.png", CV_LOAD_IMAGE_GRAYSCALE);

	Ptr<Feature2D> detector = ORB::create();

	vector<KeyPoint> keypoints;
	Mat descriptors;

	detector->detectAndCompute(image, noArray(), keypoints, descriptors);

	Mat output;
	cvtColor(image, output, CV_GRAY2RGB);

	drawKeypoints(image, keypoints, output);

	cout << descriptors << endl;

	imshow("Keypoints", output);

	waitKey(0);

}

void example2() {
	Mat trainImage = imread("box.png", CV_LOAD_IMAGE_GRAYSCALE);
	Mat queryImage = imread("box_in_scene.png", CV_LOAD_IMAGE_GRAYSCALE);

	Ptr<Feature2D> detector = ORB::create();

	vector<KeyPoint> trainKeypoints;
	Mat trainDescriptors;

	detector->detectAndCompute(trainImage, noArray(), trainKeypoints, trainDescriptors);

	vector<KeyPoint> queryKeypoints;
	Mat queryDescriptors;
	detector->detectAndCompute(queryImage, noArray(), queryKeypoints, queryDescriptors);

	BFMatcher matcher(NORM_HAMMING);

	vector<DMatch> matches;

	matcher.match(queryDescriptors, trainDescriptors, matches);

	Mat matchesImage;
	drawMatches(queryImage, queryKeypoints, trainImage, trainKeypoints, matches, matchesImage);

	imshow("matches", matchesImage);

	waitKey();

}

void example3() {

	Mat trainImage = imread("box.png", CV_LOAD_IMAGE_GRAYSCALE);
	Mat queryImage = imread("box_in_scene.png", CV_LOAD_IMAGE_GRAYSCALE);

	Ptr<Feature2D> detector = ORB::create();

	vector<KeyPoint> trainKeypoints;
	Mat trainDescriptors;

	detector->detectAndCompute(trainImage, noArray(), trainKeypoints, trainDescriptors);

	vector<KeyPoint> queryKeypoints;
	Mat queryDescriptors;
	detector->detectAndCompute(queryImage, noArray(), queryKeypoints, queryDescriptors);

	BFMatcher matcher(NORM_HAMMING);

	vector< vector<DMatch> > initialMatches;

	matcher.knnMatch(queryDescriptors, trainDescriptors, initialMatches, 2);

	vector<DMatch> finalMatches;

	for (int i = 0; i < initialMatches.size(); ++i) {
		if (initialMatches[i][0].distance / initialMatches[i][1].distance <= 0.7) {
			finalMatches.push_back(initialMatches[i][0]);
		}
	}

	Mat matchesImage;
	drawMatches(queryImage, queryKeypoints, trainImage, trainKeypoints, finalMatches, matchesImage);

	imshow("matches", matchesImage);

	waitKey();
}

Mat aux(Mat queryImage, Mat trainImage) {

	Ptr<Feature2D> detector = ORB::create();

	vector<KeyPoint> trainKeypoints;
	Mat trainDescriptors;

	detector->detectAndCompute(trainImage, noArray(), trainKeypoints, trainDescriptors);

	vector<KeyPoint> queryKeypoints;
	Mat queryDescriptors;
	detector->detectAndCompute(queryImage, noArray(), queryKeypoints, queryDescriptors);

	BFMatcher matcher(NORM_HAMMING);

	vector< vector<DMatch> > initialMatches;

	matcher.knnMatch(queryDescriptors, trainDescriptors, initialMatches, 2);

	vector<DMatch> finalMatches;

	for (int i = 0; i < initialMatches.size(); ++i) {
		if (initialMatches[i][0].distance / initialMatches[i][1].distance <= 0.7) {
			finalMatches.push_back(initialMatches[i][0]);
		}
	}

	Mat matchesImage;
	drawMatches(queryImage, queryKeypoints, trainImage, trainKeypoints, finalMatches, matchesImage);

	return matchesImage;
}

void exercise() {

	Mat trainImage = imread("train.png", CV_LOAD_IMAGE_GRAYSCALE);

	VideoCapture capture(0);

	while (true) {
		Mat frame;
		capture >> frame;

		cvtColor(frame, frame, CV_RGB2GRAY);

		Mat show = aux(frame, trainImage);

		imshow("Show", show);

		if (waitKey(1) == 27)
			break;
	}
}

int main() {

//	example1();
//	example2();
//	example3();

	exercise();

	return 0;
}
