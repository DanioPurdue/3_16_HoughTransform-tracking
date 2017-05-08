
#if !defined FTRACKER
#define FTRACKER

#include <iostream>
#include <stdlib.h>
#include <string>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/video/background_segm.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/calib3d/calib3d.hpp>

#include "videoprocessor.h"

using namespace cv;

class FeatureTracker : public FrameProcessor {
	
	cv::Mat gray;			// current gray-level image
	cv::Mat gray_prev;		// previous gray-level image
	std::vector<cv::Point2f> points[2]; // tracked features from 0->1
	std::vector<cv::Point2f> initial;   // initial position of tracked points
	std::vector<cv::Point2f> features;  // detected features
	int max_count;	  // maximum number of features to detect
	double qlevel;    // quality level for feature detection
	double minDist;   // minimum distance between two feature points
	std::vector<uchar> status; // status of tracked features
    std::vector<float> err;    // error in tracking

	/*Variables for image processing*/
	double learningRate;
	cv::BackgroundSubtractorMOG mog;

	/*record circle center for the hough transform*/
	std::vector<cv::Point> circleCenters;
	std::vector<float> circleRadius;
	int isFirstTimeFindCircles;

	/*surf*/
  public:
	  SurfFeatureDetector detector;
	FeatureTracker() : max_count(500), qlevel(0.01), minDist(10.), isFirstTimeFindCircles(1){
		detector.hessianThreshold = 400;
	}
	
	std::vector<KeyPoint> keypoints_object, keypoints_scene;
	std::vector< DMatch > matches;
	//int findRobotFirsTime;
	
	// processing method
	void process(cv:: Mat &frame, cv:: Mat &output) {

		// convert to gray-level image
		//surf

		cv::cvtColor(frame, gray, CV_BGR2GRAY);
		Mat surfImage;
		gray.copyTo(surfImage);
		surfTracker2(gray, "newModel2Body.JPG");

		//hough transform
		frame.copyTo(output);
		removeTheDust(gray);
		//findCircle(gray, frame);      
	}
	// determine which tracked point should be accepted
	// here we keep only moving points
	bool acceptTrackedPoint(int i) {
		/*origirnal*/
		return status[i] &&
			// if point has moved
			(abs(points[0][i].x-points[1][i].x)+
			(abs(points[0][i].y-points[1][i].y))>2);
	}

	// handle the currently tracked points
	void handleTrackedPoints(cv:: Mat &frame, cv:: Mat &output) {
		// for all tracked points
		for(int i= 0; i < points[1].size(); i++ ) {
			// draw line and circle
		    cv::line(output, initial[i], points[1][i], cv::Scalar(255,0,0));
			cv::circle(output, points[1][i], 3, cv::Scalar(255,0,0),-1);
		}
	}

	// preprocess the image to make it easier for detection
	void removeTheDust(cv::Mat &frame)
	{
		
		//cv::Mat structElement1(3, 3, CV_8UC1, cv::Scalar(1));
		Mat structElement1 = getStructuringElement(CV_SHAPE_ELLIPSE, Size(12, 12));
		cv::morphologyEx(frame, frame, cv::MORPH_OPEN, structElement1);
		cv::threshold(frame, frame, 75, 230, CV_THRESH_BINARY);

	}

	//this function extracting the robot from the background
	void imageSubtraction(cv::Mat &foreground)
	{
		learningRate = 0.01;
		mog(foreground, foreground, learningRate);
		cv::threshold(foreground, foreground, 128, 255, cv::THRESH_BINARY_INV);
	}

	void showImage(cv::Mat image, char* frameTitle)
	{
		cv::namedWindow(frameTitle);
		cv::imshow(frameTitle, image);
		cv::waitKey(27);
	}

	//this function returns the absolute difference between two points
	double difference(cv::Point refPoint, cv::Point currPoint)
	{
		return abs(refPoint.x - currPoint.x) + abs(refPoint.y - currPoint.y);
	}
	

	void printCircileInformation(int x, int y, int radius)
	{
		std::cout << "+++++++++++++++++++++++++++++++" << std::endl;
		std::cout << "Radius: " << radius<< std::endl;
		std::cout << "x location: " << x << std::endl;
		std::cout << "y location: " << y << std::endl;
		return;
	}

	//applying hough transform identify the circles
	void findCircle(cv::Mat &grayImage, cv::Mat &originalImage) //the image is in grayscale
	{	
		cv::Mat outPutImage;
		int radius = 20;
		grayImage.copyTo(outPutImage);
		//++++++++++++++++++++++++++++++++
		int edgeThresh = 1;
		int lowThreshold = 75;
		int const max_lowThreshold = 100;
		int ratio = 3;
		int kernel_size = 3;
		char* window_name = "Edge Map";
		
		Canny(outPutImage, outPutImage, 50, 200, kernel_size);

		//image processing
		//cv::GaussianBlur(outPutImage, outPutImage, cv::Size(5, 5), 1.5);
		std::vector<cv::Vec3f> circles;

		/*change the set up of the hough transform*/
		cv::HoughCircles(outPutImage, circles, CV_HOUGH_GRADIENT,
			2,   // accumulator resolution (size of the image / 2) 
			20,  // minimum distance between two circles
			200, // Canny high threshold 200
			60, // minimum number of votes 
			5, outPutImage.rows / 7); // min and max radius
		//outPutImage.rows / 4

		//Draw the circles detected by using HoughCricles function
		std::vector<cv::Vec3f>::const_iterator itc = circles.begin();
		while (itc != circles.end()) {
			/*draw the outline of circles*/
			int radius = (*itc)[2];
			int x = (*itc)[0];
			int y = (*itc)[1];
			cv::circle(grayImage,
				cv::Point((*itc)[0], (*itc)[1]), // circle centre
				radius, // circle radius
				cv::Scalar(255,255,255), // color 
				3, 8, 0); // thickness
			++itc;
			printCircileInformation(x, y, radius);
			//apply this to remove the edge of the circle
			//removeTheDust(grayImage);	
		}
		showImage(outPutImage, "inside the hough function");
		return;
	}

	//surf tracker
	void surfTracker2(Mat &grayImage, String fileName)
	{

		Mat img_object = imread(fileName, CV_LOAD_IMAGE_GRAYSCALE); //read in the targe image
		Mat img_scene;
		grayImage.copyTo(img_scene); //read the entire image

		if (!img_object.data || !img_scene.data)
		{
			std::cout << " --(!) Error reading images " << std::endl;
			return;
		}

		//-- Step 1: Detect the keypoints using SURF Detector
		int minHessian = 400;

		SurfFeatureDetector detector(minHessian);

		//	std::vector<KeyPoint> keypoints_object, keypoints_scene;

		detector.detect(img_object, keypoints_object);
		detector.detect(img_scene, keypoints_scene);

		//-- Step 2: Calculate descriptors (feature vectors)
		SurfDescriptorExtractor extractor;

		Mat descriptors_object, descriptors_scene;

		extractor.compute(img_object, keypoints_object, descriptors_object);
		extractor.compute(img_scene, keypoints_scene, descriptors_scene);

		//-- Step 3: Matching descriptor vectors using FLANN matcher
		FlannBasedMatcher matcher;

		//std::vector< DMatch > matches;
		matcher.match(descriptors_object, descriptors_scene, matches);

		double max_dist = 0; double min_dist = 100;

		//-- Quick calculation of max and min distances between keypoints
		for (int i = 0; i < descriptors_object.rows; i++)
		{
			double dist = matches[i].distance;
			if (dist < min_dist) min_dist = dist;
			if (dist > max_dist) max_dist = dist;
		}

		printf("-- Max dist : %f \n", max_dist);
		printf("-- Min dist : %f \n", min_dist);

		//-- Draw only "good" matches (i.e. whose distance is less than 3*min_dist )
		std::vector< DMatch > good_matches;

		for (int i = 0; i < descriptors_object.rows; i++)
		{
			if (matches[i].distance < 3 * min_dist)
			{
				good_matches.push_back(matches[i]);
			}
		}

		Mat img_matches;
		drawMatches(img_object, keypoints_object, img_scene, keypoints_scene,
			good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
			vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

		//-- Localize the object
		std::vector<Point2f> obj;
		std::vector<Point2f> scene;

		for (int i = 0; i < good_matches.size(); i++)
		{
			//-- Get the keypoints from the good matches
			obj.push_back(keypoints_object[good_matches[i].queryIdx].pt);
			scene.push_back(keypoints_scene[good_matches[i].trainIdx].pt);
		}
		//std::cout << "object size:" << obj.size() << std::endl;
		//std::cout << "scene size:" << scene.size() << std::endl;

		if (obj.size() < 4)
			return;
		Mat H = findHomography(obj, scene, CV_RANSAC);

		//-- Get the corners from the image_1 ( the object to be "detected" )
		std::vector<Point2f> obj_corners(4);
		obj_corners[0] = cvPoint(0, 0);
		obj_corners[1] = cvPoint(img_object.cols, 0);
		obj_corners[2] = cvPoint(img_object.cols, img_object.rows);
		obj_corners[3] = cvPoint(0, img_object.rows);

		std::vector<Point2f> scene_corners(4);

		perspectiveTransform(obj_corners, scene_corners, H);

		//-- Draw lines between the corners (the mapped object in the scene - image_2 )
		//Top image
		line(img_matches, scene_corners[0] + Point2f(img_object.cols, 0), scene_corners[1] + Point2f(img_object.cols, 0), Scalar(0, 255, 0), 4);
		//Right image
		line(img_matches, scene_corners[1] + Point2f(img_object.cols, 0), scene_corners[2] + Point2f(img_object.cols, 0), Scalar(0, 255, 0), 4);
		//Bottom image
		line(img_matches, scene_corners[2] + Point2f(img_object.cols, 0), scene_corners[3] + Point2f(img_object.cols, 0), Scalar(0, 255, 0), 4);
		//Left image
		line(img_matches, scene_corners[3] + Point2f(img_object.cols, 0), scene_corners[0] + Point2f(img_object.cols, 0), Scalar(0, 255, 0), 4);

		//get center of the square
		double m1 = getSlope(scene_corners[0] , scene_corners[2]);
		double c1 = getOffset(scene_corners[0] + Point2f(img_object.cols, 0), scene_corners[2] + Point2f(img_object.cols, 0));

		double m2 = getSlope(scene_corners[1] + Point2f(img_object.cols, 0), scene_corners[3] + Point2f(img_object.cols, 0));
		double c2 = getOffset(scene_corners[1] + Point2f(img_object.cols, 0), scene_corners[3] + Point2f(img_object.cols, 0));

		double intersectX = (c2 - c1) / (m1 - m2);
		double intersectY = (m1*c1 - c2*m2) / m1 - m2;
		std::cout << "intersect X: " << intersectX << " intersect Y: " << intersectY << std::endl;
		Point2f midPoint = getMidPoint(scene_corners[0], scene_corners[2]);
		//Left image
		//line(img_matches, scene_corners[1] + Point2f(img_object.cols, 0), midPoint, Scalar(0, 255, 0), 4);


		cv::namedWindow("Good Matches & Object detection");
		//-- Show detected matches
		imshow("Good Matches & Object detection", img_matches);
		//write the image to the scene
		waitKey(27);
		//=============================Rotated Image ROI==============================
		// rect is the RotatedRect
		//Calculate Angle and size
		double lengthOfRectangle = difference(scene_corners[1], scene_corners[2]);
		double widthOfRectangle = difference(scene_corners[0], scene_corners[1]);
		double RecAngle = atan2(scene_corners[0].y - scene_corners[1].y, scene_corners[0].x - scene_corners[1].x);
		RotatedRect rect = RotatedRect(midPoint, Size2f(widthOfRectangle,lengthOfRectangle), RecAngle);

		// matrices we'll use
		Mat M, rotated, cropped;
		// get angle and size from the bounding box
		float angle = rect.angle;
		Size rect_size = rect.size;
		

		if (rect.angle < -45.) {
			angle += 90.0;
			//swap(rect_size.width, rect_size.height);
		}
		// get the rotation matrix
		M = getRotationMatrix2D(rect.center, angle + 90, 1.0);	//angle here is substracted with -60
		// perform the affine transformation
		warpAffine(img_scene, rotated, M, img_matches.size(), INTER_CUBIC);
		// crop the resulting image
		getRectSubPix(rotated, rect_size, rect.center, cropped);

		//call the template matching method
		callTemplateMatching(cropped);

		int YbodyLocation = (int)(cropped.rows / 2);
		//probeTracking(cropped, YbodyLocation);
		waitKey(27);
	}

	void probeTracking(Mat robotBody, int YbodyLocation)
	{
		Mat img_object = imread("newModelProbe2.jpg", CV_LOAD_IMAGE_GRAYSCALE); //read in the targe image
		Mat img_scene;
		robotBody.copyTo(img_scene); //read the entire image

		if (!img_object.data || !img_scene.data)
		{
			std::cout << " --(!) Error reading images " << std::endl;
			return;
		}

		//-- Step 1: Detect the keypoints using SURF Detector
		int minHessian = 400;

		SurfFeatureDetector detector(minHessian);

		//	std::vector<KeyPoint> keypoints_object, keypoints_scene;

		detector.detect(img_object, keypoints_object);
		detector.detect(img_scene, keypoints_scene);

		//-- Step 2: Calculate descriptors (feature vectors)
		SurfDescriptorExtractor extractor;

		Mat descriptors_object, descriptors_scene;

		extractor.compute(img_object, keypoints_object, descriptors_object);
		extractor.compute(img_scene, keypoints_scene, descriptors_scene);

		//-- Step 3: Matching descriptor vectors using FLANN matcher
		FlannBasedMatcher matcher;

		//std::vector< DMatch > matches;
		matcher.match(descriptors_object, descriptors_scene, matches);

		double max_dist = 0; double min_dist = 100;

		//-- Quick calculation of max and min distances between keypoints
		for (int i = 0; i < descriptors_object.rows; i++)
		{
			double dist = matches[i].distance;
			if (dist < min_dist) min_dist = dist;
			if (dist > max_dist) max_dist = dist;
		}

		printf("-- Max dist : %f \n", max_dist);
		printf("-- Min dist : %f \n", min_dist);

		//-- Draw only "good" matches (i.e. whose distance is less than 3*min_dist )
		std::vector< DMatch > good_matches;

		for (int i = 0; i < descriptors_object.rows; i++)
		{
			if (matches[i].distance < 3 * min_dist)
			{
				good_matches.push_back(matches[i]);
			}
		}

		Mat img_matches;
		drawMatches(img_object, keypoints_object, img_scene, keypoints_scene,
			good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
			vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

		//-- Localize the object
		std::vector<Point2f> obj;
		std::vector<Point2f> scene;

		for (int i = 0; i < good_matches.size(); i++)
		{
			//-- Get the keypoints from the good matches
			obj.push_back(keypoints_object[good_matches[i].queryIdx].pt);
			scene.push_back(keypoints_scene[good_matches[i].trainIdx].pt);
		}
		
		if (obj.size() < 4)
			return;
		Mat H = findHomography(obj, scene, CV_RANSAC);
	
		//-- Get the corners from the image_1 ( the object to be "detected" )
		std::vector<Point2f> obj_corners(4);
		obj_corners[0] = cvPoint(0, 0);
		obj_corners[1] = cvPoint(img_object.cols, 0);
		obj_corners[2] = cvPoint(img_object.cols, img_object.rows);
		obj_corners[3] = cvPoint(0, img_object.rows);

		std::vector<Point2f> scene_corners(4);

		perspectiveTransform(obj_corners, scene_corners, H);

		//-- Draw lines between the corners (the mapped object in the scene - image_2 )
		//Top image
		line(img_matches, scene_corners[0] + Point2f(img_object.cols, 0), scene_corners[1] + Point2f(img_object.cols, 0), Scalar(0, 255, 0), 4);
		//Right image
		line(img_matches, scene_corners[1] + Point2f(img_object.cols, 0), scene_corners[2] + Point2f(img_object.cols, 0), Scalar(0, 255, 0), 4);
		//Bottom image
		line(img_matches, scene_corners[2] + Point2f(img_object.cols, 0), scene_corners[3] + Point2f(img_object.cols, 0), Scalar(0, 255, 0), 4);
		//Left image
		line(img_matches, scene_corners[3] + Point2f(img_object.cols, 0), scene_corners[0] + Point2f(img_object.cols, 0), Scalar(0, 255, 0), 4);


		int probeYLocation = (int)findtheCenterOfProbe(scene, scene_corners[2].y, scene_corners[0].y);
		//get center of the square
		double m1 = getSlope(scene_corners[0], scene_corners[2]);
		double c1 = getOffset(scene_corners[0] + Point2f(img_object.cols, 0), scene_corners[2] + Point2f(img_object.cols, 0));

		double m2 = getSlope(scene_corners[1] + Point2f(img_object.cols, 0), scene_corners[3] + Point2f(img_object.cols, 0));
		double c2 = getOffset(scene_corners[1] + Point2f(img_object.cols, 0), scene_corners[3] + Point2f(img_object.cols, 0));

		double intersectX = (c2 - c1) / (m1 - m2);
		double intersectY = (m1*c1 - c2*m2) / m1 - m2;
		std::cout << "intersect X: " << intersectX << " intersect Y: " << intersectY << std::endl;
		Point2f midPoint = getMidPoint(scene_corners[0], scene_corners[2]);



		std::stringstream ss;
		std::stringstream ss2;
		ss << "Y Loc: " << probeYLocation;
		ss2 << "Body Loc: " << YbodyLocation;
		putText(img_matches, ss.str(), Point(3, img_matches.cols / 8 * 5), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 200, 200));
		putText(img_matches, ss2.str(), Point(3, img_matches.cols / 8 * 6.5), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 200, 200));


		cv::namedWindow("Probe Matching");
		//-- Show detected matches
		imshow("Probe Matching", img_matches);
		//write the image to the scene
		waitKey(27);
	}


	//This function returns the average y location of the probe
	double findtheCenterOfProbe(vector<Point2f> featurePoints, float upperYBound, float lowerYBound)
	{
		double sum = 0;
		int validPointsNum = 0;
		for (int i = 0; i < featurePoints.size(); i++)
		{
			if ((double)featurePoints[i].y < upperYBound && (double)featurePoints[i].y > lowerYBound)
			{
				sum += featurePoints[i].y;
				validPointsNum++;
			}
		}
		return sum / validPointsNum;
	}

	double getSlope(Point2f a, Point2f b)
	{
		double m_slope = (b.y - a.y) / (b.x - a.x);
		return m_slope;
	}


	double getOffset(Point2f a, Point2f b)
	{
		std::cout << " a.X: " << a.x << " a.Y " << a.y<< std::endl;
		std::cout << " b.X: " << b.x << " b.Y " << b.y << std::endl;

		double m_slope = (b.y - a.y) / (b.x - a.x);
		double m_offset = a.y - m_slope * (a.x);

		std::cout << " slope: " << m_slope<< std::endl;
		std::cout << " offset: " << m_offset << std::endl;

		return m_offset;
	}

	Point2f getMidPoint(Point2f a, Point2f b)
	{
		
		float x = (a.x + b.x) / 2;
		float y = (a.y + b.y) / 2;
		return Point2f(x, y);
	}

	void callTemplateMatching(Mat img)
	{
		Mat result;
		Mat templ = imread("newModelProbe2.jpg", CV_LOAD_IMAGE_GRAYSCALE); //read in the targe image
		int match_method = 0; //there are 6 different modes from 0 to 5
		matchingMethod(0, 0, img, templ, result, match_method);
	}

	void matchingMethod(int, void*, Mat img, Mat templ, Mat result, int match_method)
	{
		/// Source image to display
		Mat img_display;
		img.copyTo(img_display);

		/// Create the result matrix
		int result_cols = img.cols - templ.cols + 1;
		int result_rows = img.rows - templ.rows + 1;

		result.create(result_rows, result_cols, CV_32FC1);

		/// Do the Matching and Normalize
		matchTemplate(img, templ, result, match_method);
		normalize(result, result, 0, 1, NORM_MINMAX, -1, Mat());

		/// Localizing the best match with minMaxLoc
		double minVal; double maxVal; Point minLoc; Point maxLoc;
		Point matchLoc;

		minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc, Mat());

		/// For SQDIFF and SQDIFF_NORMED, the best matches are lower values. For all the other methods, the higher the better
		if (match_method == CV_TM_SQDIFF || match_method == CV_TM_SQDIFF_NORMED)
		{
			matchLoc = minLoc;
		}
		else
		{
			matchLoc = maxLoc;
		}

		///result image
		std::stringstream ss;
		ss << "Probe Center: " << (matchLoc.x + templ.rows) / 2;
		putText(img_display, ss.str(), Point(3, img_display.cols / 8 * 6.5), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(200, 200, 200));
		
		std::stringstream ss2;
		ss2 << "Body Center: " << (int) (img_display.rows) / 2;
		putText(img_display, ss2.str(), Point(3, img_display.cols / 8 * 7.5), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(200, 200, 200));
		
		rectangle(img_display, matchLoc, Point(matchLoc.x + templ.cols, matchLoc.y + templ.rows), Scalar::all(0), 2, 8, 0);
		imshow("Template Matching", img_display);

		return;
	}
};



#endif
