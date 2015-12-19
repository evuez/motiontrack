#include "MotionTracker.h"


using namespace std;
using namespace cv;


MotionTracker::MotionTracker(Mat& dst, Range range, int fps, int options)
	: m_dst(dst), m_range(range), m_fps(fps), m_options(m_options) {
}


void MotionTracker::m_stepdiff(Mat a, Mat b, Mat c, Mat& dst) {
	Mat diff_ab, diff_bc;

	absdiff(b, a, diff_ab);
	absdiff(c, b, diff_bc);

	bitwise_and(diff_ab, diff_bc, dst);
}


void MotionTracker::m_threshold(Mat& src_dst) {
	vector<cv::Mat> channels;
	Mat thresh;

	// split channels
	split(src_dst, channels);

	// apply threshold to channels and combine
	Mat threshed = cv::Mat::zeros(src_dst.size(), CV_8UC1);
	for (size_t i = 0; i < channels.size(); i++) {
		threshold(channels[i], thresh, 45, 255, CV_THRESH_BINARY);
		threshed |= thresh;
	}

	// modify src
	src_dst = threshed.clone();
}


void MotionTracker::stack(Mat frame) {
	if (m_frames == NULL) {
		m_frames[0] = frame;
		m_frames[1] = frame;
		m_frames[2] = frame;
	} else {
		m_frames[0] = m_frames[1].clone();
		m_frames[1] = m_frames[2].clone();
		m_frames[2] = frame;
	}
}


void MotionTracker::detect() {
	if (m_frames == NULL)
		return;

	if (&m_frames[0] == &m_frames[2])
		return;

	// do difference
	m_stepdiff(m_frames[0], m_frames[1], m_frames[2], m_dst);

	// apply threshold on all channels
	m_threshold(m_dst);

	// contours definition
	vector<std::vector<cv::Point> > contours;
	vector<vector<Point> > objects;
	vector<Point> container;

	findContours(m_dst, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);

	if ((m_options & DETECT_ALL) == DETECT_ALL) {
		objects = contours; // set container too
	} else {
		for (size_t i = 0; i < contours.size(); i++) {
			double area = cv::contourArea(contours[i]);
			if (area > m_range.start && area < m_range.end)
				objects.push_back(contours[i]);
			// set container...
		}
	}
}