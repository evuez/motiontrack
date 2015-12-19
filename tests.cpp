#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>


struct AreaRange {
	int min;
	int max;
};


struct WeightedPoint {
	double x;
	double y;
	double w;

	WeightedPoint() {}
	WeightedPoint(double x, double y, double w) : x(x), y(y), w(w) {}
	WeightedPoint(const WeightedPoint &p) : x(p.x), y(p.y), w(p.w) {}

	WeightedPoint operator + (const WeightedPoint &p) const { return WeightedPoint(x + p.x, y + p.y, w + p.w); }
	WeightedPoint operator - (const WeightedPoint &p) const { return WeightedPoint(x - p.x, y - p.y, w - p.w); }
	WeightedPoint operator * (double c) const { return WeightedPoint(x * c,   y * c, w * c); }
	WeightedPoint operator / (double c) const { return WeightedPoint(x / c,   y / c, w / c); }

	WeightedPoint& operator += (const WeightedPoint &p) { x += p.x; y += p.y; w += p.w; return *this; }
	WeightedPoint& operator -= (const WeightedPoint &p) { x -= p.x; y -= p.y; w -= p.w; return *this; }
	WeightedPoint& operator *= (double c) { *this = *this * c; return *this; }
	WeightedPoint& operator /= (double c) { *this = *this / c; return *this; }
};


/**
 * Return the difference between 3 cv::Mat objects using bitwise and operator
 * @param a First incoming frame
 * @param b Second incoming frame
 * @param c Third incoming frame
 * @param dst cv::Mat representation of the difference dst
 */
void stepdiff(cv::Mat a, cv::Mat b, cv::Mat c, cv::Mat& dst) {
	cv::Mat diff_a, diff_b;

	cv::absdiff(b, a, diff_a);
	cv::absdiff(c, b, diff_b);

	cv::bitwise_and(diff_b, diff_a, dst);
}


/**
 * Create a weighted point from a given cv::Rect object
 * @param rect
 * @return WeightedPoint
 */
WeightedPoint rect2wpoint(cv::Rect rect) { // use range if weight < 1 required
	WeightedPoint w_point;

	w_point.x = (rect.tl().x + rect.br().x) / 2;
	w_point.y = (rect.tl().y + rect.br().y) / 2;
	w_point.w = (rect.br().x - rect.tl().x) + (rect.br().y - rect.tl().y);

	return w_point;
}


/**
 * Calculate a global weighted point for all given containers
 * @param containers
 * @return WeightedPoint
 * @untested
 */
WeightedPoint global_wpoint(std::vector<cv::Rect> containers) {
	WeightedPoint global_wpoint = WeightedPoint(0, 0, 0);

	for (size_t i = 0; i < containers.size(); i++)
		global_wpoint += rect2wpoint(containers[i]);

	global_wpoint /= (double) containers.size();

	return global_wpoint;
}


// give a weight to containers depending on their size and OPTIONS

/**
 * Calulate the global direction of every moving objects
 * @param containers
 * @return translate
 * @untested
 */
cv::Point direction(std::vector<WeightedPoint>& wpoints) {
	cv::Point translate = cv::Point(0, 0);

	for (size_t i = 0; i < wpoints.size(); i++) {
		if (i + 1 > wpoints.size())
			break;
		translate.x += wpoints[i].x - wpoints[i + 1].x;
		translate.y += wpoints[i].y - wpoints[i + 1].y;
	}

	if (wpoints.size() > 10) // > framerate
		wpoints.clear();

	return translate;
}


void motion_detection(cv::Mat frames[1], cv::Mat& out, AreaRange range, std::vector<WeightedPoint>& wpoints) {
	out = frames[1].clone();

	/**
	 * image diff
	 * may need to apply blur in case of noisy frames
	 */
	cv::Mat diff;
	stepdiff(frames[0], frames[1], frames[2], diff);

	/**
	 * split channels
	 */
	std::vector<cv::Mat> channels;
	cv::split(diff, channels);

	/**
	 * apply threshold to channels and combine
	 */
	cv::Mat threshed = cv::Mat::zeros(diff.size(), CV_8UC1);
	for (size_t i = 0; i < channels.size(); i++) {
		cv::Mat thresh;
		cv::threshold(channels[i], thresh, 45, 255, CV_THRESH_BINARY);
		threshed |= thresh;
	}

	/**
	 * find contours
	 */
	cv::Mat threshed_clone = threshed.clone();
	std::vector<std::vector<cv::Point> > contours;

	// use watershed before trying to detect contours
	// should greatly improve object detection
	// and avoid to split them

	cv::findContours(threshed_clone, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);

	/**
	 * select contours with enough surface
	 */
	std::vector<std::vector<cv::Point> > objects;

	for (size_t i = 0; i < contours.size(); i++) {
		double area = cv::contourArea(contours[i]);
		if (area > range.min && area < range.max) {
			objects.push_back(contours[i]);
		}
	}

	/**
	 * create display mask
	 */
	cv::Mat mask = cv::Mat::zeros(out.size(), CV_8UC3);
	//cv::drawContours(mask, objects, -1, CV_RGB(255, 0, 0), CV_FILLED);

	/**
	 * hightlight objects
	 */
	if (objects.size()) {
		// out = (out / 4 & ~mask) + (out & mask);
		cv::drawContours(out, objects, -1, CV_RGB(255, 0, 0), CV_FILLED);
		/**
		 * draw containers
		 */
		std::vector<cv::Rect> containers;
		for (size_t i = 0; i < objects.size(); i++) {
			cv::Rect b_rect = cv::boundingRect(cv::Mat(objects[i]).reshape(2));
			containers.push_back(b_rect);
			cv::rectangle(out, b_rect, CV_RGB(0, 255, 0), 4, CV_AA);
		}
		wpoints.push_back(global_wpoint(containers));
		cv::Point p = direction(wpoints);

		// check global_wpoint data
		/************/
		// std::cout << "x " << p.x << std::endl;
		// std::cout << "y " << p.y << std::endl;

		if (p.x < 0) {
			std::cout << "RIGHT" << std::endl;
		} else {
			std::cout << "LEFT" << std::endl;
		}

		WeightedPoint g_wpoint = global_wpoint(containers);

		cv::circle(out, p, g_wpoint.w, CV_RGB(0, 0, 255), 2, CV_AA);
		cv::circle(out, cv::Point(g_wpoint.x, g_wpoint.y), g_wpoint.w, CV_RGB(255, 0, 255), 2, CV_AA);
		/************/

		// check global_wpoint distance from center
		/************/
		// get frame center
		cv::Point center = cv::Point(
			out.rows / 2,
			out.cols / 2
		);
		std::cout << "DIST FROM X: " << center.x - g_wpoint.x << std::endl;
		std::cout << "DIST FROM Y: " << center.y - g_wpoint.y << std::endl;
		// add a center_on(Point& point) function
		// should return Vector(x, y) to be able to center on given point
		/************/

	}
}


int main() {
	cv::Mat frames[3];
	cv::Mat out;
	AreaRange range = {0, 20000};
	//double fps;

	cv::VideoCapture cap(0);
	cv::namedWindow("Frame");

	//fps = cap::get(CV_CAP_PROP_FPS);

	cap >> frames[0];
	frames[1] = frames[0];
	frames[2] = frames[0];


	cv::vector<WeightedPoint> wpoints;

	while(true) {
		motion_detection(frames, out, range, wpoints);

		cv::imshow("Frame", out);

		frames[0] = frames[1].clone();
		frames[1] = frames[2].clone();
		cap >> frames[2];

		if(cv::waitKey(30) >= 0)
			break;
	}
	return 0;
}
