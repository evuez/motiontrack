#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>


class MotionTracker {
	private:

		cv::Mat&  m_dst;
		cv::Mat   m_frames[3];
		cv::Range m_range;
		int       m_fps;
		int       m_options;

		void m_stepdiff(cv::Mat a, cv::Mat b, cv::Mat c, cv::Mat& dst);
		void m_threshold(cv::Mat& dst);

	public:

		static const int DETECT_IN_RANGE  = 0x0001;
		static const int DETECT_OUT_RANGE = 0x0002;
		static const int DETECT_LARGEST   = 0x0004;
		static const int DETECT_SMALLEST  = 0x0008;
		static const int DETECT_ALL       = 0x000f;

		MotionTracker(cv::Mat& dst, cv::Range range, int fps, int options = DETECT_IN_RANGE | DETECT_LARGEST);

		void stack(cv::Mat frame);
		void detect();
		bool in_motion() const;
		cv::Point compute_direction() const;
};