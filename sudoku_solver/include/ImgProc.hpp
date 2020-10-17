#ifndef IMGPROC_HPP
#define IMGPROC_HPP

#include <iostream>
#include "vector"
#include <opencv4/opencv2/core.hpp>
#include <opencv4/opencv2/core/utility.hpp>
#include <opencv4/opencv2/imgproc.hpp>
#include <opencv4/opencv2/highgui.hpp>

using namespace std;

class ImgProc{

    public:
        explicit ImgProc(const cv::Mat& img);
        void run();
        // cv::Mat getProcessedImg();
        vector<vector<cv::Rect> > getSudokuCells() const;

        static cv::Mat invertImg(const cv::Mat& input);
        static bool isSquare(const cv::Rect& r);
        static cv::Rect expand(const cv::Rect& rect, float by);
        static void crop(cv::Mat& img, cv::Mat& cropped, float by = 0.02);
        static int pointInRect(const cv::Point& p, const cv::Rect& r);
        static cv::Rect maxRect(vector<cv::Point2i>& points);
        static cv::Point2f lineIntersection(const cv::Vec2f& line1, const cv::Vec2f& line2);
        static bool isHorizontal(const cv::Vec2f& line, double degThreshold=5);
        static bool isVertical(const cv::Vec2f& line, double degThreshold=5);
        static bool isHorizontalOrVertical(const cv::Vec2f& line, double degThreshold=5);

    private:
        cv::Mat origImg;
        cv::Mat processedImg;
        vector<cv::Vec2f> houghLines;
        vector<cv::Point2f> houghIntersections;
        vector<cv::Point2i> kmeansIntersections;
        cv::Rect sudokuROI;
        vector<vector<cv::Rect> > sudokuCells;

        void processImg();
        cv::Mat houghExtraction(cv::Mat& img);
        void calcHoughIntersections();

        void locateSudokuCells();

        cv::Rect locateSudokuROI(const vector<vector<cv::Point> >& contours);
        void findSudokuGrid();

};

#endif