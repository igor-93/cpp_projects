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
        static cv::Rect expand(const cv::Rect& rect, float by = 0.05);
        static void crop(cv::Mat& img, cv::Mat& cropped, float by = 0.05);

    private:
        cv::Mat origImg;
        cv::Mat processedImg;
        cv::Rect sudokuROI;
        vector<vector<cv::Rect> > smallROIs;
        vector<vector<cv::Rect> > sudokuCells;

        void processImg();
        void findSudokuGrid();

};

#endif