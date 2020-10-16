#include "ImgProc.hpp"

using namespace std;
using namespace cv;

// Helper functions
Mat ImgProc::invertImg(const Mat& input){
    return 255 - input;
}

Mat erosion(const Mat& inputImg){
    Mat outputImg;
    int erosionSize = 1;
	Mat element = getStructuringElement(MORPH_RECT, Size(2 * erosionSize + 1, 2 * erosionSize + 1));

	erode(inputImg, outputImg, element);
	return outputImg;
}

Mat dilation(const Mat& inputImg){
	Mat outputImg;

    int dilationSize = 1;
	Mat element = getStructuringElement(MORPH_RECT, Size(2 * dilationSize + 1, 2 * dilationSize + 1));

	dilate(inputImg, outputImg, element);
	return outputImg;
}

Mat cannyEdge(const Mat& inputImg){
	Mat outputImg, midImg;

    int blurSize = 3;
	if(blurSize % 2 == 0)
		blurSize = blurSize + 1;

	blur(inputImg, midImg, Size(blurSize, blurSize));
    double cannyThreshold = 100.0;
	Canny(midImg, outputImg, cannyThreshold, cannyThreshold*2, 3);

	return outputImg;
}

bool isSquare(const Rect& r){
    double ratio1 = (double)r.width / (double)r.height;
    double ratio2 = (double)r.height / (double)r.width;
    bool res = (ratio1 < 1.2) || (ratio2 < 1.2);
    return res;
}

bool isSudokuSubRect(const Rect& rBig, const Rect& rSmall){
    if(isSquare(rSmall)){
        if(rSmall.width < 0.4 * rBig.width && rSmall.width > 0.27 * rBig.width){
            if(rSmall.height < 0.4 * rBig.height && rSmall.height > 0.27 * rBig.height){
                return true;
            }
        }
    }
    // cout << "not sudoku sub rect: " << rBig << ", " << rSmall << endl;
    return false;
}

vector<Rect> findSudokuROIs(const vector<vector<Point> >& contours, const vector<Vec4i>& hierarchy){
    /**
     * Given the contours and their hierarchy, it looks for those whose ROIs make the most sense for sudoku.
     * I.e. one big rectange and at least minPlausibleChildren smaller rectangles inside 
     * that are 1/3 in height and width of the big one.
     */
    vector<Rect> rects;
    for(int i = 0; i < contours.size(); i++){
        Rect rect = boundingRect(contours[i]);
        rects.push_back(rect);
    }

    int minPlausibleChildren = 4;
    vector<pair<int, Rect> > result;
    for(int i = 0; i < contours.size(); i++){
        if(!isSquare(rects[i])) continue;
        int child = hierarchy[i][2];
        int nPlausibleChildren = 0;
        while(child >= 0){
            if(isSudokuSubRect(rects[i], rects[child])){
                nPlausibleChildren++;
            }
            child = hierarchy[child][0];
        }
        if(nPlausibleChildren >= minPlausibleChildren){
            //cout << "found " << nPlausibleChildren << " plausible children for rect: " << rects[i] << endl;
            pair<int, Rect> P =make_pair(nPlausibleChildren, rects[i]);
            result.push_back(P);
        } else{
            // cout << "checking grandchildren..." << endl;
            int child = hierarchy[hierarchy[i][2]][2];
            // cout << child << endl;
            nPlausibleChildren = 0;
            while(child >= 0){
                if(isSudokuSubRect(rects[i], rects[child])){
                    nPlausibleChildren++;
                }
                child = hierarchy[child][0];
            }
            if(nPlausibleChildren >= minPlausibleChildren){
                //cout << "found " << nPlausibleChildren << " plausible grandchildren for rect: " << rects[i] << endl;
                pair<int, Rect> P =make_pair(nPlausibleChildren, rects[i]);
                result.push_back(P);
            }
        }
    }

    // sort according to number of plausible children and then extract only rects
    sort(result.begin(), result.end(), [](auto &left, auto &right) {
        return left.first < right.first;
    });
    vector<Rect> only_rect(result.size());
    for(int i=0; i < result.size(); i++){
        only_rect[i] = result[i].second;
    }

    return only_rect;
}

Rect ImgProc::expand(const Rect& rect, float by){
    int new_w = rect.width * (1.0 + 2.0 * by); 
    int new_h = rect.height * (1.0 + 2.0 * by); 

    int new_x = rect.x - by * (float)rect.width; 
    int new_y = rect.y - by * (float)rect.height; 

    return Rect(new_x, new_y, new_w, new_h);
}

void ImgProc::crop(Mat& img, Mat& cropped, float by){
    int new_w = img.size().width * (1.0 - 2.0 * by); 
    int new_h = img.size().height * (1.0 - 2.0 * by); 

    int new_x = by * (float)img.size().width; 
    int new_y = by * (float)img.size().height; 

    Rect roi = Rect(new_x, new_y, new_w, new_h);
    cropped = img(roi);
} 

int splitSudokuROI(const Rect& sudokuROI, vector<vector<Rect> >& smallROIs, vector<vector<Rect> >& cells){
    /** 
     * The function takes sudokuROI and splits it into 3x3 and 9x9 grid by diving it into equal subsets
     */
    int smallROIheight = (int)(sudokuROI.height / 3.0);
    int smallROIwidth = (int)(sudokuROI.width / 3.0);
    int cellheight = (int)(sudokuROI.height / 9.0);
    int cellwidth = (int)(sudokuROI.width / 9.0);
    for(int i_33=0; i_33<3; i_33++){
        for(int j_33=0; j_33<3; j_33++){
            int x = sudokuROI.x + i_33 * (int)(sudokuROI.width / 3.0);
            int y = sudokuROI.y + j_33 * (int)(sudokuROI.height / 3.0);
            Rect smallROI(x, y, smallROIwidth, smallROIheight);
            smallROIs[i_33][j_33] = smallROI;
            for(int i=0; i<3; i++){
                for(int j=0; j<3; j++){
                    int x = smallROI.x + i * (int)(smallROI.width / 3.0);
                    int y = smallROI.y + j * (int)(smallROI.height / 3.0);
                    Rect cell(x, y, cellwidth, cellheight);
                    cells[i_33 * 3 + i][j_33 * 3 + j] = ImgProc::expand(cell);
                }
            }
        }
    }
   
    return 0; 
}

// Main functions
ImgProc::ImgProc(const Mat& img){
    origImg = img.clone();

    smallROIs = vector<vector<cv::Rect> >(3, vector<cv::Rect>(3));
    sudokuCells = vector<vector<cv::Rect> >(9, vector<cv::Rect>(9));
}

void ImgProc::processImg(){
    Mat inv = invertImg(this->origImg);
    //imshow("invert_image", inv);
    Mat eroded = erosion(inv);
    //imshow("eroded", eroded);

    Mat cannyImg = cannyEdge(eroded);
    imshow("cannyImg", cannyImg);

    Mat dilatedImg = dilation(cannyImg);
    imshow("dilatedImg", dilatedImg);

    processedImg = dilatedImg;
}

void ImgProc::findSudokuGrid(){
    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;
    // RETR_TREE gives the whole hierarchy of contours
    findContours(processedImg, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);

    Mat origColored;
    cvtColor(origImg, origColored, COLOR_GRAY2RGB);
    for(int i = 0; i < contours.size(); i++){
        Rect rect = boundingRect(contours[i]);
        // cout << "rect: " << rect << endl;
        // cout << "hierarchy: " << hierarchy[i] << endl;
        // cout << endl;
        cv::Scalar color(rand()*255, rand()*255, rand()*255);
        rectangle(origColored, rect, color, 1);
    }
        

    vector<Rect> sudokuROIs = findSudokuROIs(contours, hierarchy);
    // best sudoku ROI is the one with most plausible children
    this->sudokuROI = sudokuROIs[sudokuROIs.size()-1];
        
    // B G R
    cv::Scalar colorOfBigSquare(255, 0, 0);
    rectangle(origColored, this->sudokuROI, colorOfBigSquare, 5);
    imshow("rectangle", origColored);

    splitSudokuROI(this->sudokuROI, this->smallROIs, this->sudokuCells);

    cv::Scalar colorOfCell(0, 255, 0);
    for(int i=0; i<9; i++){
        for(int j=0; j<9; j++){
            rectangle(origColored, sudokuCells[i][j], colorOfCell, 2);
            
        }
    }
    imshow("cells", origColored);

    waitKey(0);

    destroyAllWindows();
}

void ImgProc::run(){
    processImg();
    findSudokuGrid();
}

Mat ImgProc::getProcessedImg(){
    return processedImg.clone();
}

vector<vector<cv::Rect> > ImgProc::getSudokuCells(){
    return sudokuCells;
}
