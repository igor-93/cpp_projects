#include "ImgProc.hpp"

using namespace std;
using namespace cv;

// Helper functions
Mat ImgProc::invertImg(const Mat& input){
    return 255 - input;
}

Mat erosion(const Mat& inputImg){
    Mat outputImg;
	Mat element = getStructuringElement(MORPH_RECT, Size(3, 3));

	erode(inputImg, outputImg, element);
	return outputImg;
}

Mat dilation(const Mat& inputImg){
	Mat outputImg;

    int dilationSize = 1;
	Mat element = getStructuringElement(MORPH_RECT, Size(3, 3));

	dilate(inputImg, outputImg, element);
	return outputImg;
}

Mat cannyEdge(const Mat& inputImg){
	Mat outputImg, midImg;
//
//    int blurSize = 3;
//	if(blurSize % 2 == 0)
//		blurSize = blurSize + 1;
//
//	blur(inputImg, midImg, Size(blurSize, blurSize));
    double cannyThreshold = 100.0;
	Canny(inputImg, outputImg, cannyThreshold, cannyThreshold*2, 3);

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
    return false;
}

vector<Rect> findSudokuROIs_OLD(const vector<vector<Point> >& contours, const vector<Vec4i>& hierarchy){
    /**
     * Given the contours and their hierarchy, it looks for those whose ROIs make the most sense for sudoku.
     * I.e. one big rectange and at least minPlausibleChildren smaller rectangles inside
     * that are 1/3 in height and width of the big one.
     */
    vector<Rect> rects(contours.size());
    for(int i=0; i<contours.size(); i++){
        rects[i] = boundingRect(contours[i]);
    }

    int minPlausibleChildren = 4;
    vector<pair<int, Rect> > result;
    for(int i = 0; i < rects.size(); i++){
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
        } else if(hierarchy[i][2] > -1){
            child = hierarchy[hierarchy[i][2]][2];
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
    if(result.empty()){
        cout << "No Sudoku square found!" << endl;
        cout << "rects and hierarchies: " << endl;
        for(int i=0; i<rects.size(); i++){
            cout << "    rect: " << rects[i] << endl;
            cout << "    hier: " << hierarchy[i] << endl;
        }
        throw exception();
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

int pointInRect(const Point& p, const Rect& r){
    if((p.x >= r.x) && (p.y >= r.y)){
        if((p.x <= (r.x + r.width)) && (p.y <= (r.y + r.height))){
            return true;
        }
    }
    return false;
}

Rect ImgProc::findSudokuROI(const vector<vector<Point> >& contours){
    vector<Rect> rects(contours.size());
    for(int i=0; i<contours.size(); i++){
        rects[i] = boundingRect(contours[i]);
    }

    int minHits = 10;
    int nHits = 0;
    vector<pair<int, Rect> > result;
    for(auto& rect : rects){
        if(isSquare(rect)) {
            nHits = 0;
            for(auto& dot: houghIntersections){
                nHits += pointInRect(dot, rect);
            }
            if(nHits >= minHits){
                result.emplace_back(nHits, rect);
            }
        }
    }
    if(result.empty()){
        cout << "No Sudoku square found!" << endl;
        cout << "rects: " << endl;
        for(auto & rect : rects){
            cout << "    rect: " << rect << endl;
        }
        throw exception();
    }

    // sort in such a way that the first one the lowest area with most hits
    sort(result.begin(), result.end(), [](auto &left, auto &right) {
        return left.first >= right.first || (left.first == right.first && left.second.area() < right.second.area());
    });

    return result[0].second;
}

Rect ImgProc::expand(const Rect& rect, float by){
    int new_w = rect.width * (1.0 + 2.0 * by);
    int new_h = rect.height * (1.0 + 2.0 * by);

    int new_x = rect.x - by * (float)rect.width;
    int new_y = rect.y - by * (float)rect.height;

    return Rect(new_x, new_y, new_w, new_h);
}

void ImgProc::crop(Mat& img, Mat& cropped, float by){
    int new_w = (int)(img.size().width * (1.0 - 2.0 * by));
    int new_h = (int)(img.size().height * (1.0 - 2.0 * by));

    int new_x = (int)(by * (float)img.size().width);
    int new_y = (int)(by * (float)img.size().height);

    Rect roi = Rect(new_x, new_y, new_w, new_h);
    cropped = img(roi);
}

Rect maxRect(vector<Point2i>& points){
    int minX=1e6, maxX=0, minY=1e6, maxY=0;
    for(auto& p: points){
        if(p.x < minX) minX = p.x;
        if(p.y < minY) minY = p.y;
        if(p.x > maxX) maxX = p.x;
        if(p.y > maxY) maxY = p.y;
    }
    int w = maxX - minX;
    int h = maxY - minY;
    return Rect(minX, minY, w, h);
}

void ImgProc::splitSudokuROI(){
    /** 
     * The function takes sudokuROI and splits it into 3x3 and 9x9 grid by diving it into equal subsets
     */
    int cellHeight = (int)(sudokuROI.height / 9.0);
    int cellWidth = (int)(sudokuROI.width / 9.0);
    int x, y;
    for(int row=0; row<9; row++){
        for(int col=0; col<9; col++){
            x = sudokuROI.x + col * (int)(sudokuROI.width / 9.0);
            y = sudokuROI.y + row * (int)(sudokuROI.height / 9.0);

            Rect cell = Rect(x, y, cellWidth, cellHeight);
            Rect expanded = ImgProc::expand(cell, 0.2);
            vector<Point2i> intersectionsInCell;
            for(auto& dot: kmeansIntersections){
                if(pointInRect(dot, expanded)){
                    intersectionsInCell.push_back(dot);
                }
            }
            sudokuCells[row][col] = maxRect(intersectionsInCell);
        }
    }
}

// Main functions
ImgProc::ImgProc(const Mat& img){
    origImg = img.clone();
    sudokuCells = vector<vector<cv::Rect> >(9, vector<cv::Rect>(9));
}

bool isHorizontal(const Vec2f& line, double degThreshold=5){
    double deg = line[1] / CV_PI * 180.0;
    bool result = false;

    if(180-degThreshold < deg && deg < 180+degThreshold){
        result = true;
    }
    if(deg < degThreshold || deg > 360-degThreshold){
        result = true;
    }
    return result;
}

bool isVertical(const Vec2f& line, double degThreshold=5){
    double deg = line[1] / CV_PI * 180.0;
    bool result = false;

    if((90-degThreshold < deg && deg < 90+degThreshold)
       || (270-degThreshold < deg && deg < 270+degThreshold)){
        result = true;
    }
    return result;
}

bool isHorizontalOrVertical(const Vec2f& line, double degThreshold=5){
    return isHorizontal(line, degThreshold) || isVertical(line, degThreshold);
}

void drawLines(const Mat& cdst, const vector<Vec2f>& lines){
    // Draw the lines
    for(const auto & i : lines){
        float rho = i[0], theta = i[1];
        // printf("Rho = %.4f, Theta = %.4f \n", rho, theta);

        Point pt1, pt2;
        double a = cos(theta), b = sin(theta);
        double x0 = a * rho, y0 = b * rho;
        pt1.x = cvRound(x0 + 1000 * (-b));
        pt1.y = cvRound(y0 + 1000 * (a));
        pt2.x = cvRound(x0 - 1000 * (-b));
        pt2.y = cvRound(y0 - 1000 * (a));
        // B G R
        // Scalar c(0, 0, 255);
        Scalar c(255);
        line(cdst, pt1, pt2, c, 1, LINE_AA);
    }
}


Point2f lineIntersection(const Vec2f& line1, const Vec2f& line2) {
    // See https://stackoverflow.com/questions/383480/intersection-of-two-lines-defined-in-rho-theta-parameterization/383527#383527
    float rho1 = line1[0], theta1 = line1[1];
    float rho2 = line2[0], theta2 = line2[1];

    float a = cos(theta1);
    float b = sin(theta1);
    float c = cos(theta2);
    float d = sin(theta2);

    float det = 1.0 / (a * d - b * c);

    float x = det * (d * rho1 - b * rho2);
    float y = det * (-c * rho1 + a * rho2);

    return Point2f(x, y);
}

void ImgProc::calcHoughIntersections(){
    vector<Vec2f*> horizontal;
    vector<Vec2f*> vertical;
    for(auto& line: houghLines){
        if(isHorizontal(line)) horizontal.push_back(&line);
        if(isVertical(line)) vertical.push_back(&line);
    }

    houghIntersections.clear();
    for(auto& hor: horizontal){
        for(auto& ver: vertical){
            houghIntersections.push_back(lineIntersection(*hor, *ver));
        }
    }
}

Mat ImgProc::houghExtraction(Mat& img){
    Mat houghImg = Mat::zeros(img.size(), img.type());

    int smallerSize = min(img.size().height, img.size().width);
    int houghThreshold = (float)smallerSize * 0.75;

    HoughLines(img, houghLines, 1, CV_PI/180, houghThreshold, 0, 0 );
    // drop all diagonal lines
    for(auto it = houghLines.begin(); it != houghLines.end();){
        if(!isHorizontalOrVertical(*it)){
            it = houghLines.erase(it);
        } else{
            ++it;
        }
    }
    calcHoughIntersections();

    drawLines(houghImg, houghLines);
    imshow("HoughLines", houghImg);

    Mat invAndHough;
    img.copyTo(invAndHough, houghImg);
    return invAndHough;
}


void ImgProc::processImg(){
    Mat inv = invertImg(this->origImg);
    imshow("inverted img", inv);

    Mat houghImg = houghExtraction(inv);
    imshow("houghImg", houghImg);

//    Mat drawing;
//    cvtColor(houghImg, drawing, COLOR_GRAY2RGB);
//    for(Point2i& dot: kmeansIntersections){
//        cv::Scalar color(0, 0, 255);
//        circle(drawing, dot, 1, color, 2);
//    }
//    cv::namedWindow("intersections", cv::WINDOW_NORMAL | cv::WINDOW_KEEPRATIO | cv::WINDOW_GUI_EXPANDED);
//    imshow("intersections", drawing);
    waitKey(0);

    processedImg = houghImg;
}

void ImgProc::findSudokuGrid(){
    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;
    // RETR_TREE gives the whole hierarchy of contours
    findContours(processedImg, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);

    Mat origColored;
    cvtColor(origImg, origColored, COLOR_GRAY2RGB);
//    for(auto & contour : contours){
//        Rect rect = boundingRect(contour);
//        if(isSquare(rect)) {
//            cv::Scalar color(rand() * 255, rand() * 255, rand() * 255);
//            rectangle(origColored, rect, color, 2);
//        }
//    }

    this->sudokuROI = findSudokuROI(contours);

    vector<Point2f> sudokuIntersections;
    for(auto& inter: houghIntersections){
        if(pointInRect(inter, this->sudokuROI))
            sudokuIntersections.push_back(inter);
    }
    Mat bestLabels;//, centers;
    vector<Point2f> centers;
    TermCriteria criteria;
    criteria.type = TermCriteria::Type::MAX_ITER;
    criteria.maxCount = 20;
    kmeans(sudokuIntersections, 100, bestLabels, criteria, 1, KMEANS_PP_CENTERS, centers);
    for(auto& c: centers){
        kmeansIntersections.emplace_back((int) c.x, (int) c.y);
    }
        
    // B G R
    cv::Scalar colorOfBigSquare(255, 0, 0);
    rectangle(origColored, this->sudokuROI, colorOfBigSquare, 2);
    imshow("Sudoku Puzzle ROI", origColored);

    for(Point2i& dot: kmeansIntersections){
        cv::Scalar color(0, 0, 255);
        circle(origColored, dot, 1, color, 2);
    }
    cv::namedWindow("intersections", cv::WINDOW_NORMAL | cv::WINDOW_KEEPRATIO | cv::WINDOW_GUI_EXPANDED);
    imshow("intersections", origColored);

    splitSudokuROI();

    cv::Scalar colorOfCell(0, 255, 0);
    for(int i=0; i<9; i++){
        for(int j=0; j<9; j++){
            rectangle(origColored, sudokuCells[i][j], colorOfCell, 1);
        }
    }
    imshow("Sudoku cells", origColored);

    waitKey(0);

    destroyAllWindows();
}

void ImgProc::run(){
    processImg();
    findSudokuGrid();
}

vector<vector<cv::Rect> > ImgProc::getSudokuCells() const{
    return sudokuCells;
}
