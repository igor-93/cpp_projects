#include <iostream>
#include <string>

#include <opencv4/opencv2/core.hpp>
#include <opencv4/opencv2/imgproc.hpp>
#include <opencv4/opencv2/highgui.hpp>

#include "Sudoku.hpp"
#include "MnistModel.hpp"
#include "ImgProc.hpp"

using namespace cv;
using namespace std;

void testSudoku(){
    int o = UNASSIGNED;
    vector<vector<int> > grid = { 
        { 3, o, 6, 5, o, 8, 4, o, o }, 
        { 5, 2, o, o, o, o, o, o, o }, 
        { o, 8, 7, o, o, o, o, 3, 1 }, 
        { o, o, 3, o, 1, o, o, 8, o }, 
        { 9, o, o, 8, 6, 3, o, o, 5 }, 
        { o, 5, o, o, 9, o, 6, o, o }, 
        { 1, 3, o, o, o, o, 2, 5, o }, 
        { o, o, o, o, o, o, o, 7, 4 }, 
        { o, o, 5, 2, o, 6, 3, o, o } 
    }; 

    Sudoku s(grid);
    s.print();
    bool solved = s.solve();
    cout << "Solved: " << solved << endl;
    s.print();
}


void testMnist(){
    MnistModel& m = MnistModel::getInstance();
    m.testLibTorch();
}

void removeEdges(Mat& img, Mat& binaryImg){
    binaryImg = img.clone();
    // flood-fill from the edges
    for(int i=0;i<img.size().height; i++){
        if(i == 0 || i == img.size().height-1) {
            for (int j = 0; j < img.size().width; j++) {
                floodFill(binaryImg, cv::Point(j, i), Scalar(0));
            }
        } else{
            floodFill(binaryImg, cv::Point(0, i), Scalar(0));
            floodFill(binaryImg, cv::Point(img.size().width - 1, i), Scalar(0));
        }
    }
} 

void centerAndCrop(Mat& img, Mat& centered){
    Point2i center(0,0);
    float cnt = 0;

    for(int i=0;i<img.size().width; i++){
        for(int j=0;j<img.size().height; j++){
            int val = img.at<uchar>(i, j);
            if(val> 0){
                center.x += j;
                center.y += i;
                cnt++;
            }
        }   
    }
    center.x = (float)(center.x) / cnt;
    center.y = (float)(center.y) / cnt;

    int w = min(min(img.size().width, center.x * 2), (int)((img.size().width - center.x) * 2.0));
    int h = min(min(img.size().height, center.y * 2), (int)((img.size().height - center.y) * 2.0));
    Rect centerRect = Rect(center.x-w/2, center.y-h/2, w, h);
    centered = img(centerRect);
    //ImgProc::crop(centered, centered, 0.01);
} 


void drawResult(const Mat& img, Mat& drawing, const vector<vector<Rect> >& cells, const Sudoku& digits){
    cvtColor(img, drawing, COLOR_GRAY2RGB);

    Scalar blue(255, 0, 0);
    Scalar red(0, 0, 255);
    Scalar green(0, 255, 0);

    for (int row=0; row<Sudoku::N; row++) { 
        for (int col=0; col<Sudoku::N; col++){
            Point2i org(cells[row][col].x + cells[row][col].width * 0.2, cells[row][col].y + cells[row][col].height / 2.0);
            string digitText = to_string(digits.getValue(row, col));
            Scalar color;
            if(digits.getProb(row, col) == UNASSIGNED){
                color = blue;
            } else if(digits.getProb(row, col) > MnistModel::acceptanceThreshold){
                color = green;
            } else{
                color = red;
            }
            putText(drawing, digitText, org, FONT_HERSHEY_PLAIN, 1, color, 2);
        }
    }
}


int main(int argc, char *argv[]){
    /**
     * Tasks:
     *  
     *  1. find the grid
     *     DONE
     *  2. detect the digits
     *     DONE
     *  3. implement the solver with depth-first search
     *     DONE
     *  4. connect the computer camera for real-time detection
     */
    if(argc < 2){
        cout << "Please provide Sudoku image to solve." << endl;
        return -1;
    }
    Mat img = imread(argv[1], IMREAD_GRAYSCALE);
    cout << "Image has size " << img.size() << ", with " << img.channels() << " channels." << endl;

    cout << "Instantiating processor and neural-net..." << endl;
    ImgProc processor(img);
    MnistModel& model = MnistModel::getInstance();
    // model.trainModel();

    cout << "Running image processor...";
    processor.run();
    cout << " done." << endl;

    vector<vector<cv::Rect> > sudokuGrid = processor.getSudokuCells();

    vector<Sudoku> possibleGames = vector<Sudoku>();
    possibleGames.emplace_back();

    cout << "Extracting digits from the Sudoku cells..." << endl;
    for(int i=0; i<Sudoku::N; i++){
        for(int j=0; j<Sudoku::N; j++){
            Rect cellROI = sudokuGrid[i][j];
            Mat cellImg = img(cellROI);

            Mat digit = ImgProc::invertImg(cellImg);
//            cv::imshow("digit", digit);

            // work on binary image
            Mat binaryImg;
            threshold(digit, binaryImg, mean(digit)[0], 255, THRESH_BINARY);
//            cv::imshow("binaryImg", binaryImg);

            Mat clean;
            removeEdges(binaryImg, clean);
//            cv::namedWindow("no edges", cv::WINDOW_NORMAL | cv::WINDOW_KEEPRATIO | cv::WINDOW_GUI_EXPANDED);
//            cv::imshow("no edges", clean);

            double minVal, maxVal; 
            Point minLoc, maxLoc; 
            minMaxLoc(clean, &minVal, &maxVal, &minLoc, &maxLoc);
            if(minVal == maxVal){
                continue;
            }

//            Mat centered;
//            centerAndCrop(clean, centered);
//            cv::namedWindow("centered", cv::WINDOW_NORMAL | cv::WINDOW_KEEPRATIO | cv::WINDOW_GUI_EXPANDED);
//            cv::imshow("centered", centered);

            vector<pair<int, float> > recognizedDigits = model.inferClass(clean);
            // drop zeros since sudoku doesnt have them definitely
            for(auto it = recognizedDigits.begin(); it != recognizedDigits.end();){
                if((*it).first == 0) it = recognizedDigits.erase(it); 
                else it = next(it);
            }

            if(recognizedDigits[0].second >= MnistModel::acceptanceThreshold){
                cout << "Definitive digit: " << recognizedDigits[0].first << " with prob: " << recognizedDigits[0].second << endl;
                for(auto & possibleGame : possibleGames){
                    possibleGame.fill(i, j, recognizedDigits[0].first, recognizedDigits[0].second);
                }
            } else{
                cout << "Possible digits: " << recognizedDigits[0] << " and " << recognizedDigits[1] << " and " << recognizedDigits[2] << endl;
                vector<Sudoku> newPossibleGames = vector<Sudoku>();
                for(auto & possibleGame : possibleGames){
                    for(int p=1; p<recognizedDigits.size(); p++){
                        Sudoku newGame(possibleGame);
                        newGame.fill(i, j, recognizedDigits[p].first, recognizedDigits[p].second);
                        newPossibleGames.push_back(newGame);
                    }
                    possibleGame.fill(i, j, recognizedDigits[0].first, recognizedDigits[0].second);
                }
                possibleGames.insert(possibleGames.end(), newPossibleGames.begin(), newPossibleGames.end() );
            }
            //waitKey(0);
        }
    }
    cout << "N total games " <<  possibleGames.size() << endl;
    //destroyAllWindows();

    #pragma omp parallel for
    for(int i=0; i<possibleGames.size(); i++){
        if(possibleGames[i].isValid())
            possibleGames[i].solve();
    }

    cout << "********** Solved Games **********" << endl;
    for(int i=0; i<possibleGames.size(); i++){
        if(!possibleGames[i].isSolved()) continue;
        possibleGames[i].print();
        cout << endl;

        Mat drawing;
        drawResult(img, drawing, sudokuGrid, possibleGames[i]);

        stringstream title;
        title << "Result " << i;
        cv::namedWindow(title.str(), cv::WINDOW_NORMAL | cv::WINDOW_KEEPRATIO | cv::WINDOW_GUI_EXPANDED);
        cv::imshow(title.str(), drawing);
    }

    waitKey(0);
    destroyAllWindows();    
}

