#include <stdio.h>
#include <iostream>
#include <string>
#include <sys/time.h>

#include <opencv4/opencv2/core.hpp>
#include <opencv4/opencv2/core/utility.hpp>
#include <opencv4/opencv2/imgproc.hpp>
#include <opencv4/opencv2/highgui.hpp>

#include "Sudoku.hpp"
#include "MnistModel.hpp"
#include "ImgProc.hpp"

using namespace cv;
using namespace std;

Scalar blue(255, 0, 0);
Scalar red(0, 0, 255);
Scalar green(0, 255, 0);
Scalar black(0, 0, 0);

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
    MnistModel m = MnistModel();
    m.testLibTorch();
}

bool isHorizontalOrVertical(Vec2f line){
    float deg = line[1] / CV_PI * 180.0;
    bool result = false;
    
    if((80 < deg && deg < 100) || (170 < deg && deg < 190) || (260 < deg && deg < 280)){
        result = true;
    }
    if(deg < 10 || deg > 350){
        result = true;
    }

    cout << "deg: " << deg << " isHorizontalOrVertical: " << result << endl;
    return result;
}

void drawLines(const Mat& cdst, const vector<Vec2f>& lines){
    // Draw the lines
    for( size_t i = 0; i < lines.size(); i++ )
    {
        float rho = lines[i][0], theta = lines[i][1];
        printf("Rho = %.4f, Theta = %.4f \n", rho, theta);
        bool to_keep = isHorizontalOrVertical(lines[i]);
        Point pt1, pt2;
        double a = cos(theta), b = sin(theta);
        double x0 = a*rho, y0 = b*rho;
        pt1.x = cvRound(x0 + 1000*(-b));
        pt1.y = cvRound(y0 + 1000*(a));
        pt2.x = cvRound(x0 - 1000*(-b));
        pt2.y = cvRound(y0 - 1000*(a));
        Scalar c(0, 0, 255);
        if(to_keep) 
            c = Scalar(255, 0, 0);
        line(cdst, pt1, pt2, c, 1, LINE_AA);
    }
}

void cutEdges(Mat& img, Mat& binaryImg){
    threshold(img, binaryImg, mean(img)[0], 255, THRESH_BINARY);

    for(int i=0;i<img.size().height; i++){
        for(int j=0;j<img.size().width; j++){
            if(i == 0 || j == 0 || i == img.size().height-1 || j == img.size().width-1){
                floodFill(binaryImg, cv::Point(j, i), Scalar(0));
            }
        }   
    }
} 

void centerAndCrop(Mat& img, Mat& centered){
    Point2i center(0,0);
    float cnt = 0;

    for(int i=0;i<img.size().height; i++){
        for(int j=0;j<img.size().width; j++){
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
    ImgProc::crop(centered, centered);
} 


void drawResult(const Mat& img, Mat& drawing, const vector<vector<Rect> >& cells, const Sudoku& digits){
    cvtColor(img, drawing, COLOR_GRAY2RGB);

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

    cout << "argc: " << argc << endl;
    if(argc < 2){
        cout << "Please provide sudoku image to solve." << endl;
        throw 1;
    }

    Mat img = imread(argv[1], IMREAD_GRAYSCALE);
    
    ImgProc processor(img);
    MnistModel model = MnistModel();
    // model.trainModel();

    cout << "Running image processor...";
    processor.run();
    cout << " done." << endl;

    vector<vector<cv::Rect> > sudokuGrid = processor.getSudokuCells();

    vector<Sudoku> possibleGames = vector<Sudoku>();
    possibleGames.push_back(Sudoku());

    for(int i=0; i<9; i++){
        for(int j=0; j<9; j++){
            cout << "ROI at [" << i << "," << j << "]: " << endl;
            Rect cellROI = sudokuGrid[i][j];
            Mat cellImg = img(cellROI);

            Mat digit = ImgProc::invertImg(cellImg);
            Mat clean;
            cutEdges(digit, clean);

            double minVal, maxVal; 
            Point minLoc, maxLoc; 
            minMaxLoc(clean, &minVal, &maxVal, &minLoc, &maxLoc);
            if(minVal == maxVal){
                cout << "no digit found, skipping" << endl;
                continue;
            }

            Mat centered;
            centerAndCrop(clean, centered);
            // cv::namedWindow("centered", cv::WINDOW_NORMAL | cv::WINDOW_KEEPRATIO | cv::WINDOW_GUI_EXPANDED);
            // cv::imshow("centered", centered);

            vector<pair<int, float> > recognizedDigits = model.inferClass(centered);
            // drop zeros since sudoku doesnt have them definitely
            for(auto it = recognizedDigits.begin(); it != recognizedDigits.end();){
                if((*it).first == 0) it = recognizedDigits.erase(it); 
                else ++it;
            }

            if(recognizedDigits[0].second >= MnistModel::acceptanceThreshold){
                cout << "Definitive digit: " << recognizedDigits[0].first << " with prob: " << recognizedDigits[0].second << endl;
                for(int g=0; g<possibleGames.size(); g++){
                    possibleGames[g].fill(i, j, recognizedDigits[0].first, recognizedDigits[0].second);
                }
            } else{
                cout << "Possible digits: " << recognizedDigits[0] << " and " << recognizedDigits[1] << " and " << recognizedDigits[2] << endl;
                vector<Sudoku> newPossibleGames = vector<Sudoku>();
                for(int g=0; g<possibleGames.size(); g++){
                    for(int p=1; p<recognizedDigits.size(); p++){
                        Sudoku newGame(possibleGames[g]);
                        newGame.fill(i, j, recognizedDigits[p].first, recognizedDigits[p].second);
                        newPossibleGames.push_back(newGame);
                    }
                    possibleGames[g].fill(i, j, recognizedDigits[0].first, recognizedDigits[0].second);
                }
                possibleGames.insert(possibleGames.end(), newPossibleGames.begin(), newPossibleGames.end() );
            }

            //waitKey(0);
        }
    }

    //destroyAllWindows();

    // remove invalid grids
    for(auto it = possibleGames.begin(); it != possibleGames.end();){
        if(!(*it).isValid()){
            it = possibleGames.erase(it); 
        } else{
            ++it;
        }
    }
    // remove duplicates
    possibleGames.erase( unique( possibleGames.begin(), possibleGames.end() ), possibleGames.end() );

    cout << "N possible valid games " <<  possibleGames.size() << endl;
    vector<Sudoku*> solvedGames;
    for(auto it = possibleGames.begin(); it != possibleGames.end();){
        (*it).print();
        cout << endl;
        bool solved = (*it).solve();
        if(solved){
            cout << "*** SOLVED: ***" << endl;
            (*it).print();
            cout << endl;
            solvedGames.push_back(&(*it));
        }
        ++it;
    }

    cout << "********** Solved Games **********" << endl;
    for(int i=0; i<solvedGames.size(); i++){
        solvedGames[i]->print();
        cout << endl;

        Mat drawing;
        drawResult(img, drawing, sudokuGrid, *(solvedGames[i]));

        stringstream title;
        title << "Result " << i;
        cv::namedWindow(title.str(), cv::WINDOW_NORMAL | cv::WINDOW_KEEPRATIO | cv::WINDOW_GUI_EXPANDED);
        cv::imshow(title.str(), drawing);
    }

    waitKey(0);
    destroyAllWindows();    
}

