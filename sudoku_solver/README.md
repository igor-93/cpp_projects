# Sudoku Solver

Tiny program that solves Sudoko that it sees on the image. It comprises of few main steps:
1. Recognize the sudoku grid:
    - solved with openCV's findContour(). For each contour get the bounding box
    - from all the bounding boxes, pick the one that has most intersections of 
    HoughLines and has the smallest area in case of a tie
    - to define each cell of Sudoku puzzle, 
        - first assume its position relative to the whole puzzle
        - expand it by 20% to include the corners
        - since most or all of the corners are stored as intersections of Hough lines, 
        use these to get maximum bounding box
        - result is the actual cell bounded by the intersections
1. Get each digit from Sudoku grid: 
    - remove the potential "frame" of the ROI of the cell by flood-filling the edges
    - discard if nothing inside (neural-net is not trained to infer non-digit)
1. Recognize the digit:
    - simple CNN with 2 conv layers and 2 fully-connected
    - pick top 3
    - if the first has prob > 80%, discard other 2 options, otherwise consider all 3 options
1. Solve Sudoku puzzle:
    - discard all invalid starting puzzles
    - solve others with depth-first search algorithm
    - discard all that don't have solution
1. Overlay inferred digits and fill in the blank cells

**ATTENTION: this is MVP version, meaning that each part works, but there is still lots of space for improvement.**

### Example 

Green is a recognized digit with high confidence, red with low confidence, blue are filled gaps.

Correct example:
![Correct Result](./result/result_correct.png)

Same example but with incorrect output: the model flasly classifed 1 as 9, and such puzzle was still validly solved.
![Wrong Result](./result/result_wrong.png)

### Requirements:
Inside of this folder the following is necessary:
1. `mnist/` with unzipped files from [Yann Lecun's site](http://yann.lecun.com/exdb/mnist/)
2. `model.pt` trained MNIST model that can be trained by uncommenting a line in main.cpp

### How to build
1. `mkdir build`
1. `cd build`
1. `cmake ..`
1. `cd .. & cmake --build build/`

### Example run:
from build folder: `build/SudokuSolver data/sudoku10.png`