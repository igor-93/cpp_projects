#include "Sudoku.hpp"

using namespace std;

// helpers
inline bool findNextCell(const vector<vector<int> >& grid, int& row, int& col){
    for (row = 0; row < Sudoku::N; row++) 
        for (col = 0; col < Sudoku::N; col++) 
            if (grid[row][col] == UNASSIGNED) 
                return true; 
    return false; 
}

inline bool isValidEntry(const vector<vector<int> >& grid, int row, int col, int value){
    // check row
    for(int c=0; c<Sudoku::N; c++)
        if(c != col && grid[row][c] == value)
            return false;

    // check column
    for(int r=0; r<Sudoku::N; r++)
        if(r != row && grid[r][col] == value)
            return false;

    // check small box
    int smallBoxR = row / 3;
    int smallBoxC = col / 3;
    for(int r=smallBoxR*3; r<(smallBoxR+1)*3; r++)
        for(int c=smallBoxC*3; c<(smallBoxC+1)*3; c++)
            if((r != row && c != col) && grid[r][c] == value)
                return false;

    return true;
}

// members

Sudoku::Sudoku(){
    this->solved = false;
    this->grid = vector<vector <int> >(N, vector<int>(N, -1));
    this->probabilities = vector<vector <double> >(N, vector<double>(N, UNASSIGNED));
    this->nIters = 0;
}

Sudoku::Sudoku(vector<vector <int> >& grid){
    this->solved = false;
    this->grid = grid;
    this->probabilities = vector<vector <double> >(N, vector<double>(N, UNASSIGNED));
    this->nIters = 0;
}

Sudoku::Sudoku(const Sudoku& other){
    this->solved = other.solved;
    this->grid = other.grid;
    this->probabilities = other.probabilities;
    this->nIters = other.nIters;
}

bool Sudoku::fill(int row, int col, int value, double probability){
    if((this->grid[row][col] < 1) && (value < 10) & (value > 0)){
        this->grid[row][col] = value;
        this->probabilities[row][col] = probability;
        return true;
    }
    return false;
}

double Sudoku::getJoinProbability() const {
    double res = 0.0;
    for(int row = 0; row < Sudoku::N; row++){
        for(int col = 0; col < Sudoku::N; col++){
            if(probabilities[row][col] != UNASSIGNED){
                res += log(probabilities[row][col]);
            }
        }
    }
    return exp(res);
}

bool Sudoku::isValid() const {
    for(int row = 0; row < Sudoku::N; row++)
        for(int col = 0; col < Sudoku::N; col++)
            if(grid[row][col] != UNASSIGNED && !isValidEntry(grid, row, col, grid[row][col]))
                return false;
    return true;
}



bool Sudoku::solve(){
    if(!isValid()){
        cout << "Starting Sudoku puzzle is NOT valid." << endl;
        throw exception();
    }
    if(this->solved)
        return true;

    this->nIters =0;
    cout << "Solving Sudoku ...";
    this->solved = this->trySolve(this->grid);

    cout << " done! Ran in " << this->nIters << " iterations." << endl;
    return this->solved;
}

bool Sudoku::trySolve(vector<vector<int> >& grid){
    this->nIters++;
    int row, col;
    if(!findNextCell(grid, row, col)){
        return true; // done
    }
    
    // try all numbers in the next free cell
    for(int value=1; value<10; value++){
        // basic check to see if it makes sense to continue
        if(isValidEntry(grid, row, col, value)){
            grid[row][col] = value;
            if(trySolve(grid))
                return true;
            // if cant solve, put it back to unassigned so the next round can try it out
            grid[row][col] = UNASSIGNED;
        }
    }
    return false;
}

void Sudoku::print() const { 
    for (int row=0; row<N; row++) { 
        for (int col=0; col<N; col++){
            int val = grid[row][col];
            if(val == UNASSIGNED) 
                cout << "X" << " ";
            else
                cout << val << " "; 
        }
        cout << '\n';
    }
    flush(cout);
} 

bool Sudoku::operator==(const Sudoku& other){
    for (int row=0; row<N; row++) { 
        for (int col=0; col<N; col++){
            if(grid[row][col] != other.grid[row][col])
                return false;
        }
    } 
    return true;
}