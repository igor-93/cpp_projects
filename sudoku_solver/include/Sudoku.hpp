#ifndef SUDOKU_HPP
#define SUDOKU_HPP

#define UNASSIGNED -1

#include "vector"
#include "tuple"
#include <math.h>
#include <iostream>

using namespace std;

class Sudoku{

    public:
        Sudoku();
        explicit Sudoku(vector<vector <int> >& grid);
        explicit Sudoku(const Sudoku& other);
        bool operator==(const Sudoku& other);
        bool fill(int row, int col, int value, double probability = UNASSIGNED);
        double getJoinProbability() const ;
        bool isValid() const;
        bool isSolved() const {return this->solved;};
        bool solve();

        void print() const;

        int getValue(int row, int col) const {return grid[row][col];};
        float getProb(int row, int col) const {return probabilities[row][col];};

        bool trySolve(vector<vector<int> >& grid);
        static const int N = 9;

    private:
        bool solved;
        int nIters;
        vector<vector<int> > grid;
        vector<vector<double> > probabilities;

};

#endif