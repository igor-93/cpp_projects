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
        Sudoku(vector<vector <int> > grid);
        Sudoku(const Sudoku& other);
        bool fill(int row, int col, int value, double probability = UNASSIGNED);
        double getJoinProbability() const ;
        bool isValid() const;
        bool solve();
        bool trySolve(vector<vector<int> >& grid);
        void print() const;

        int getValue(int row, int col) const {return grid[row][col];};
        float getProb(int row, int col) const {return probabilities[row][col];};

        bool operator==(const Sudoku& other);

        static const int N;

    private:
        bool solved;
        int n_iters;
        vector<vector<int> > grid;
        vector<vector<double> > probabilities;

};

#endif