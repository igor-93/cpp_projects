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
        // default constructor
        Sudoku() = default;
        // conversion constructor
        explicit Sudoku(vector<vector <int> >& grid): grid(grid) {};
        // explicit copy-constructor
        explicit Sudoku(const Sudoku& other);
        // move constructor
        Sudoku(Sudoku &&) = default;
        // move assignment is necessary when defining move constructor
        Sudoku& operator=( Sudoku && ) = default;
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
        bool solved{false};
        int nIters{0};
        vector<vector<int> > grid = vector<vector <int> >(N, vector<int>(N, UNASSIGNED));
        vector<vector<double> > probabilities = vector<vector <double> >(N, vector<double>(N, UNASSIGNED));

};

#endif