#include<iostream>
#include <vector>
#include <functional>
#pragma once
    
class Matrix{
    private:
        int rows, columns;
        std::vector<double> data; // this is the flattened grid

    public:
        Matrix(int r, int c);

        double get(int r, int c) const;

        void set(int r, int c, double val);

        void check(const Matrix &B) const ;

        Matrix mul(const Matrix &B) const;
        Matrix add(const Matrix &B) const;
        Matrix subtract(const Matrix &B) const;
        Matrix mulElement(const Matrix &B) const; //Hadamard product definition
        
        Matrix applyFunc(std::function< double(double)> func)const;

        //for back prop, we need to transpose the matrices so;
        Matrix transpose() const;

        void randomize(); //for the start, where we need random biases and wehtihs
        void print() const;

        int getRows() const {return rows;};
        int getCols() const {return columns;};
};