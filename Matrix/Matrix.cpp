#include "Matrix.h"

#include <vector>
#include <math.h>
#include <iostream>
#include <functional>
#include <random>

Matrix::Matrix(int r, int c) : rows(r), columns(c)  {
    data.resize(r*c);
    std::fill(data.begin(), data.end(), 0);
    //initialization is over
}

void Matrix::check(const Matrix &B) const {
    if (rows != B.rows || columns != B.columns) {
        throw std::invalid_argument("Matrix dimensions do not match");
        
    }
}

double Matrix::get(int r, int c) const{
    return data[r*getCols() + c];
}

void Matrix::set(int r, int c, double val){
    data[r*getCols() + c] = val;
}

Matrix Matrix::mul(const Matrix &B) const{
    if (columns != B.rows) {
        throw std::invalid_argument("Matrix multiplication dimension mismatch");
    }
    Matrix res(rows, B.columns);

    for (int i = 0; i < rows; i++){
        for(int j = 0; j < B.columns; j++){
            double s = 0;

            for (int k = 0; k <columns; k++){
                s += data[i*columns + k] * B.data[k*B.columns + j];
            }

            res.data[i*res.columns +j] = s;
        }
    }

    return res;

}

Matrix Matrix::add(const Matrix &B) const{
    check(B);
    Matrix res(rows, columns);
    
    for(int i = 0; i < rows; i++){
        for (int j = 0; j < columns; j++){
            int pos = i*getCols() + j;
            res.data[pos] = B.data[pos] + data[pos]; 
        }
    }

    return res;
}

Matrix Matrix::subtract(const Matrix &B) const{
    check(B);
    Matrix res(rows, columns);
    
    for(int i = 0; i < rows; i++){
        for (int j = 0; j < columns; j++){
            int pos = i*getCols() + j;
            res.data[pos] = - B.data[pos] + data[pos]; 
        }
    }

    return res;
}

Matrix Matrix::mulElement(const Matrix &B) const {
    check(B); 
    Matrix res(rows, columns);
    for (int i = 0; i < rows * columns; i++) {
        res.data[i] = data[i] * B.data[i];
    }
    return res;
}

Matrix Matrix::applyFunc(std::function<double(double)> func) const { 
    Matrix res(rows, columns);

    for(int i = 0; i < rows*columns; i++){
        res.data[i] = func(data[i]);
    }

    return res;
}

Matrix Matrix::transpose() const{
    Matrix res(columns, rows);

    for(int i = 0; i < rows; i++){
        for (int j = 0; j < columns; j++){
            res.data[j*rows + i] = data[i*columns + j];
        }
    }

    return res;
}

void Matrix::randomize(){
    std::random_device rd;//for random seed generation
    std::mt19937 gen(rd());

    std::uniform_real_distribution<> dis(-1,1);
    for(int i = 0; i < rows*columns; i++){
        data[i] = dis(gen);
    }
}

void Matrix::print() const{
    int newlinecnt = 0;
    for (int i = 0; i < rows*columns; i++){
        if(newlinecnt == columns){
            std::cout<<"\n";
            newlinecnt = 0;
        }
        std::cout<<data[i] << "\t";

        newlinecnt++;
    }
}