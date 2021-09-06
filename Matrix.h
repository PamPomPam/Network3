#pragma once

#include <random>
#include <iostream>
#include <string>
#include <math.h>
#include <time.h>
#include <cstring>

using namespace std;


struct Activation {
    float* values;
    int size;

    int rows;
    int cols;
    int depth;

    inline float& at(int j) {
        return values[j];
    }
    inline float& at(int j, int i, int d) {
        return values[d * cols * rows + j * cols + i];
    }

    Activation();
    Activation(int size_);
    Activation(int size_, int rows_, int cols_, int depth_);
    Activation(vector<float> values_, int size_, int rows_ = 1, int cols_ = 1, int depth_ = 1);
    ~Activation();
    Activation(const Activation& that) = delete;
    Activation& operator=(const Activation& that) = delete;

    void update(uint8_t arr[]);
    void update(int size_);
    void update(int size_, int rows_, int cols_, int depth_);
    void update(const Activation& that);
    void initialize(float mean = 0, float stddev = 1);

    void print();
    void printout();
    void print_image();
    void shape();
};

struct Vector {
    float* values;
    int rows;

    inline float& at(int j) {
        return values[j];
    }
    Vector();
    Vector(int rows_);
    Vector(int rows_, float mean, float stddev);
    Vector(vector<float> values_, int rows_);
    ~Vector();
    Vector(const Vector& that) = delete;
    Vector& operator=(const Vector& that) = delete;

    void update(const Vector& that);
    void initialize(float mean, float stddev);
    void make_zero();

    void print();
    void printout();
    void shape();
};

struct Matrix {

    float* values;
    int rows;
    int cols;

    inline float& at(int j, int i) { return values[j * cols + i]; }

    Matrix();
    Matrix(int rows_, int cols_);
    Matrix(int rows_, int cols_, float mean, float stddev);
    Matrix(vector<float> values_, int rows_, int cols_);
    ~Matrix();
    Matrix(const Matrix& that) = delete;
    Matrix& operator=(const Matrix& that) = delete;

    void update(const Matrix& that);
    void initialize(float mean, float stddev);
    void make_zero();

    void print();
    void printout();
    void shape();


    
};

struct Cube {
    float* values;
    int rows;
    int cols;
    int depth;

    inline float& at(int j, int i, int d) { return values[d * cols * rows + j * cols + i]; }

    Cube();
    Cube(int rows_, int cols_, int depth_);
    Cube(int rows_, int cols_, int depth_, float mean, float stddev);
    Cube(vector<float> values_, int rows_, int cols_, int depth_);
    ~Cube();
    Cube(const Cube& that) = delete;
    Cube& operator=(const Cube& that) = delete;

    void update(const Cube& that);
    void initialize(float mean, float stddev);
    void make_zero();

    void shape();

};

struct Tesseract {
    float* values;
    int rows;
    int cols;
    int depth;
    int n_depth;

    inline float& at(int j, int i, int d, int n) {
        return values[n * depth * cols * rows + d * cols * rows + j * cols + i];
    }

    Tesseract();
    Tesseract(int rows_, int cols_, int depth_, int n_depth_);
    Tesseract(int rows_, int cols_, int depth_, int n_depth_, float mean, float stddev);
    Tesseract(vector<float> values_, int rows_, int cols_, int depth_, int n_depth_);
    ~Tesseract();
    Tesseract(const Tesseract& that) = delete;
    Tesseract& operator=(const Tesseract& that) = delete;

    void update(const Tesseract& that);
    void initialize(float mean, float stddev);
    void make_zero();

    void shape();
};
