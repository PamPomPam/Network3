
#include "Matrix.h"
#include <stdexcept>
#include <string>
#include <time.h>
#include <sstream>
#include <vector>
using namespace std;

unsigned seed = time(NULL);
std::default_random_engine generator(seed);


Activation::Activation() : size(-1), rows(-1), cols(-1), depth(-1), values(nullptr) {}
Activation::Activation(int size_) : size(size_), rows(size_), cols(1), depth(1) {
    values = new float[size];
}
Activation::Activation(int size_, int rows_, int cols_, int depth_) : size(size_), rows(rows_), cols(cols_), depth(depth_) {
    if (rows * cols * depth != size) {
        cout << "Invalid activation dimensions!" << endl;
        throw invalid_argument("");
    }
    values = new float[size];
}
Activation::Activation(vector<float> values_, int size_, int rows_, int cols_, int depth_) : size(size_), rows(rows_), cols(cols_), depth(depth_) {
    if (rows * cols * depth != size || size != values_.size()) {
        cout << "Invalid activation dimensions!" << endl;
        throw invalid_argument("");
    }
    values = new float[size];
    for (int i = 0; i < size; i++) {
        values[i] = values_[i];
    }
}
Activation::~Activation() {
    delete[] values;
}

void Activation::update(uint8_t arr[]) { // for mnist data, assumes size, cols and depth are already right
    delete[] values;
    values = new float[size];
    for (unsigned int i = 0; i < size; i++) {
        values[i] = arr[i] / 256.0;
    }
}
void Activation::update(int size_) {
    size = size_;
    rows = size_;
    cols = 1;
    depth = 1;
    values = new float[size];
}
void Activation::update(int size_, int rows_, int cols_, int depth_) {
    if (rows_ * cols_ * depth_ != size_) {
        cout << "Invalid activation dimensions!" << endl;
        throw invalid_argument("");
    }
    size = size_;
    rows = rows_;
    cols = cols_;
    depth = depth_;
    values = new float[size];
}
void Activation::update(const Activation& that) {
    cout << "should not be used in normal cases!" << endl;
    values = new float[size];
    memcpy(values, that.values, size * sizeof(that.values[0]));
}
void Activation::initialize(float mean, float stddev) {
    std::normal_distribution<float> distribution(mean, stddev);
    for (int j = 0; j < size; j++) {
        values[j] = distribution(generator);
    }
}

void Activation::print() {
    cerr << "---------------" << endl;
    string str;
    for (int j = 0; j < size; j++) {
        str = std::to_string(values[j]);
        cerr << str << endl;
    }
    cerr << "---------------" << endl << endl;
}
void Activation::printout() {
    cout << "---------------" << endl;
    string str;
    for (int j = 0; j < size; j++) {
        str = std::to_string(values[j]);
        cout << str << endl;
    }
    cout << "---------------" << endl << endl;
}
void Activation::print_image() { // for mnist data
    for (int y = 0; y < 28; y++) {
        for (int x = 0; x < 28; x++) {
            if (values[28 * y + x] > 0.5) {
                cerr << 'x';
            }
            else {
                cerr << ' ';
            }
        }
        cerr << endl;
    }
}
void Activation::shape() {
    cerr << "shape: " << rows << 'x' << cols << 'x' << depth << endl;
}




Vector::Vector() {
    rows = 0;
    values = nullptr;
}
Vector::Vector(int rows_) : rows(rows_) {
    values = new float[rows];
}
Vector::Vector(int rows_, float mean, float stddev) : rows(rows_) {
    values = new float[rows];
    initialize(mean, stddev);
}
Vector::Vector(vector<float> values_, int rows_) : rows(rows_) {
    if (rows != values_.size()) {
        cout << "Invalid vector dimensions!" << endl;
        throw invalid_argument("");
    }
    values = new float[rows];
    for (int i = 0; i < rows; i++) {
        values[i] = values_[i];
    }
}
Vector::~Vector() {
    delete[] values;
}

void Vector::update(const Vector& that) {
    rows = that.rows;
    values = new float[rows];
    memcpy(values, that.values, rows * sizeof(that.values[0]));
}
void Vector::initialize(float mean, float stddev) {
    std::normal_distribution<float> distribution(mean, stddev);
    for (int j = 0; j < rows; j++) {
        values[j] = distribution(generator);
    }
}
void Vector::make_zero() {
    memset(values, 0, rows * sizeof(values[0]));
}

void Vector::print() {
    cerr << "---------------" << endl;
    string str;
    for (int j = 0; j < rows; j++) {
        str = std::to_string(at(j));
        cerr << str << endl;
    }
    cerr << "---------------" << endl << endl;
}
void Vector::printout() {
    cout << "---------------" << endl;
    string str;
    for (int j = 0; j < rows; j++) {
        str = std::to_string(at(j));
        cout << str << endl;
    }
    cout << "---------------" << endl << endl;
}
void Vector::shape() {
    cerr << "shape: " << rows << endl;
}




Matrix::Matrix() {
    rows = 0;
    cols = 0;
    values = nullptr;
}
Matrix::Matrix(int rows_, int cols_) : rows(rows_), cols(cols_) {
    values = new float[rows * cols];
    //memset(values, 0, sizeof(values));
}
Matrix::Matrix(int rows_, int cols_, float mean, float stddev) : rows(rows_), cols(cols_) {
    values = new float[rows * cols];
    initialize(mean, stddev);
}
Matrix::Matrix(vector<float> values_, int rows_, int cols_) : rows(rows_), cols(cols_) {
    if (rows * cols != values_.size()) {
        cout << "Invalid matrix dimensions!" << endl;
        throw invalid_argument("");
    }
    values = new float[rows * cols];
    for (int i = 0; i < rows * cols; i++) {
        values[i] = values_[i];
    }
}
Matrix::~Matrix() {
    delete[] values;
}

void Matrix::update(const Matrix& that) {
    rows = that.rows;
    cols = that.cols;
    values = new float[rows * cols];
    memcpy(values, that.values, cols * rows * sizeof(that.values[0]));
}
void Matrix::initialize(float mean, float stddev) {
    std::normal_distribution<float> distribution(mean, stddev);
    for (int y = 0; y < rows; y++) {
        for (int x = 0; x < cols; x++) {
            values[y * cols + x] = distribution(generator);
        }
    }
}
void Matrix::make_zero() {
    memset(values, 0, cols * rows * sizeof(values[0]));
}

void Matrix::print() {
    cerr << "---------------" << endl;
    string str;
    for (int y = 0; y < rows; y++) {
        for (int x = 0; x < cols; x++) {
            str = std::to_string(at(y, x));
            cerr << str;
            for (unsigned int i = 0; i < (12 - str.length()); i++) { cerr << ' '; }
        }
        cerr << endl;
    }
    cerr << endl;
}
void Matrix::printout() {
    cout << "---------------" << endl;
    string str;
    for (int y = 0; y < rows; y++) {
        for (int x = 0; x < cols; x++) {
            str = std::to_string(at(y, x));
            cout << str;
            for (unsigned int i = 0; i < (12 - str.length()); i++) { cerr << ' '; }
        }
        cout << endl;
    }
    cout << endl;
}
void Matrix::shape() {
    cerr << "shape: " << rows << 'x' << cols << endl;
}



Cube::Cube() {
    rows = -1;
    cols = -1;
    depth = -1;
    values = nullptr;
}
Cube::Cube(int rows_, int cols_, int depth_) : rows(rows_), cols(cols_), depth(depth_) {
    values = new float[rows * cols * depth];
    //memset(values, 0, sizeof(values));
}
Cube::Cube(int rows_, int cols_, int depth_, float mean, float stddev) : rows(rows_), cols(cols_), depth(depth_) {
    values = new float[rows * cols * depth];
    initialize(mean, stddev);
}
Cube::Cube(vector<float> values_, int rows_, int cols_, int depth_) : rows(rows_), cols(cols_), depth(depth_) {
    if (rows * cols * depth != values_.size()) {
        cout << "Invalid matrix dimensions!" << endl;
        throw invalid_argument("");
    }
    values = new float[rows * cols * depth];
    for (int i = 0; i < rows * cols * depth; i++) {
        values[i] = values_[i];
    }
}
Cube::~Cube() {
    delete[] values;
}

void Cube::update(const Cube& that) {
    rows = that.rows;
    cols = that.cols;
    depth = that.depth;
    values = new float[rows * cols * depth];
    memcpy(values, that.values, cols * rows * depth * sizeof(that.values[0]));
}
void Cube::initialize(float mean, float stddev) {
    std::normal_distribution<float> distribution(mean, stddev);
    for (int j = 0; j < rows; j++) {
        for (int i = 0; i < cols; i++) {
            for (int d = 0; d < depth; d++) {
                at(j, i, d) = distribution(generator);
            }
        }
    }
}
void Cube::make_zero() {
    memset(values, 0, cols * rows * depth * sizeof(values[0]));
}

void Cube::shape() {
    cerr << "shape: " << rows << 'x' << cols << 'x' << depth << endl;
}



Tesseract::Tesseract() {
    rows = -1;
    cols = -1;
    depth = -1;
    n_depth = -1;
    values = nullptr;
}
Tesseract::Tesseract(int rows_, int cols_, int depth_, int n_depth_) : rows(rows_), cols(cols_), depth(depth_), n_depth(n_depth_) {
    values = new float[rows * cols * depth * n_depth];
    //memset(values, 0, sizeof(values));
}
Tesseract::Tesseract(int rows_, int cols_, int depth_, int n_depth_, float mean, float stddev) : rows(rows_), cols(cols_), depth(depth_), n_depth(n_depth_) {
    values = new float[rows * cols * depth * n_depth];
    initialize(mean, stddev);
}
Tesseract::Tesseract(vector<float> values_, int rows_, int cols_, int depth_, int n_depth_) : rows(rows_), cols(cols_), depth(depth_), n_depth(n_depth_) {
    if (rows * cols * depth * n_depth != values_.size()) {
        cout << "Invalid matrix dimensions!" << endl;
        throw invalid_argument("");
    }
    values = new float[rows * cols * depth * n_depth];
    for (int i = 0; i < rows * cols * depth * n_depth; i++) {
        values[i] = values_[i];
    }
}
Tesseract::~Tesseract() {
    delete[] values;
}

void Tesseract::update(const Tesseract& that) {
    rows = that.rows;
    cols = that.cols;
    depth = that.depth;
    n_depth = that.n_depth;
    values = new float[rows * cols * depth * n_depth];
    memcpy(values, that.values, cols * rows * depth * n_depth * sizeof(that.values[0]));
}
void Tesseract::initialize(float mean, float stddev) {
    std::normal_distribution<float> distribution(mean, stddev);
    for (int j = 0; j < rows; j++) {
        for (int i = 0; i < cols; i++) {
            for (int d = 0; d < depth; d++) {
                for (int n = 0; n < n_depth; n++) {
                    at(j, i, d, n) = distribution(generator);
                }
            }
        }
    }
}
void Tesseract::make_zero() {
    memset(values, 0, cols * rows * depth * n_depth * sizeof(values[0]));
}

void Tesseract::shape() {
    cerr << "shape: " << rows << 'x' << cols << 'x' << depth << 'x' << n_depth << endl;
}
