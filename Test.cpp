#include "Matrix.h"
#include "Network.h"
#include <time.h>
#include <algorithm>
#include <fstream>
#include <cstring>

using namespace std;


void MNIST_test() {
    srand(time(NULL));
    fstream f1;
    uint8_t* test_labels = new uint8_t[10000];
    uint8_t* train_labels = new uint8_t[60000];
    uint8_t* test_data = new uint8_t[10000 * 784];
    uint8_t* train_data = new uint8_t[60000 * 784];
    int magic_number;
    int nr_items;
    int row_size;
    int col_size;

    f1.open("C:\\Users\\jonel\\OneDrive\\Documenten\\Programmas\\newnn\\t10k-labels-idx1-ubyte", ios::in | ios::binary);
    f1.read((char*)&magic_number, sizeof(magic_number)); // 2049
    f1.read((char*)&nr_items, sizeof(nr_items)); // 10000
    for (int i = 0; i < 10000; ++i) {
        f1.read((char*)&(test_labels[i]), 1);
    }
    f1.close();

    f1.open("C:\\Users\\jonel\\OneDrive\\Documenten\\Programmas\\newnn\\train-labels-idx1-ubyte", ios::in | ios::binary);
    f1.read((char*)&magic_number, sizeof(magic_number)); // 2049
    f1.read((char*)&nr_items, sizeof(nr_items)); // 60000
    for (int i = 0; i < 60000; ++i) {
        f1.read((char*)&(train_labels[i]), 1);
    }
    f1.close();


    f1.open("C:\\Users\\jonel\\OneDrive\\Documenten\\Programmas\\newnn\\t10k-images-idx3-ubyte", ios::in | ios::binary);
    f1.read((char*)&magic_number, sizeof(magic_number)); // 2051
    f1.read((char*)&nr_items, sizeof(nr_items)); // 10000
    f1.read((char*)&col_size, sizeof(col_size)); // 28
    f1.read((char*)&row_size, sizeof(row_size)); // 28
    for (int i = 0; i < 10000; ++i) {
        f1.read((char*)&(test_data[784 * i]), 784);
    }
    f1.close();

    f1.open("C:\\Users\\jonel\\OneDrive\\Documenten\\Programmas\\newnn\\train-images-idx3-ubyte", ios::in | ios::binary);
    f1.read((char*)&magic_number, sizeof(magic_number)); // 2051
    f1.read((char*)&nr_items, sizeof(nr_items)); // 60000
    f1.read((char*)&col_size, sizeof(col_size)); // 28
    f1.read((char*)&row_size, sizeof(row_size)); // 28
    for (int i = 0; i < 60000; ++i) {
        f1.read((char*)&(train_data[784 * i]), 784);
    }
    f1.close();


    Network mynet("conv 10 4 2 dense 10 sigm", 784);
    mynet.Test_accuracy(test_data, test_labels, 10000);
    for (int i = 0; i < 30; i++) {
        tic();
        for (int j = 0; j < 6000; j++) {
            mynet.Batch_update(10, 2, train_data, train_labels);
        }
        mynet.Test_accuracy(test_data, test_labels, 10000);
        toc();
    }

    delete[] test_labels;
    delete[] train_labels;
    delete[] test_data;
    delete[] train_data;
}

int main() {
    MNIST_test();
}