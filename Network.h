#pragma once

#include "Matrix.h"
#include <vector>
#include <string>

using namespace std;

static time_t t;
void tic();
void toc();

enum class layertype { DENSE, SIGM, RELU, CONV, POOL, NONE };
layertype trans(string s);

struct Layer {
	layertype tp = layertype::NONE;

	Activation* input = nullptr;
	int input_size = 0;
	int input_rows = 0;
	int input_cols = 0;
	int input_depth = 0;

	Activation output;
	int output_size = 0;
	int output_rows = 0;
	int output_cols = 0;
	int output_depth = 0;
	
	virtual void FFW() = 0;
	virtual void Backprop() = 0;
	virtual void Update(float alpha) = 0;
};

struct Dense : Layer {
	Matrix weights;
	Vector biases;

	Matrix w_updates;
	Vector b_updates;

	Dense(Activation* input_, int output_dim_, float mean = 0, float stddev = 1);
	void FFW() override;
	void Backprop() override;
	void Update(float alpha) override;
};

struct Sigm : Layer {
	Sigm(Activation* input_);
	void FFW() override;
	void Backprop() override;
	void Update(float alpha) override;
};

struct Relu : Layer {
	Relu(Activation* input_);
	void FFW() override;
	void Backprop() override;
	void Update(float alpha) override;
};

struct Conv : Layer {
	int n_filters;
	int f_size; // all filters are squares
	int s;
	int padding;

	Tesseract weights;
	Vector biases;
	Tesseract w_updates;
	Vector b_updates;

	Conv(Activation* input_, int n_filters_, int f_size_, int s_, int padding_ = 2, float mean = 0, float stddev = 1);
	void FFW() override;
	void Backprop() override;
	void Update(float alpha) override;
};

struct Pool : Layer {
	int p_size;
	int s;

	Activation output_copy; // for backprop

	Pool(Activation* input_, int p_size_, int s_);
	void FFW() override;
	void Backprop() override;
	void Backprop2();
	void Update(float alpha) override;
};






struct Network {
	Activation input;
	int input_size;
	int input_rows;
	int input_cols;
	int input_depth;

	vector<Layer*> layers;
	int n_layers;
	
	int output_dim;
	float* output_values;

	Network(string design, int input_size_, int input_rows = -1,  int input_cols_ = 1, int input_depth_ = 1);
	~Network();

	
	void Test_accuracy(uint8_t* testdata, uint8_t* testlabels, int sz);
	void Batch_update(int batch_size, float eta, uint8_t* traindata, uint8_t* trainlabels);

	int GetResult();
	void Change_updates(int label);
	void ApplyUpdates(float alpha);
};
