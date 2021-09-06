#include <iostream>
#include <sstream>
#include <string>
#include <algorithm>
#include "Network.h"

using namespace std;

inline float sigmoid(float x) { return 1 / (1 + exp(-x)); }
inline float d_sigmoid(float x) { return sigmoid(x) * (1 - sigmoid(x)); }
inline int random60000() { return rand() % 300 + (rand() % 200) * 300; }

void tic() {t = clock();}
void toc() {
	double time_taken = ((double)(clock() - t)) / CLOCKS_PER_SEC;
	cerr << "time taken: " << time_taken << endl;
}

layertype trans(string s) {
	if (s == "dense") {
		return layertype::DENSE;
	}
	else if (s == "sigm") {
		return layertype::SIGM;
	}
	else if (s == "relu") {
		return layertype::RELU;
	}
	else if (s == "conv") {
		return layertype::CONV;
	}
	else if (s == "sigm") {
		return layertype::POOL;
	}
	else {
		throw invalid_argument("no layertype");
	}
}

Dense::Dense(Activation* input_, int output_dim_, float mean, float stddev) {
	
	tp = layertype::DENSE;
	
	input = input_;
	input_size = input->size;
	input_rows = input->rows;
	input_cols = input->cols;
	input_depth = input->depth;

	output.update(output_dim_);
	output_size = output_dim_;
	output_rows = output_dim_;
	output_cols = 1;
	output_depth = 1;

	weights.update(Matrix(output_dim_, input_size, mean, stddev));
	biases.update(Vector(output_dim_, mean, stddev));
	w_updates.update(Matrix(output_dim_, input_size));
	b_updates.update(Vector(output_dim_));
	w_updates.make_zero();
	b_updates.make_zero();

}
void Dense::FFW() {
	float sum;
	int j;
	int i;
	for (j = 0; j < output_rows; j++) {
		sum = 0;
		for (i = 0; i < input_size; i++) {
			sum += weights.at(j, i) * input->at(i);
		}
		output.at(j) = sum + biases.at(j);
	}
}
void Dense::Backprop() {
	float sum;
	int j;
	int i;
	for (j = 0; j < output_rows; j++) {
		b_updates.at(j) += output.at(j);
	}
	for (i = 0; i < input_rows; i++) {
		sum = 0;
		for (j = 0; j < output_rows; j++) {
			w_updates.at(j, i) += output.at(j) * input->at(i);
			sum += weights.at(j, i) * output.at(j);
		}
		input->at(i) = sum;
	}
}
void Dense::Update(float alpha) {
	int j;
	int i;
	//b_updates.print();
	for (j = 0; j < output_rows; j++) {
		biases.at(j) -= alpha * b_updates.at(j);
		b_updates.at(j) = 0;
		for (i = 0; i < input_rows; i++) {
			weights.at(j, i) -= alpha * w_updates.at(j, i);
			w_updates.at(j, i) = 0;
		}
	}
}

Sigm::Sigm(Activation* input_) {
	tp = layertype::SIGM;

	input = input_;
	input_size = input->size;
	input_rows = input->rows;
	input_cols = input->cols;
	input_depth = input->depth;

	output.update(input->size, input->rows, input->cols, input->depth);
	output_size = input->size;
	output_rows = input->rows;
	output_cols = input->cols;
	output_depth = input->depth;
}

void Sigm::FFW() {
	int j;
	for (j = 0; j < input_size; j++) {
		output.at(j) = sigmoid(input->at(j));
	}
}
void Sigm::Backprop() {
	int j;
	for (j = 0; j < input_size; j++) {
		input->at(j) = output.at(j) * d_sigmoid(input->at(j));
	}
}
void Sigm::Update(float alpha) {}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

Relu::Relu(Activation* input_) {
	tp = layertype::RELU;

	input = input_;
	int input_size = input->size;
	int input_rows = input->rows;
	int input_cols = input->cols;
	int input_depth = input->depth;

	output.update(input->size, input->rows, input->cols, input->depth);
	int output_size = input->size;
	int output_rows = input->rows;
	int output_cols = input->cols;
	int output_depth = input->depth;
}
void Relu::FFW() {
	int j;
	for (j = 0; j < input_size; j++) {
		output.at(j) = max(input->at(j), 0.0f);
	}
}
void Relu::Backprop() {
	int j;
	for (j = 0; j < input_size; j++) {
		input->at(j) = output.at(j) * (input->at(j) > 0);
	}
}
void Relu::Update(float alpha) {}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

Conv::Conv(Activation* input_, int n_filters_, int f_size_, int s_, int padding_, float mean, float stddev) :
	s(s_), f_size(f_size_), n_filters(n_filters_), padding(padding_) {
	
	tp = layertype::CONV;
	
	input = input_;
	input_size = input->size;
	input_rows = input->rows;
	input_cols = input->cols;
	input_depth = input->depth;

	if ((input_rows + 2 * padding - f_size) % s != 0 ||
		(input->cols + 2 * padding - f_size) % s != 0) {
		throw invalid_argument("Invalid size/stride/padding combination");
	}
	output_rows = (input_rows + 2 * padding - f_size) / s;
	output_cols = (input_cols + 2 * padding - f_size) / s;
	output_depth = n_filters * input_depth;
	output_size = output_rows * output_cols * output_depth;
	output.update(output_size, output_rows, output_cols, output_depth);


	weights.update(Tesseract(n_filters, f_size, f_size, input_depth, mean, stddev));
	biases.update(Vector(n_filters, mean, stddev));
	w_updates.update(Tesseract(n_filters, f_size, f_size, input_depth));
	b_updates.update(Vector(n_filters));
	w_updates.make_zero();
	b_updates.make_zero();
}
void Conv::FFW() {
	//conv_ffw(input, &activation, &weights, &biases, filter_size, n_filters, filter_stride, padding);
}
void Conv::Backprop() {
	float sum;
	float val;
	int n;
	int d;
	int ja;
	int ia;
	int i_;
	int j_;
	int i;
	int j;

	// biases only!
	for (n = 0; n < output_depth; n++) {
		// actual
		sum = 0;
		for (ja = 0; ja < output_rows; ja++) {
			for (ia = 0; ia < output_cols; ia++) {
				sum += output.at(ja, ia, n);
			}
		}
		b_updates.at(n) = sum;
	}

	// weights only!
	for (n = 0; n < output_depth; n++) {
		for (i_ = 0; i_ < f_size; i_++) {
			for (j_ = 0; j_ < f_size; j_++) {
				for (d = 0; d < input_depth; d++) {
					//actual
					sum = 0;
					for (ja = 0; ja < output_rows; ja++) {
						for (ia = 0; ia < output_cols; ia++) {
							sum += output.at(ja, ia, d) * input->at(ja * s + j_, ia * s + i_, d);
						}
					}
					w_updates.at(j_, i_, d, n) = sum;			
				}
			}
		}
	}

	// inputs only!
	for (d = 0; d < input_depth; d++) {
		for (j = 0; j < input_rows; j++) {
			for (i = 0; i < input_cols; i++) {

				//actual
				sum = 0;
				for (ja = j / s; ja > (j - f_size) / s; j--) {
					j_ = j - s * ja;
					for (ia = i / s; ia > (i - f_size) / s; i--) {
						i_ = i - s * ia;
						for (n = 0; n < output_depth; n++) {
							sum += weights.at(j_, i_, d, n) * output.at(ja, ia, n);
						}
					}
				}
				input->at(j, i, d) = sum;
			}		
		}
	}
}
void Conv::Update(float alpha) {
	int n;
	int d;
	int j_;
	int i_;
	for (n = 0; n < output_depth; n++) {
		for (d = 0; d < input_depth; d++) {
			for (j_ = 0; j_ < f_size; j_++) {
				for (i_ = 0; i_ < f_size; i_++) {
					weights.at(j_, i_, d, n) += w_updates.at(j_, i_, d, n);
				}
			}
		}
	}
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

Pool::Pool(Activation* input_, int p_size_, int s_) : p_size(p_size_), s(s_) {
	tp = layertype::POOL;

	input = input_;
	input_size = input->size;
	input_rows = input->rows;
	input_cols = input->cols;
	input_depth = input->depth;

	if ((input_rows - p_size) % s != 0 ||
		(input->cols - p_size) % s != 0) {
		throw invalid_argument("Invalid size/stride/padding combination");
	}
	output_rows = (input_rows - p_size) / s;
	output_cols = (input_cols - p_size) / s;
	output_depth = input_depth;
	output_size = output_rows * output_cols * output_depth;
	output.update(output_size, output_rows, output_cols, output_depth);
}
void Pool::FFW() {
	float best;
	int d;
	int i_;
	int j_;
	int ja;
	int ia;

	for (d = 0; d < input_depth; d++) {
		for (ja = 0; ja < output_rows; ja++) {
			for (ia = 0; ia < output_cols; ia++) {
				// actual stuff
				best = -1000;
				for (j_ = 0; j_ < p_size; j_++) {
					for (i_ = 0; i_ < p_size; i_++) {
						best = max(best, input->at(ja * s + j_, ia * s + i_, d));
					}
				}
				output.at(ja, ia, d) = best;
			}
		}
	}
	output_copy.update(output);
}
void Pool::Backprop() {
	float val;
	float sum;
	int d;
	int j;
	int i;
	int ja;
	int ia;

	for (d = 0; d < input_depth; d++) {
		for (j = 0; j < input_rows; j++) {
			for (i = 0; i < input_cols; i++) {
				// actual stuff
				sum = 0;
				val = input->at(j, i, d);
				for (ja = j / s; ja > (j - p_size) / s; j--) {
					for (ia = i / s; ia > (i - p_size) / s; i--) {
						if (output_copy.at(ja, ia, d) == val) {
							sum += output.at(ja, ia, d);
						}
					}
				}
				input->at(j, i, d) *= sum;
			}
		}
	}


}

void Pool::Update(float alpha) {} // empty
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

Network::Network(string design, int input_size_, int input_rows_, int input_cols_, int input_depth_) :
	input(input_size_, input_size_/input_cols_/input_depth_, input_cols_, input_depth_), input_size(input_size_), input_rows(input_rows_), input_cols(input_cols_), input_depth(input_depth_) {
	istringstream iss(design);
	string temp;
	n_layers = 0;
	Activation* prev = &input;

	while (iss >> temp) {
		n_layers++;
		switch (trans(temp)) {
		case layertype::DENSE: {
			iss >> temp;
			Dense* p_tempobj = new Dense(prev, stoi(temp));
			layers.emplace_back(p_tempobj);
			prev = &layers.back()->output;
			break;
		}
		case layertype::SIGM: {
			Sigm* p_tempobj2 = new Sigm(prev);
			layers.emplace_back(p_tempobj2);
			prev = &layers.back()->output;
			break;
		}
		case layertype::RELU: {
			Relu* p_tempobj3 = new Relu(prev);
			layers.emplace_back(p_tempobj3);
			prev = &layers.back()->output;
			break;
		}
		case layertype::CONV: {
			iss >> temp;
			int n_filters = stoi(temp);
			iss >> temp;
			int f_size = stoi(temp);
			iss >> temp;
			int s = stoi(temp);
			Conv* p_tempobj = new Conv(prev, n_filters, f_size, s);
			layers.emplace_back(p_tempobj);
			prev = &layers.back()->output;
			break;
		}
		case layertype::POOL: {
			iss >> temp;
			int pool_stride = stoi(temp);
			iss >> temp;
			int pool_size = stoi(temp);
			Pool* p_tempobj = new Pool(prev, pool_size, pool_stride);
			layers.emplace_back(p_tempobj);
			prev = &layers.back()->output;
			break;
		}
		}
	}
	output_dim = layers.back()->output.size;
	output_values = layers.back()->output.values;
}
Network::~Network() {
	for (auto& layer : layers) {
		delete layer;
	}
}
int Network::GetResult() {
	for (auto& layer : layers) {
		layer->FFW();
	}
	return max_element(output_values, output_values + output_dim) - output_values;
}
void Network::Test_accuracy(uint8_t* testdata, uint8_t* testlabels, int sz) {
	int correct = 0;
	for (int i = 0; i < sz; i++) {
		input.update(&testdata[i * input_size]);
		correct += (GetResult() == testlabels[i]);
	}
	double percentage = correct / (sz / 100.0);
	cout << "Got " << correct << " correct, this is " << percentage << " percent" << endl;
}
void Network::ApplyUpdates(float alpha) {
	for (auto& layer : layers) { layer->Update(alpha); }
}
void Network::Change_updates(int label) {
	for (auto& layer : layers) { layer->FFW(); }
	layers.back()->output.at(label) -= 1;
	for (auto layer = layers.rbegin(); layer != layers.rend(); ++layer) { (*layer)->Backprop(); }
}
void Network::Batch_update(int batch_size, float eta, uint8_t* traindata, uint8_t* trainlabels) {
	Matrix temp;
	int adress;
	for (int i = 0; i < batch_size; i++) {
		adress = random60000();
		input.update(&traindata[adress * input_size]);
		Change_updates(trainlabels[adress]);
	}
	ApplyUpdates(eta / batch_size);
}


