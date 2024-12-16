#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include <cassert>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <stdexcept>
#include <iomanip> 

using namespace std;

#define CHECK(call) \
{ \
	const cudaError_t error = call; \
	if (error != cudaSuccess) \
	{ \
		fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__); \
		fprintf(stderr, "code: %d, reason: %s\n", error, cudaGetErrorString(error)); \
		exit(EXIT_FAILURE); \
	} \
}

// Function to parse a CSV file into a 2D vector
vector<vector<double>> load_csv(const string& filepath, bool normalize = false) {
    ifstream file(filepath);
    if (!file.is_open()) throw runtime_error("Cannot open file: " + filepath);

    vector<vector<double>> data;
    string line;

    while (getline(file, line)) {
        vector<double> row;
        stringstream ss(line);
        string value;

        while (getline(ss, value, ',')) {
            double val = stod(value);
            if (normalize) val /= 255.0; // Normalize pixel values to [0, 1] if required
            row.push_back(val);
        }
        data.push_back(row);
    }

    file.close();
    return data;
}

// Function to load labels and convert to one-hot encoding
vector<vector<double>> load_labels_csv(const string& filepath, int num_classes = 10) {
    ifstream file(filepath);
    if (!file.is_open()) throw runtime_error("Cannot open file: " + filepath);

    vector<vector<double>> labels;
    string line;

    while (getline(file, line)) {
        int label = stoi(line); // Read the label
        vector<double> one_hot(num_classes, 0.0);
        one_hot[label] = 1.0; // One-hot encode
        labels.push_back(one_hot);
    }

    file.close();
    return labels;
}

// Main function to load and preprocess the dataset
void load_and_preprocess_dataset_csv(const string& train_images_path,
    const string& train_labels_path,
    const string& test_images_path,
    const string& test_labels_path,
    vector<vector<double>>& train_images,
    vector<vector<double>>& train_labels,
    vector<vector<double>>& test_images,
    vector<vector<double>>& test_labels) {
    try {
        // Load train images and normalize
        train_images = load_csv(train_images_path, true);

        // Load train labels and one-hot encode
        train_labels = load_labels_csv(train_labels_path);

        // Load test images and normalize
        test_images = load_csv(test_images_path, false);

        // Load test labels and one-hot encode
        test_labels = load_labels_csv(test_labels_path);

        // Verify consistency of data
        if (train_images.size() != train_labels.size()) {
            throw runtime_error("Mismatch between number of train images and labels.");
        }
        if (test_images.size() != test_labels.size()) {
            throw runtime_error("Mismatch between number of test images and labels.");
        }

        cout << "Dataset loaded successfully:" << endl;
        cout << " - Train samples: " << train_images.size() << endl;
        cout << " - Test samples: " << test_images.size() << endl;

    }
    catch (const exception& ex) {
        cerr << "Error while loading dataset: " << ex.what() << endl;
    }
}

// CUDA Kernel for matrix multiplication
__global__ void matmul(double* a, double* b, double* c, int rows_a, int cols_a, int cols_b) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows_a && col < cols_b) {
        double sum = 0.0;
        for (int i = 0; i < cols_a; ++i) {
            sum += a[row * cols_a + i] * b[i * cols_b + col];
        }
        c[row * cols_b + col] = sum;
    }
}

// CUDA Kernel for ReLU activation
__global__ void relu_activation(double* data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] = fmax(0.0, data[idx]);
    }
}

// CUDA Kernel for ReLU derivative
__global__ void relu_derivative(double* gradients, double* outputs, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        gradients[idx] *= (outputs[idx] > 0.0 ? 1.0 : 0.0);
    }
}

// CUDA Kernel for softmax activation
__global__ void softmax_activation(double* data, int size, int batch_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < batch_size) {
        double max_val = -INFINITY;
        for (int i = 0; i < size; ++i) {
            max_val = fmax(max_val, data[idx * size + i]);
        }

        double sum = 0.0;
        for (int i = 0; i < size; ++i) {
            data[idx * size + i] = exp(data[idx * size + i] - max_val);
            sum += data[idx * size + i];
        }

        for (int i = 0; i < size; ++i) {
            data[idx * size + i] /= sum;
        }
    }
}

// CUDA Kernel for weight updates
__global__ void update_weights(double* weights, double* gradients, double learning_rate, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        weights[idx] -= learning_rate * gradients[idx];
    }
}

// CUDA Kernel for predicting the class
__global__ void predict_kernel(double* outputs, int* predictions, int batch_size, int output_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < batch_size) {
        double max_val = outputs[idx * output_size];
        int max_idx = 0;
        for (int i = 1; i < output_size; ++i) {
            if (outputs[idx * output_size + i] > max_val) {
                max_val = outputs[idx * output_size + i];
                max_idx = i;
            }
        }
        predictions[idx] = max_idx;
    }
}

// Neural Network Layer
class CUDALayer {
public:
    int input_size, output_size;
    double* weights;
    double* biases;
    double* outputs;
    double* inputs;
    double* gradients;

    CUDALayer(int input_size, int output_size)
        : input_size(input_size), output_size(output_size) {
        // Allocate GPU memory for weights, biases, outputs, and gradients
        CHECK(cudaMalloc(&weights, input_size * output_size * sizeof(double)));
        CHECK(cudaMalloc(&biases, output_size * sizeof(double)));
        CHECK(cudaMalloc(&outputs, output_size * sizeof(double)));
        CHECK(cudaMalloc(&gradients, output_size * sizeof(double)));

        // Initialize weights and biases
        vector<double> h_weights(input_size * output_size);
        vector<double> h_biases(output_size);

        for (auto& w : h_weights) w = ((double)rand() / RAND_MAX) * 0.01;
        for (auto& b : h_biases) b = 0.0;

        CHECK(cudaMemcpy(weights, h_weights.data(), input_size * output_size * sizeof(double), cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(biases, h_biases.data(), output_size * sizeof(double), cudaMemcpyHostToDevice));
    }

    // ~CUDALayer() {
    //     // Free GPU memory
    //     CHECK(cudaFree(weights));
    //     CHECK(cudaFree(biases));
    //     CHECK(cudaFree(outputs));
    //     CHECK(cudaFree(gradients));
    // }

    ~CUDALayer() {
        if (weights) {
            cout << "Freeing weights memory..." << endl;
            CHECK(cudaFree(weights));
            weights = nullptr;
        }
        if (biases) {
            cout << "Freeing biases memory..." << endl;
            CHECK(cudaFree(biases));
            biases = nullptr;
        }
        if (outputs) {
            cout << "Freeing outputs memory..." << endl;
            CHECK(cudaFree(outputs));
            outputs = nullptr;
        }
        if (gradients) {
            cout << "Freeing gradients memory..." << endl;
            CHECK(cudaFree(gradients));
            gradients = nullptr;
        }
    }
    

    void forward(double* d_inputs, int batch_size, bool softmax = false) {
        // Matrix multiplication for inputs * weights
        dim3 blockSize(16, 16);
        dim3 gridSize((output_size + blockSize.x - 1) / blockSize.x,
                      (batch_size + blockSize.y - 1) / blockSize.y);

        matmul<<<gridSize, blockSize>>>(d_inputs, weights, outputs, batch_size, input_size, output_size);
        CHECK(cudaGetLastError());

        if (softmax) {
            // Apply softmax to the output layer
            softmax_activation<<<(batch_size + 255) / 256, 256>>>(outputs, output_size, batch_size);
            CHECK(cudaGetLastError());
        } else {
            // Add biases and apply activation function (ReLU)
            int total_outputs = batch_size * output_size;
            relu_activation<<<(total_outputs + 255) / 256, 256>>>(outputs, total_outputs);
            CHECK(cudaGetLastError());
        }
    }

    void backward(double* d_inputs, double* d_gradients, int batch_size, double learning_rate) {
        // Backpropagation logic to compute gradients
        int total_outputs = batch_size * output_size;
        relu_derivative<<<(total_outputs + 255) / 256, 256>>>(d_gradients, outputs, total_outputs);
        CHECK(cudaGetLastError());

        // Update weights
        int weight_size = input_size * output_size;
        update_weights<<<(weight_size + 255) / 256, 256>>>(weights, d_gradients, learning_rate, weight_size);
        CHECK(cudaGetLastError());
    }
};

// Cross-entropy loss
double cross_entropy_loss(const vector<double>& predictions, const vector<double>& labels) {
    double loss = 0.0;
    for (size_t i = 0; i < predictions.size(); ++i) {
        loss -= labels[i] * log(predictions[i] + 1e-15); // Avoid log(0)
    }
    return loss;
}

// Neural Network
class CUDANeuralNetwork {
public:
    vector<CUDALayer> layers;

    CUDANeuralNetwork(const vector<int>& architecture) {
        for (size_t i = 1; i < architecture.size(); i++) {
            layers.emplace_back(architecture[i - 1], architecture[i]);
        }
    }

    // Forward Pass
    void forward_pass(double* d_inputs, int batch_size) {
        for (size_t i = 0; i < layers.size(); i++) {
            bool is_output_layer = (i == layers.size() - 1);
            layers[i].forward(d_inputs, batch_size, is_output_layer);
            d_inputs = layers[i].outputs; // Pass outputs to the next layer
        }
    }

    // Backward Pass
    void backward_pass(double* d_inputs,double* d_labels, int batch_size, double learning_rate) {
        double* d_gradients = nullptr;
        for (int i = layers.size() - 1; i >= 0; i--) {
            layers[i].backward(d_inputs, d_gradients, batch_size, learning_rate);
            d_gradients = layers[i].gradients; // Pass gradients to the previous layer
        }
    }

    void train(double* d_inputs, double* d_labels, int epochs, int batch_size, double learning_rate) {
        for (int epoch = 0; epoch < epochs; epoch++) {
            forward_pass(d_inputs, batch_size);  // Forward Pass
            backward_pass(d_inputs, d_labels, batch_size, learning_rate);  // Backward Pass

            cout << "Epoch " << epoch + 1 << " completed." << endl;
        }
    }

    // Evaluate the model
    double evaluate(double* d_inputs, double* d_labels, int batch_size, int total_samples, int output_size) {
        int* d_predictions;
        CHECK(cudaMalloc(&d_predictions, sizeof(int) * total_samples));

        forward_pass(d_inputs, batch_size);
        predict_kernel<<<(batch_size + 255) / 256, 256>>>(layers.back().outputs, d_predictions, batch_size, output_size);
        CHECK(cudaGetLastError());

        // Copy predictions back to host
        vector<int> h_predictions(total_samples);
        CHECK(cudaMemcpy(h_predictions.data(), d_predictions, sizeof(int) * total_samples, cudaMemcpyDeviceToHost));

        // Evaluate accuracy
        int correct = 0;
        vector<int> h_labels(total_samples);
        CHECK(cudaMemcpy(h_labels.data(), d_labels, sizeof(int) * total_samples, cudaMemcpyDeviceToHost));

        for (int i = 0; i < total_samples; ++i) {
            if (h_predictions[i] == h_labels[i]) correct++;
        }

        CHECK(cudaFree(d_predictions));
        return static_cast<double>(correct) / total_samples;
    }
};

int main() {
    vector<vector<double>> train_images, train_labels;
    vector<vector<double>> test_images, test_labels;

    const string train_images_path = "x_train.csv";
    const string train_labels_path = "y_train.csv";
    const string test_images_path = "x_test.csv";
    const string test_labels_path = "y_test.csv";

    const string model_path = "nn_model.bin";
	cout<<"1";

    // Load and preprocess the dataset
    load_and_preprocess_dataset_csv(train_images_path, train_labels_path, test_images_path, test_labels_path,
        train_images, train_labels, test_images, test_labels);
	cout<<"2";
    // Example setup: Define the architecture
    CUDANeuralNetwork nn({ 784, 128, 128, 10 });

    // Allocate GPU memory for inputs and labels
    double* d_train_images;
    double* d_train_labels;
    double* d_test_images;
    double* d_test_labels;
	cout<<"3";

    CHECK(cudaMalloc(&d_train_images, train_images.size() * train_images[0].size() * sizeof(double)));
    CHECK(cudaMalloc(&d_train_labels, train_labels.size() * train_labels[0].size() * sizeof(double)));
    CHECK(cudaMalloc(&d_test_images, test_images.size() * test_images[0].size() * sizeof(double)));
    CHECK(cudaMalloc(&d_test_labels, test_labels.size() * test_labels[0].size() * sizeof(double)));

    CHECK(cudaMemcpy(d_train_images, train_images.data(), train_images.size() * train_images[0].size() * sizeof(double), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_train_labels, train_labels.data(), train_labels.size() * train_labels[0].size() * sizeof(double), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_test_images, test_images.data(), test_images.size() * test_images[0].size() * sizeof(double), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_test_labels, test_labels.data(), test_labels.size() * test_labels[0].size() * sizeof(double), cudaMemcpyHostToDevice));
	cout<<"4";
    // Train the model
    nn.train(d_train_images, d_train_labels, 10, 32, 0.01);

    // Evaluate the model
	cout<<"5";

    double train_accuracy = nn.evaluate(d_train_images, d_train_labels, 32, train_images.size(), 10);
    double test_accuracy = nn.evaluate(d_test_images, d_test_labels, 32, test_images.size(), 10);

    cout << "Train Accuracy: " << train_accuracy * 100.0 << "%" << endl;
    cout << "Test Accuracy: " << test_accuracy * 100.0 << "%" << endl;

    // Free GPU memory
    CHECK(cudaFree(d_train_images));
    CHECK(cudaFree(d_train_labels));
    CHECK(cudaFree(d_test_images));
    CHECK(cudaFree(d_test_labels));

    return 0;
}
