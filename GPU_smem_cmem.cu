#include <iostream>
#include <cmath>
#include <random>
#include <algorithm>
#include <cassert>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <iomanip>
#include <cstring> // For memset and memcpy

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
struct GpuTimer
{
    cudaEvent_t start;
    cudaEvent_t stop;

    GpuTimer()
    {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
    }

    ~GpuTimer()
    {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    void Start()
    {
        cudaEventRecord(start, 0);
        cudaEventSynchronize(start);
    }

    void Stop()
    {
        cudaEventRecord(stop, 0);
    }

    float Elapsed()
    {
        float elapsed;
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsed, start, stop);
        return elapsed;
    }
};
// Function to parse a CSV file into a pointer array
float* load_csv(const string& filepath, int& rows, int& cols, bool normalize = false) {
    ifstream file(filepath);
    if (!file.is_open()) throw runtime_error("Cannot open file: " + filepath);

    vector<vector<float>> data;
    string line;

    while (getline(file, line)) {
        vector<float> row;
        stringstream ss(line);
        string value;

        while (getline(ss, value, ',')) {
            float val = stof(value);
            if (normalize) val /= 255.0; // Normalize pixel values to [0, 1]
            row.push_back(val);
        }
        data.push_back(row);
    }

    rows = data.size();
    cols = data[0].size();
    float* result = new float[rows * cols];
    for (int i = 0; i < rows; ++i) {
        memcpy(result + i * cols, data[i].data(), cols * sizeof(float));
    }

    file.close();
    return result;
}

// Function to load labels and convert to one-hot encoding
float* load_labels_csv(const string& filepath, int& rows, int num_classes = 10) {
    ifstream file(filepath);
    if (!file.is_open()) throw runtime_error("Cannot open file: " + filepath);

    vector<vector<float>> labels;
    string line;

    while (getline(file, line)) {
        int label = stoi(line); // Read the label
        vector<float> one_hot(num_classes, 0.0);
        one_hot[label] = 1.0; // One-hot encode
        labels.push_back(one_hot);
    }

    rows = labels.size();
    float* result = new float[rows * num_classes];
    for (int i = 0; i < rows; ++i) {
        memcpy(result + i * num_classes, labels[i].data(), num_classes * sizeof(float));
    }

    file.close();
    return result;
}

// Main function to load and preprocess the dataset
void load_and_preprocess_dataset_csv(const string& train_images_path,
    const string& train_labels_path,
    const string& test_images_path,
    const string& test_labels_path,
    float*& train_images,
    float*& train_labels,
    float*& test_images,
    float*& test_labels,
    int& train_image_rows, int& train_label_rows,
    int& test_image_rows, int& test_label_rows,
    int& image_cols, int& num_classes) {
    try {
        // Load train images and normalize
        train_images = load_csv(train_images_path, train_image_rows, image_cols, true);

        // Load train labels and one-hot encode
        train_labels = load_labels_csv(train_labels_path, train_label_rows, num_classes);

        // Load test images and normalize
        test_images = load_csv(test_images_path, test_image_rows, image_cols, true);

        // Load test labels and one-hot encode
        test_labels = load_labels_csv(test_labels_path, test_label_rows, num_classes);

        // Verify consistency of data
        if (train_image_rows != train_label_rows) {
            throw runtime_error("Mismatch between number of train images and labels.");
        }
        if (test_image_rows != test_label_rows) {
            throw runtime_error("Mismatch between number of test images and labels.");
        }

        cout << "Dataset loaded successfully:" << endl;
        cout << " - Train samples: " << train_image_rows << endl;
        cout << " - Test samples: " << test_image_rows << endl;

    }
    catch (const exception& ex) {
        cerr << "Error while loading dataset: " << ex.what() << endl;
    }
}

// Activation functions
inline float relu(float x) { return max(0.0f, x); }
inline float relu_derivative(float x) { return x > 0 ? 1.0f : 0.0f; }

void softmax(float* outputs, int size) {
    float max_val = *max_element(outputs, outputs + size);
    float sum = 0.0;
    for (int i = 0; i < size; ++i) {
        outputs[i] = exp(outputs[i] - max_val); // Subtract max_val for numerical stability
        sum += outputs[i];
    }
    for (int i = 0; i < size; ++i) outputs[i] /= sum;
}

// Cross-entropy loss
float cross_entropy_loss(const float* predictions, const float* labels, int size) {
    float loss = 0.0;
    for (int i = 0; i < size; ++i) {
        loss -= labels[i] * log(predictions[i] + 1e-15); // Avoid log(0)
    }
    return loss;
}
// CUDA kernel for forward pass
// Define constant memory for biases
__constant__ float c_biases[128]; // Adjust size as needed

// CUDA kernel for forward pass
__global__ void forward_kernel1(const float* inputs, float* outputs, const float* weights, int input_size, int output_size, int batch_size) {
    extern __shared__ float shared_inputs[];

    int sample_idx = blockIdx.x;
    int neuron_idx = threadIdx.x;
    if (threadIdx.x < input_size) {
        shared_inputs[threadIdx.x] = inputs[sample_idx * input_size + threadIdx.x];
    }
    __syncthreads();
    if (sample_idx < batch_size && neuron_idx < output_size) {
        float sum = c_biases[neuron_idx];
        for (int j = 0; j < input_size; ++j) {
            sum += shared_inputs[j] * weights[neuron_idx * input_size + j];
        }
        outputs[sample_idx * output_size + neuron_idx] = sum;
    }
}
// ...existing code...
__global__ void backward_kernel(const float* output_gradients, const float* weights,
                                 float* input_gradients, float* weight_gradients,
                                 float* bias_gradients, const float* inputs,
                                 int input_size, int output_size, int batch_size) {
    extern __shared__ float shared_output_gradients[];

    int sample = blockIdx.y;
    int input_idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (threadIdx.x < output_size) {
        shared_output_gradients[threadIdx.x] = output_gradients[sample * output_size + threadIdx.x];
    }
    __syncthreads();

    if (input_idx < input_size && sample < batch_size) {
        float gradient = 0.0;
        for (int i = 0; i < output_size; ++i) {
            float output_gradient = shared_output_gradients[i];
            gradient += output_gradient * weights[i * input_size + input_idx];

            // Accumulate weight gradients
            atomicAdd(&weight_gradients[i * input_size + input_idx], output_gradient * inputs[sample * input_size + input_idx]);

            // Accumulate bias gradients
            if (threadIdx.x == 0) atomicAdd(&bias_gradients[i], output_gradient);
        }
        gradient *= (inputs[sample * input_size + input_idx] > 0) ? 1.0f : 0.0f; // relu_derivative
        // Accumulate input gradients
        atomicAdd(&input_gradients[sample * input_size + input_idx], gradient);
    }
}
__global__ void backward_kernel2(const float* output_gradients, const float* weights,
                                 float* input_gradients, float* weight_gradients,
                                 float* bias_gradients, const float* inputs,
                                 int input_size, int output_size, int batch_size) {
    extern __shared__ float shared_output_gradients[];

    int sample = blockIdx.y;
    int input_idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (threadIdx.x < output_size) {
        shared_output_gradients[threadIdx.x] = output_gradients[sample * output_size + threadIdx.x];
    }
    __syncthreads();

    if (input_idx < input_size && sample < batch_size) {
        float gradient = 0.0;
        for (int i = 0; i < output_size; ++i) {
            float output_gradient = shared_output_gradients[i];
            gradient += output_gradient * weights[i * input_size + input_idx];

            // Accumulate weight gradients
            atomicAdd(&weight_gradients[i * input_size + input_idx], output_gradient * inputs[sample * input_size + input_idx]);
        }
        gradient *= (inputs[sample * input_size + input_idx] > 0) ? 1.0f : 0.0f; // relu_derivative
        // Accumulate input gradients
        atomicAdd(&input_gradients[sample * input_size + input_idx], gradient);
    }

    if (threadIdx.x == 0) {
        for (int i = 0; i < output_size; ++i) {
            atomicAdd(&bias_gradients[i], shared_output_gradients[i]);
        }
    }
}

__global__ void backward_kernel1(const float* batch_output_gradients, const float* inputs, const float* weights, 
                                float* weight_gradients, float* bias_gradients, float* batch_input_gradients, 
                                int input_size, int output_size, int batch_size) {
    extern __shared__ float shared_mem[];
    float* shared_inputs = shared_mem;
    float* shared_output_gradients = shared_mem + input_size;

    int sample = blockIdx.x;
    int i = threadIdx.x;
    // for (int j=0;j<(input_size+output_size-1)/output_size;++j)
    // {
    //     int k=i*((input_size+output_size-1)/output_size)+j;
    //     if (k < input_size) {
    //         shared_inputs[k] = inputs[sample * input_size + k];
    //     }
    // }
    if (i < input_size) {
        shared_inputs[i] = inputs[sample * input_size + i];
    }
    if (i < output_size) {
        shared_output_gradients[i] = batch_output_gradients[sample * output_size + i];
    }
    __syncthreads();

    if (sample < batch_size && i < output_size) {
        for (int j = 0; j < input_size; ++j) {
            atomicAdd(&weight_gradients[i * input_size + j], shared_output_gradients[i] * shared_inputs[j]);
        }
        atomicAdd(&bias_gradients[i], shared_output_gradients[i]);

        float gradient = 0.0;
        for (int k = 0; k < output_size; ++k) {
            gradient += shared_output_gradients[k] * weights[k * input_size + i];
        }
        batch_input_gradients[sample * input_size + i] = gradient * (shared_inputs[i] > 0 ? 1.0f : 0.0f); // relu_derivative
    }
}
// ...existing code...

// CUDA kernel for weight updates
__global__ void update_weights_kernel(float* weights, float* biases, const float* weight_gradients,
                                       const float* bias_gradients, float learning_rate,
                                       int input_size, int output_size, int batch_size) {
    int neuron = threadIdx.x + blockIdx.x * blockDim.x;

    if (neuron < output_size) {
        for (int i = 0; i < input_size;++i) {
            weights[neuron * input_size + i] -= learning_rate * weight_gradients[neuron * input_size + i] / batch_size;
        }
        biases[neuron] -= learning_rate * bias_gradients[neuron] / batch_size;
    }
}
// Updated Layer class to handle batches
class Layer {
public:
    float* weights;
    float* biases;
    float* outputs = nullptr;
    float* inputs;
    float* weight_gradients;
    float* bias_gradients;
    float* d_input;
    float* d_output;
    float* d_weights;
    float* d_weight_gradients;
    float* d_bias_gradients;
    float* d_batch_input_gradients;
    float* d_biases;
    int input_size, output_size;

    Layer(int input_size, int output_size)
        : input_size(input_size), output_size(output_size) {

        weights = new float[input_size * output_size];
        biases = new float[output_size]();
        weight_gradients = new float[input_size * output_size]();
        bias_gradients = new float[output_size]();
        memset(bias_gradients, 0, output_size * sizeof(float));
        memset(weight_gradients, 0, input_size * output_size * sizeof(float));
        initialize_weights();

    }

    ~Layer() {
        delete[] inputs;
        delete[] outputs;
        delete[] weights;
        delete[] biases;
        delete[] weight_gradients;
        delete[] bias_gradients;

        CHECK(cudaFree(d_input));
        CHECK(cudaFree(d_output));
        CHECK(cudaFree(d_weights));
        CHECK(cudaFree(d_biases));
        CHECK(cudaFree(d_weight_gradients));
        CHECK(cudaFree(d_bias_gradients));
        CHECK(cudaFree(d_batch_input_gradients));

    }

    void initialize_weights() {
        random_device rd;
        mt19937 gen(rd());
        normal_distribution<> dist(0.0f, 0.1f);
        for (int i = 0; i < input_size * output_size; ++i) {
            weights[i] = dist(gen);
        }
    }
    void cuda_malloc(int batch_size) {
        CHECK(cudaMalloc(&d_biases, output_size * sizeof(float)));
        CHECK(cudaMalloc(&d_input, batch_size * input_size * sizeof(float)));
        CHECK(cudaMalloc(&d_output, batch_size * output_size * sizeof(float)));
        CHECK(cudaMalloc(&d_weights, input_size * output_size * sizeof(float)));
        CHECK(cudaMalloc(&d_weight_gradients, input_size * output_size * sizeof(float)));
        CHECK(cudaMalloc(&d_bias_gradients, output_size * sizeof(float)));
        CHECK(cudaMalloc(&d_batch_input_gradients, batch_size * input_size * sizeof(float)));
    }
    void forward(const float* batch_input, int batch_size) {
       // if (inputs != nullptr) delete[] inputs;
        inputs = new float[batch_size * input_size];
        memcpy(inputs, batch_input, batch_size * input_size * sizeof(float)); // Copy input data

        CHECK(cudaMemcpy(d_input, inputs, batch_size * input_size * sizeof(float), cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(d_weights, weights, input_size * output_size * sizeof(float), cudaMemcpyHostToDevice));
        CHECK(cudaMemcpyToSymbol(c_biases, biases, output_size * sizeof(float)));

        int threads = input_size;
        int blocks = batch_size;
        int shared_memory_size = input_size * sizeof(float);

        forward_kernel1<<<blocks, threads, shared_memory_size>>>(d_input, d_output, d_weights, input_size, output_size, batch_size);

        CHECK(cudaGetLastError());
        CHECK(cudaDeviceSynchronize());
        CHECK(cudaGetLastError());

        if (outputs == nullptr) outputs = new float[batch_size * output_size];
        CHECK(cudaMemcpy(outputs, d_output, batch_size * output_size * sizeof(float), cudaMemcpyDeviceToHost));
    }

    void backward(const float* batch_output_gradients, int batch_size) {
        float *d_batch_output_gradients;
        CHECK(cudaMalloc(&d_batch_output_gradients, batch_size * output_size * sizeof(float)));
        CHECK(cudaMemcpy(d_batch_output_gradients, batch_output_gradients, batch_size * output_size * sizeof(float), cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(d_input, inputs, batch_size * input_size * sizeof(float), cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(d_weights, weights, input_size * output_size * sizeof(float), cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(d_weight_gradients, weight_gradients, input_size * output_size * sizeof(float), cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(d_bias_gradients, bias_gradients, output_size * sizeof(float), cudaMemcpyHostToDevice));
        CHECK(cudaMemset(d_batch_input_gradients, 0, batch_size * input_size * sizeof(float)));

        dim3 blocks(1,batch_size);
        dim3 threads(input_size);

        backward_kernel2<<<blocks, threads, output_size * sizeof(float)>>>(d_batch_output_gradients, d_weights, d_batch_input_gradients, d_weight_gradients, d_bias_gradients, d_input, input_size, output_size, batch_size);

        CHECK(cudaGetLastError());
        CHECK(cudaDeviceSynchronize());

        CHECK(cudaMemcpy(weight_gradients, d_weight_gradients, input_size * output_size * sizeof(float), cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(bias_gradients, d_bias_gradients, output_size * sizeof(float), cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(inputs, d_batch_input_gradients, batch_size * input_size * sizeof(float), cudaMemcpyDeviceToHost));

        CHECK(cudaFree(d_batch_output_gradients));
    }

    void update_weights(float learning_rate, int batch_size) {
   
       
        CHECK(cudaMemcpy(d_weights, weights, input_size * output_size * sizeof(float), cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(d_biases, biases, output_size * sizeof(float), cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(d_weight_gradients, weight_gradients, input_size * output_size * sizeof(float), cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(d_bias_gradients, bias_gradients, output_size * sizeof(float), cudaMemcpyHostToDevice));

        dim3 threads(128);
        dim3 blocks((output_size + threads.x - 1) / threads.x);
        update_weights_kernel<<<blocks, threads>>>(d_weights, d_biases, d_weight_gradients, d_bias_gradients, learning_rate, input_size, output_size, batch_size);

        CHECK(cudaDeviceSynchronize());

        CHECK(cudaMemcpy(weights, d_weights, input_size * output_size * sizeof(float), cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(biases, d_biases, output_size * sizeof(float), cudaMemcpyDeviceToHost));

        memset(weight_gradients, 0, input_size * output_size * sizeof(float));
        memset(bias_gradients, 0, output_size * sizeof(float));
    }
};

class NeuralNetwork {
public:
    Layer** layers;
    int num_layers;

    NeuralNetwork(const vector<int>& architecture) {
        num_layers = architecture.size() - 1;
        layers = new Layer * [num_layers];
        for (int i = 0; i < num_layers; ++i) {
            layers[i] = new Layer(architecture[i], architecture[i + 1]);
        }
    }

    ~NeuralNetwork() {
        for (int i = 0; i < num_layers; ++i) {
            delete layers[i];
        }
        delete[] layers;
    }

    void cuda_malloc(int batch_size) {
        for (int i = 0; i < num_layers; ++i) {
            layers[i]->cuda_malloc(batch_size);
        }
    }
    float* forward(const float* batch_input, int batch_size) {
        float* activations = const_cast<float*>(batch_input);
        for (int i = 0; i < num_layers; ++i) {
            layers[i]->forward(activations, batch_size);
            activations = layers[i]->outputs;
            if (i < num_layers - 1) {
                for (int j = 0; j < batch_size * layers[i]->output_size; ++j) {
                    activations[j] = relu(activations[j]);
                }
            }
            else {
                for (int j = 0; j < batch_size; ++j) {
                    softmax(activations + j * layers[i]->output_size, layers[i]->output_size);
                }
            }
        }
        return activations;
    }

    void backward(const float* predictions, const float* labels, int batch_size, int num_classes) {
        float* batch_output_gradients = new float[batch_size * num_classes];
        for (int sample = 0; sample < batch_size; ++sample) {
            for (int i = 0; i < num_classes; ++i) {
                batch_output_gradients[sample * num_classes + i] = predictions[sample * num_classes + i] - labels[sample * num_classes + i];
            }
        }

        for (int i = num_layers - 1; i >= 0; --i) {
            layers[i]->backward(batch_output_gradients, batch_size);
            batch_output_gradients = layers[i]->inputs;
        }
        //delete[] batch_output_gradients;
    }

    void update_weights(float learning_rate, int batch_size) {
        for (int i = 0; i < num_layers; ++i) {
            layers[i]->update_weights(learning_rate, batch_size);
        }
    }

    void train(const float* data, const float* labels, int num_samples, int batch_size, int num_classes, int epochs, float learning_rate) {
        cuda_malloc(batch_size);
        for (int epoch = 1; epoch <= epochs; ++epoch) {
            float total_loss = 0.0;
            for (int i = 0; i < num_samples; i += batch_size) {
                int current_batch_size = min(batch_size, num_samples - i);
                const float* batch_data = data + i * layers[0]->input_size;
                const float* batch_labels = labels + i * num_classes;

                float* predictions = forward(batch_data, current_batch_size);
                for (int j = 0; j < current_batch_size; ++j) {
                    total_loss += cross_entropy_loss(predictions + j * num_classes, batch_labels + j * num_classes, num_classes);
                }
                backward(predictions, batch_labels, current_batch_size, num_classes);
                update_weights(learning_rate, current_batch_size);
            }
            cout << "Epoch " << epoch << " - Loss: " << total_loss / num_samples << endl;
        }
    }

    // Save the model to a file
    void save_model(const string& filepath) {
        ofstream file(filepath, ios::binary);
        if (!file.is_open()) throw runtime_error("Cannot open file to save model: " + filepath);

        // Save the number of layers
        file.write(reinterpret_cast<char*>(&num_layers), sizeof(num_layers));

        for (int i = 0; i < num_layers; ++i) {
            Layer* layer = layers[i];

            // Save input_size and output_size
            file.write(reinterpret_cast<char*>(&layer->input_size), sizeof(layer->input_size));
            file.write(reinterpret_cast<char*>(&layer->output_size), sizeof(layer->output_size));

            // Save weights
            file.write(reinterpret_cast<char*>(layer->weights), layer->input_size * layer->output_size * sizeof(float));

            // Save biases
            file.write(reinterpret_cast<char*>(layer->biases), layer->output_size * sizeof(float));
        }

        file.close();
        cout << "Model saved to " << filepath << endl;
    }

    // Load the model from a file
    void load_model(const string& filepath) {
        ifstream file(filepath, ios::binary);
        if (!file.is_open()) throw runtime_error("Cannot open file to load model: " + filepath);

        // Read the number of layers
        int saved_num_layers;
        file.read(reinterpret_cast<char*>(&saved_num_layers), sizeof(saved_num_layers));

        // Ensure the architecture matches the saved model
        if (saved_num_layers != num_layers) {
            throw runtime_error("Saved model architecture does not match the current network.");
        }

        for (int i = 0; i < num_layers; ++i) {
            Layer* layer = layers[i];

            // Read input_size and output_size
            int saved_input_size, saved_output_size;
            file.read(reinterpret_cast<char*>(&saved_input_size), sizeof(saved_input_size));
            file.read(reinterpret_cast<char*>(&saved_output_size), sizeof(saved_output_size));

            if (saved_input_size != layer->input_size || saved_output_size != layer->output_size) {
                throw runtime_error("Layer dimensions do not match the saved model.");
            }

            // Load weights
            file.read(reinterpret_cast<char*>(layer->weights), layer->input_size * layer->output_size * sizeof(float));

            // Load biases
            file.read(reinterpret_cast<char*>(layer->biases), layer->output_size * sizeof(float));
        }

        file.close();
        cout << "Model loaded from " << filepath << endl;
    }
};



int main() {
    float* train_images;
    float* train_labels;
    float* test_images;
    float* test_labels;

    int train_image_rows, train_label_rows;
    int test_image_rows, test_label_rows;
    int image_cols, num_classes = 10;

    const string train_images_path = "x_train.csv";
    const string train_labels_path = "y_train.csv";
    const string test_images_path = "x_test.csv";
    const string test_labels_path = "y_test.csv";

    // Load and preprocess the dataset
    load_and_preprocess_dataset_csv(train_images_path, train_labels_path, test_images_path, test_labels_path,
        train_images, train_labels, test_images, test_labels,
        train_image_rows, train_label_rows, test_image_rows, test_label_rows,
        image_cols, num_classes);

    // Define network architecture and create neural network
    vector<int> architecture = { image_cols, 128, 128, num_classes };
    NeuralNetwork nn(architecture);

    GpuTimer timer;
    timer.Start();
    nn.train(train_images, train_labels, train_image_rows, 16, num_classes, 10, 0.01f);
    timer.Stop();
    float time = timer.Elapsed();
cout << "Processing time: " << time << " ms" << endl;
    nn.save_model("model.bin");
    NeuralNetwork nn1(architecture);

    nn1.load_model("model.bin");

    // Evaluate accuracy
    nn1.cuda_malloc(test_image_rows);
    float* test_predictions = nn1.forward(test_images, test_image_rows);
    int correct = 0;
    for (int i = 0; i < test_image_rows; ++i) {
        int predicted = max_element(test_predictions + i * num_classes, test_predictions + (i + 1) * num_classes) - (test_predictions + i * num_classes);
        int actual = max_element(test_labels + i * num_classes, test_labels + (i + 1) * num_classes) - (test_labels + i * num_classes);
        if (predicted == actual) ++correct;
    }
    cout << "Test Accuracy: " << (correct / static_cast<float>(test_image_rows)) * 100.0f << "%" << endl;

    // Clean up dynamically allocated memory
    delete[] train_images;
    delete[] train_labels;
    delete[] test_images;
    delete[] test_labels;

    return 0;
}