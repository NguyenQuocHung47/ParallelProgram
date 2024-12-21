#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <stdexcept>
#include <algorithm>
#include <cmath>
#include <random>

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
// Function to load CSV data
vector<vector<float>> load_csv(const string& filepath, bool normalize = false) {
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
            if (normalize) val /= 255.0f; // Normalize pixel values to [0, 1] 
            row.push_back(val);
        }
        data.push_back(row);
    }

    file.close();
    return data;
}

// Function to load labels and convert to one-hot encoding
vector<vector<float>> load_labels_csv(const string& filepath, int num_classes = 10) {
    ifstream file(filepath);
    if (!file.is_open()) throw runtime_error("Cannot open file: " + filepath);

    vector<vector<float>> labels;
    string line;

    while (getline(file, line)) {
        int label = stoi(line); // Read the label
        vector<float> one_hot(num_classes, 0.0f);
        one_hot[label] = 1.0f; // One-hot encode
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
    vector<vector<float>>& train_images,
    vector<vector<float>>& train_labels,
    vector<vector<float>>& test_images,
    vector<vector<float>>& test_labels) {
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

// Activation functions
inline float relu(float x) { return max(0.0f, x); }
inline float relu_derivative(float x) { return x > 0 ? 1.0f : 0.0f; }

void softmax(vector<float>& outputs) {
    float max_val = *max_element(outputs.begin(), outputs.end());
    float sum = 0.0f;
    for (float& val : outputs) {
        val = expf(val - max_val); // Subtract max_val for numerical stability
        sum += val;
    }
    for (float& val : outputs) val /= sum;
}

// Cross-entropy loss
float cross_entropy_loss(const vector<float>& predictions, const vector<float>& labels) {
    float loss = 0.0f;
    for (size_t i = 0; i < predictions.size(); ++i) {
        loss -= labels[i] * logf(predictions[i] + 1e-15f); // Avoid log(0)
    }
    return loss;
}

// Updated Layer class to handle batches
__global__ void forward_kernel(float* d_inputs, float* d_weights, float* d_biases, float* d_outputs, 
                               int input_size, int output_size, int batch_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size * output_size) {
        int sample = idx / output_size;
        int neuron = idx % output_size;

        float sum = d_biases[neuron];
        for (int j = 0; j < input_size; ++j) {
            sum += d_inputs[sample * input_size + j] * d_weights[neuron * input_size + j];
        }
        d_outputs[sample * output_size + neuron] = sum;
    }
}

__global__ void backward_kernel(float* d_output_gradients, float* d_inputs, float* d_weights, 
                                float* d_weight_gradients, float* d_bias_gradients, float* d_input_gradients,
                                int input_size, int output_size, int batch_size) {
    // Calculate weight gradients and bias gradients
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size * output_size) {
        int sample = idx / output_size;
        int neuron = idx % output_size;

        for (int j = 0; j < input_size; ++j) {
            atomicAdd(&d_weight_gradients[neuron * input_size + j], 
                      d_output_gradients[sample * output_size + neuron] * d_inputs[sample * input_size + j]);
        }
        atomicAdd(&d_bias_gradients[neuron], d_output_gradients[sample * output_size + neuron]);
    }

    // Calculate input gradients
    if (idx < batch_size * input_size) {
        int sample = idx / input_size;
        int input_idx = idx % input_size;

        float gradient = 0.0f;
        for (int neuron = 0; neuron < output_size; ++neuron) {
            gradient += d_output_gradients[sample * output_size + neuron] * d_weights[neuron * input_size + input_idx];
        }
        d_input_gradients[sample * input_size + input_idx] = gradient;
    }
}

class Layer {
public:
    vector<vector<float>> weights, weight_gradients;
    vector<float> biases, bias_gradients;
    vector<vector<float>> outputs, inputs;

    float *d_weights, *d_biases, *d_inputs, *d_outputs, *d_weight_gradients, *d_bias_gradients;

    int input_size, output_size;

    Layer(int input_size, int output_size)
        : input_size(input_size), output_size(output_size) {
        weights = vector<vector<float>>(output_size, vector<float>(input_size));
        biases = vector<float>(output_size, 0.0f);
        weight_gradients = vector<vector<float>>(output_size, vector<float>(input_size, 0.0f));
        bias_gradients = vector<float>(output_size, 0.0f);

        initialize_weights();
        allocate_device_memory();
    }

    ~Layer() {
        free_device_memory();
    }

    void initialize_weights() {
        random_device rd;
        mt19937 gen(rd());
        normal_distribution<> dist(0.0, 0.1);
        for (auto& row : weights)
            for (auto& weight : row)
                weight = dist(gen);
    }

    void allocate_device_memory() {
        cudaMalloc(&d_weights, input_size * output_size * sizeof(float));
        cudaMalloc(&d_biases, output_size * sizeof(float));
        cudaMalloc(&d_inputs, input_size * output_size * sizeof(float));
        cudaMalloc(&d_outputs, input_size * output_size * sizeof(float));
        cudaMalloc(&d_weight_gradients, input_size * output_size * sizeof(float));
        cudaMalloc(&d_bias_gradients, output_size * sizeof(float));
    }

    void free_device_memory() {
        cudaFree(d_weights);
        cudaFree(d_biases);
        cudaFree(d_inputs);
        cudaFree(d_outputs);
        cudaFree(d_weight_gradients);
        cudaFree(d_bias_gradients);
    }

    void forward(const vector<vector<float>>& batch_input) {
        int batch_size = batch_input.size();

        vector<float> flattened_inputs(batch_size * input_size);
        for (int i = 0; i < batch_size; ++i) {
            copy(batch_input[i].begin(), batch_input[i].end(), flattened_inputs.begin() + i * input_size);
        }

        CHECK(cudaMemcpy(d_inputs, flattened_inputs.data(), batch_size * input_size * sizeof(float), cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(d_weights, &weights[0][0], input_size * output_size * sizeof(float), cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(d_biases, biases.data(), output_size * sizeof(float), cudaMemcpyHostToDevice));

        int num_threads = 32;
        int num_blocks = (batch_size * output_size + num_threads - 1) / num_threads;
        forward_kernel<<<num_blocks, num_threads>>>(d_inputs, d_weights, d_biases, d_outputs, input_size, output_size, batch_size);
        CHECK(cudaGetLastError());
        cudaDeviceSynchronize();
        vector<float> flattened_outputs(batch_size * output_size);
        CHECK(cudaMemcpy(flattened_outputs.data(), d_outputs, batch_size * output_size * sizeof(float), cudaMemcpyDeviceToHost));

        // Reshape flattened_outputs back into batch format
        outputs.assign(batch_size, vector<float>(output_size));
        for (int i = 0; i < batch_size; ++i) {
            copy(flattened_outputs.begin() + i * output_size, flattened_outputs.begin() + (i + 1) * output_size, outputs[i].begin());
        }
    }

    void backward(const vector<vector<float>>& batch_output_gradients, float learning_rate) {
        int batch_size = batch_output_gradients.size();
        vector<float> flattened_gradients(batch_size * output_size);

        for (int i = 0; i < batch_size; ++i) {
            copy(batch_output_gradients[i].begin(), batch_output_gradients[i].end(), flattened_gradients.begin() + i * output_size);
        }

        CHECK(cudaMemcpy(d_outputs, flattened_gradients.data(), batch_size * output_size * sizeof(float), cudaMemcpyHostToDevice));

        int num_threads = 32;
        int num_blocks = (batch_size * output_size + num_threads - 1) / num_threads;
        backward_kernel<<<num_blocks, num_threads>>>(d_outputs, d_inputs, d_weights, d_weight_gradients, d_bias_gradients, d_inputs,
                                                     input_size, output_size, batch_size);
        CHECK(cudaGetLastError());                                             
        cudaDeviceSynchronize();
        CHECK(cudaMemcpy(&weights[0][0], d_weights, input_size * output_size * sizeof(float), cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(biases.data(), d_biases, output_size * sizeof(float), cudaMemcpyDeviceToHost));
    }
    void update_weights(const float& learning_rate) {
        for (size_t i = 0; i < weights.size(); ++i) {
            for (size_t j = 0; j < weights[i].size(); ++j) {
                weights[i][j] -= learning_rate * weight_gradients[i][j] / inputs.size(); // Normalize by batch size
                weight_gradients[i][j] = 0.0f;
            }
            biases[i] -= learning_rate * bias_gradients[i] / inputs.size(); // Normalize by batch size
            bias_gradients[i] = 0.0f;
        }
    }
};

// NeuralNetwork class
class NeuralNetwork {
public:
    vector<Layer> layers;

    NeuralNetwork(const vector<int>& architecture) {
        for (size_t i = 1; i < architecture.size(); ++i) {
            layers.emplace_back(architecture[i - 1], architecture[i]);
        }
    }

    vector<vector<float>> forward(const vector<vector<float>>& batch_input) {
        vector<vector<float>> activations = batch_input;
        for (size_t i = 0; i < layers.size(); ++i) {
            layers[i].forward(activations);
            activations = layers[i].outputs;
            if (i < layers.size() - 1) {
                for (auto& sample : activations) {
                    transform(sample.begin(), sample.end(), sample.begin(), relu);
                }
            }
            else {
                for (auto& sample : activations) {
                    softmax(sample);
                }
            }
        }
        return activations;
    }

    void backward(const vector<vector<float>>& predictions, const vector<vector<float>>& labels, float learning_rate) {
        vector<vector<float>> batch_output_gradients(predictions.size(), vector<float>(predictions[0].size()));
        for (size_t sample = 0; sample < predictions.size(); ++sample) {
            for (size_t i = 0; i < predictions[sample].size(); ++i) {
                batch_output_gradients[sample][i] = predictions[sample][i] - labels[sample][i];
            }
        }
        for (int i = layers.size() - 1; i >= 0; --i) {
            layers[i].backward(batch_output_gradients, learning_rate);
            batch_output_gradients = layers[i].inputs;
        }
    }

    void train(const vector<vector<float>>& data, const vector<vector<float>>& labels, int epochs, int batch_size, float learning_rate) {
        for (int epoch = 1; epoch <= epochs; ++epoch) {
            float total_loss = 0.0f;
            for (size_t i = 0; i < data.size(); i += batch_size) {
                size_t batch_end = min(i + batch_size, data.size());
                vector<vector<float>> batch_data(data.begin() + i, data.begin() + batch_end);
                vector<vector<float>> batch_labels(labels.begin() + i, labels.begin() + batch_end);

                vector<vector<float>> predictions = forward(batch_data);
                for (size_t j = 0; j < batch_data.size(); ++j) {
                    total_loss += cross_entropy_loss(predictions[j], batch_labels[j]);
                }
                backward(predictions, batch_labels, learning_rate);

                for (int i = layers.size() - 1; i >= 0; --i) {
                    layers[i].update_weights(learning_rate);
                }
            }
            cout << "Epoch " << epoch << " - Loss: " << total_loss / data.size() << endl;
        }
    }

    vector<float> predict(const vector<float>& input) {
        vector<float> activations = input;
        for (size_t i = 0; i < layers.size(); ++i) {
            layers[i].forward({ activations }); // Wrap input in a batch of size 1
            activations = layers[i].outputs[0]; // Extract the single sample's output
            if (i < layers.size() - 1) {
                transform(activations.begin(), activations.end(), activations.begin(), relu);
            }
            else {
                softmax(activations);
            }
        }
        return activations;
    }
};

// Function to evaluate the model on a dataset
float evaluate(NeuralNetwork& model, const vector<vector<float>>& images,
    const vector<vector<float>>& labels) {
    int correct = 0;

    for (size_t i = 0; i < images.size(); ++i) {
        vector<float> predicted = model.predict(images[i]);
        int predicted_class = distance(predicted.begin(), max_element(predicted.begin(), predicted.end()));
        int true_class = distance(labels[i].begin(), max_element(labels[i].begin(), labels[i].end()));

        if (predicted_class == true_class) {
            ++correct;
        }
    }

    return static_cast<float>(correct) / images.size(); // Return accuracy as a fraction
}

// Function to save the model to a file
void save_model(const NeuralNetwork& model, const string& filepath) {
    ofstream file(filepath, ios::out | ios::binary);
    if (!file.is_open()) throw runtime_error("Cannot open file to save model: " + filepath);

    for (const auto& layer : model.layers) {
        // Save weights
        for (const auto& row : layer.weights) {
            for (float weight : row) {
                file.write(reinterpret_cast<const char*>(&weight), sizeof(weight));
            }
        }

        // Save biases
        for (float bias : layer.biases) {
            file.write(reinterpret_cast<const char*>(&bias), sizeof(bias));
        }
    }

    file.close();
    cout << "Model saved to " << filepath << endl;
}

// Function to load the model from a file
void load_model(NeuralNetwork& model, const string& filepath) {
    ifstream file(filepath, ios::in | ios::binary);
    if (!file.is_open()) throw runtime_error("Cannot open file to load model: " + filepath);

    for (auto& layer : model.layers) {
        // Load weights
        for (auto& row : layer.weights) {
            for (float& weight : row) {
                file.read(reinterpret_cast<char*>(&weight), sizeof(weight));
            }
        }

        // Load biases
        for (float& bias : layer.biases) {
            file.read(reinterpret_cast<char*>(&bias), sizeof(bias));
        }
    }

    file.close();
    cout << "Model loaded from " << filepath << endl;
}

// Helper function to display an image (assumes 28x28 grayscale)
void display_image(const vector<float>& image) {
    const int size = 28;
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            if (image[i * size + j] > 0.5f) cout << "##"; // Threshold for visual representation
            else cout << "..";
        }
        cout << endl;
    }
}

// Demonstration: Predict and display the first 10 samples
void demonstrate_predictions(NeuralNetwork& model, const vector<vector<float>>& images,
    const vector<vector<float>>& labels) {
    const int num_samples = 20;

    for (int i = 0; i < num_samples; ++i) {
        const auto& image = images[i];
        const auto& label = labels[i];

        vector<float> predicted = model.predict(image);
        int predicted_class = distance(predicted.begin(), max_element(predicted.begin(), predicted.end()));
        int true_class = distance(label.begin(), max_element(label.begin(), label.end()));

        cout << "Sample " << i + 1 << ":" << endl;
        display_image(image);
        cout << "Predicted Label: " << predicted_class << ", Actual Label: " << true_class << endl;
        cout << "-----------------------------" << endl;
    }
}

int main() {

    vector<vector<float>> train_images, train_labels;
    vector<vector<float>> test_images, test_labels;

    const string train_images_path = "x_train.csv";
    const string train_labels_path = "y_train.csv";
    const string test_images_path = "x_test.csv";
    const string test_labels_path = "y_test.csv";

    const string model_path = "nn_model.bin";

    // Load and preprocess the dataset
    load_and_preprocess_dataset_csv(train_images_path, train_labels_path, test_images_path, test_labels_path,
        train_images, train_labels, test_images, test_labels);

    //// Example setup: Define the architecture
    NeuralNetwork nn({ 784, 128, 128, 10 });
    //// Load and preprocess Fashion MNIST dataset here.
    //// Use nn.train() to train the network.
    nn.train(train_images, train_labels, 10, 32, 0.01f);

    //// Save the trained model
    save_model(nn, model_path);

    //// Load the model to demonstrate persistence
    NeuralNetwork loaded_nn({ 784, 128, 128, 10 });
    load_model(loaded_nn, model_path);

    double train_accuracy = evaluate(loaded_nn, train_images, train_labels);
    double test_accuracy = evaluate(loaded_nn, test_images, test_labels);

    cout << "Train Accuracy: " << train_accuracy * 100.0f << "%" << endl;
    cout << "Test Accuracy: " << test_accuracy * 100.0f << "%" << endl;

    // Demonstrate predictions
    demonstrate_predictions(loaded_nn, test_images, test_labels);

    return 0;
}
