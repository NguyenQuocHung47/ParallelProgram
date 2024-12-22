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
        test_images = load_csv(test_images_path, test_image_rows, image_cols, false);

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

// Updated Layer class to handle batches
class Layer {
public:
    float* weights;
    float* biases;
    float* outputs;
    float* inputs;
    float* weight_gradients;
    float* bias_gradients;

    int input_size, output_size;

    Layer(int input_size, int output_size)
        : input_size(input_size), output_size(output_size) {

        weights = new float[input_size * output_size];
        biases = new float[output_size]();
        weight_gradients = new float[input_size * output_size]();
        bias_gradients = new float[output_size]();

        initialize_weights();
    }

    ~Layer() {
        delete[] weights;
        delete[] biases;
        delete[] weight_gradients;
        delete[] bias_gradients;
    }

    void initialize_weights() {
        random_device rd;
        mt19937 gen(rd());
        normal_distribution<> dist(0.0f, 0.1f);
        for (int i = 0; i < input_size * output_size; ++i) {
            weights[i] = dist(gen);
        }
    }

    void forward(const float* batch_input, int batch_size) {
        if (inputs) delete[] inputs; 
        inputs = new float[batch_size * input_size];
        memcpy(inputs, batch_input, batch_size * input_size * sizeof(float)); // Copy input data
        outputs = new float[batch_size * output_size]();

        for (int sample = 0; sample < batch_size; ++sample) {
            for (int i = 0; i < output_size; ++i) {
                outputs[sample * output_size + i] = biases[i];
                for (int j = 0; j < input_size; ++j) {
                    outputs[sample * output_size + i] += batch_input[sample * input_size + j] * weights[i * input_size + j];
                }
            }
        }
    }

    void backward(const float* batch_output_gradients, int batch_size) {
        float* batch_input_gradients = new float[batch_size * input_size]();

        for (int sample = 0; sample < batch_size; ++sample) {
            for (int i = 0; i < output_size; ++i) {
                for (int j = 0; j < input_size; ++j) {
                    weight_gradients[i * input_size + j] += batch_output_gradients[sample * output_size + i] * inputs[sample * input_size + j];
                }
                bias_gradients[i] += batch_output_gradients[sample * output_size + i];
            }

            for (int j = 0; j < input_size; ++j) {
                float gradient = 0.0;
                for (int i = 0; i < output_size; ++i) {
                    gradient += batch_output_gradients[sample * output_size + i] * weights[i * input_size + j];
                }
                batch_input_gradients[sample * input_size + j] = gradient * relu_derivative(inputs[sample * input_size + j]);
            }
        }
        if (inputs) delete[] inputs;
        inputs = new float[batch_size * input_size];
        inputs = batch_input_gradients; // Set gradients for propagation

        //delete[] batch_input_gradients;
    }

    void update_weights(float learning_rate, int batch_size) {
        for (int i = 0; i < output_size; ++i) {
            for (int j = 0; j < input_size; ++j) {
                weights[i * input_size + j] -= learning_rate * weight_gradients[i * input_size + j] / batch_size;
                weight_gradients[i * input_size + j] = 0.0;
            }
            biases[i] -= learning_rate * bias_gradients[i] / batch_size;
            bias_gradients[i] = 0.0;
        }
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

    // Train the network
    nn.train(train_images, train_labels, train_image_rows, 16, num_classes, 10, 0.01f);
    nn.save_model("model.bin");
    nn.load_model("model.bin");

    // Evaluate accuracy
    float* test_predictions = nn.forward(test_images, test_image_rows);
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
