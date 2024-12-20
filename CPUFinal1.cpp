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

#include <iomanip> // For formatting
using namespace std;
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
            if (normalize) val /= 255.0; // Normalize pixel values to [0, 1] 
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


// Activation functions
inline double relu(double x) { return max(0.0, x); }
inline double relu_derivative(double x) { return x > 0 ? 1.0 : 0.0; }

void softmax(vector<double>& outputs) {
    double max_val = *max_element(outputs.begin(), outputs.end());
    double sum = 0.0;
    for (double& val : outputs) {
        val = exp(val - max_val); // Subtract max_val for numerical stability
        sum += val;
    }
    for (double& val : outputs) val /= sum;
}

// Cross-entropy loss
double cross_entropy_loss(const vector<double>& predictions, const vector<double>& labels) {
    double loss = 0.0;
    for (size_t i = 0; i < predictions.size(); ++i) {
        loss -= labels[i] * log(predictions[i] + 1e-15); // Avoid log(0)
    }
    return loss;
}

// Updated Layer class to handle batches
class Layer {
public:
    vector<vector<double>> weights;
    vector<double> biases;
    vector<vector<double>> outputs;
    vector<vector<double>> inputs;
    vector<vector<double>> weight_gradients;
    vector<double> bias_gradients;

    Layer(int input_size, int output_size)
        : weights(output_size, vector<double>(input_size)),
        biases(output_size, 0.0),
        weight_gradients(output_size, vector<double>(input_size, 0.0)),
        bias_gradients(output_size, 0.0) {
        initialize_weights();
    }

    void initialize_weights() {
        random_device rd;
        mt19937 gen(rd());
        normal_distribution<> dist(0.0, 0.1);
        for (auto& row : weights)
            for (auto& weight : row)
                weight = dist(gen);
    }

    void forward(const vector<vector<double>>& batch_input) {
        inputs = batch_input;
        outputs.assign(batch_input.size(), vector<double>(biases.size(), 0.0));

        for (size_t sample = 0; sample < batch_input.size(); ++sample) {
            for (size_t i = 0; i < biases.size(); ++i) {
                outputs[sample][i] = biases[i];
                for (size_t j = 0; j < batch_input[sample].size(); ++j) {
                    outputs[sample][i] += batch_input[sample][j] * weights[i][j];
                }
            }
        }
    }

    void backward(const vector<vector<double>>& batch_output_gradients, double learning_rate) {
        vector<vector<double>> batch_input_gradients(inputs.size(), vector<double>(inputs[0].size(), 0.0));

        for (size_t sample = 0; sample < inputs.size(); ++sample) {
            for (size_t i = 0; i < weights.size(); ++i) {
                for (size_t j = 0; j < inputs[sample].size(); ++j) {
                    weight_gradients[i][j] += batch_output_gradients[sample][i] * inputs[sample][j];
                }
                bias_gradients[i] += batch_output_gradients[sample][i];
            }

            for (size_t j = 0; j < inputs[sample].size(); ++j) {
                double gradient = 0.0;
                for (size_t i = 0; i < weights.size(); ++i) {
                    gradient += batch_output_gradients[sample][i] * weights[i][j];
                }
                batch_input_gradients[sample][j] = gradient * relu_derivative(inputs[sample][j]);
            }
        }

        inputs = batch_input_gradients; // Set gradients for propagation
    }

    void update_weights(const double& learning_rate) {
        for (size_t i = 0; i < weights.size(); ++i) {
            for (size_t j = 0; j < weights[i].size(); ++j) {
                weights[i][j] -= learning_rate * weight_gradients[i][j] / inputs.size(); // Normalize by batch size
                weight_gradients[i][j] = 0.0;
            }
            biases[i] -= learning_rate * bias_gradients[i] / inputs.size(); // Normalize by batch size
            bias_gradients[i] = 0.0;
        }
    }
};

// Update NeuralNetwork forward and backward to use batch processing
class NeuralNetwork {
public:
    vector<Layer> layers;

    NeuralNetwork(const vector<int>& architecture) {
        for (size_t i = 1; i < architecture.size(); ++i) {
            layers.emplace_back(architecture[i - 1], architecture[i]);
        }
    }

    vector<vector<double>> forward(const vector<vector<double>>& batch_input) {
        vector<vector<double>> activations = batch_input;
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

    void backward(const vector<vector<double>>& predictions, const vector<vector<double>>& labels, double learning_rate) {
        vector<vector<double>> batch_output_gradients(predictions.size(), vector<double>(predictions[0].size()));
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

    void train(const vector<vector<double>>& data, const vector<vector<double>>& labels, int epochs, int batch_size, double learning_rate) {
        for (int epoch = 1; epoch <= epochs; ++epoch) {
            double total_loss = 0.0;
            for (size_t i = 0; i < data.size(); i += batch_size) {
                size_t batch_end = min(i + batch_size, data.size());
                vector<vector<double>> batch_data(data.begin() + i, data.begin() + batch_end);
                vector<vector<double>> batch_labels(labels.begin() + i, labels.begin() + batch_end);

                vector<vector<double>> predictions = forward(batch_data);
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


    vector<double> predict(const vector<double>& input) {
        vector<double> activations = input;
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
double evaluate(NeuralNetwork& model, const vector<vector<double>>& images,
    const vector<vector<double>>& labels) {
    int correct = 0;

    for (size_t i = 0; i < images.size(); ++i) {
        vector <double> predicted= model.predict(images[i]);
        int predicted_class = distance(predicted.begin(), max_element(predicted.begin(), predicted.end()));
        int true_class = distance(labels[i].begin(), max_element(labels[i].begin(), labels[i].end()));

        if (predicted_class == true_class) {
            ++correct;
        }
    }

    return static_cast<double>(correct) / images.size(); // Return accuracy as a fraction
}

// Function to save the model to a file
void save_model(const NeuralNetwork& model, const string& filepath) {
    ofstream file(filepath, ios::out | ios::binary);
    if (!file.is_open()) throw runtime_error("Cannot open file to save model: " + filepath);

    for (const auto& layer : model.layers) {
        // Save weights
        for (const auto& row : layer.weights) {
            for (double weight : row) {
                file.write(reinterpret_cast<const char*>(&weight), sizeof(weight));
            }
        }

        // Save biases
        for (double bias : layer.biases) {
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
            for (double& weight : row) {
                file.read(reinterpret_cast<char*>(&weight), sizeof(weight));
            }
        }

        // Load biases
        for (double& bias : layer.biases) {
            file.read(reinterpret_cast<char*>(&bias), sizeof(bias));
        }
    }

    file.close();
    cout << "Model loaded from " << filepath << endl;
}

// Helper function to display an image (assumes 28x28 grayscale)
void display_image(const vector<double>& image) {
    const int size = 28;
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            if (image[i * size + j] > 0.5) cout << "##"; // Threshold for visual representation
            else cout << "..";
        }
        cout << endl;
    }
}
// Demonstration: Predict and display the first 10 samples
void demonstrate_predictions(NeuralNetwork& model, const vector<vector<double>>& images,
    const vector<vector<double>>& labels) {
    const int num_samples = 20;

    for (int i = 0; i < num_samples; ++i) {
        const auto& image = images[i];
        const auto& label = labels[i];

        vector <double> predicted = model.predict(image);
        int predicted_class = distance(predicted.begin(), max_element(predicted.begin(), predicted.end()));
        int true_class = distance(label.begin(), max_element(label.begin(), label.end()));

        cout << "Sample " << i + 1 << ":" << endl;
        display_image(image);
        cout << "Predicted Label: " << predicted_class << ", Actual Label: " << true_class << endl;
        cout << "-----------------------------" << endl;
    }
}


int main() {

    vector<vector<double>> train_images, train_labels;
    vector<vector<double>> test_images, test_labels;

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
    nn.train(train_images, train_labels, 10, 32, 0.01);



    //// Save the trained model
    save_model(nn, model_path);

    //// Load the model to demonstrate persistence
    NeuralNetwork loaded_nn({ 784, 128, 128, 10 });
    load_model(loaded_nn, model_path);
    

    double train_accuracy = evaluate(loaded_nn, train_images, train_labels);
    double test_accuracy = evaluate(loaded_nn, test_images, test_labels);

    cout << "train accuracy: " << train_accuracy * 100.0 << "%" << endl;
    cout << "test accuracy: " << test_accuracy * 100.0 << "%" << endl;

    // Demonstrate predictions
    demonstrate_predictions(loaded_nn, test_images, test_labels);
    //cuBLAS
    return 0;
}
