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

// Function to parse a CSV file into a 2D vector
std::vector<std::vector<double>> load_csv(const std::string& filepath, bool normalize = false) {
    std::ifstream file(filepath);
    if (!file.is_open()) throw std::runtime_error("Cannot open file: " + filepath);

    std::vector<std::vector<double>> data;
    std::string line;

    while (std::getline(file, line)) {
        std::vector<double> row;
        std::stringstream ss(line);
        std::string value;

        while (std::getline(ss, value, ',')) {
            double val = std::stod(value);
            if (normalize) val /= 255.0; // Normalize pixel values to [0, 1] if required
            row.push_back(val);
        }
        data.push_back(row);
    }

    file.close();
    return data;
}
// Function to load labels and convert to one-hot encoding
std::vector<std::vector<double>> load_labels_csv(const std::string& filepath, int num_classes = 10) {
    std::ifstream file(filepath);
    if (!file.is_open()) throw std::runtime_error("Cannot open file: " + filepath);

    std::vector<std::vector<double>> labels;
    std::string line;

    while (std::getline(file, line)) {
        int label = std::stoi(line); // Read the label
        std::vector<double> one_hot(num_classes, 0.0);
        one_hot[label] = 1.0; // One-hot encode
        labels.push_back(one_hot);
    }

    file.close();
    return labels;
}

// Main function to load and preprocess the dataset
void load_and_preprocess_dataset_csv(const std::string& train_images_path,
    const std::string& train_labels_path,
    const std::string& test_images_path,
    const std::string& test_labels_path,
    std::vector<std::vector<double>>& train_images,
    std::vector<std::vector<double>>& train_labels,
    std::vector<std::vector<double>>& test_images,
    std::vector<std::vector<double>>& test_labels) {
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
            throw std::runtime_error("Mismatch between number of train images and labels.");
        }
        if (test_images.size() != test_labels.size()) {
            throw std::runtime_error("Mismatch between number of test images and labels.");
        }

        std::cout << "Dataset loaded successfully:" << std::endl;
        std::cout << " - Train samples: " << train_images.size() << std::endl;
        std::cout << " - Test samples: " << test_images.size() << std::endl;

    }
    catch (const std::exception& ex) {
        std::cerr << "Error while loading dataset: " << ex.what() << std::endl;
    }
}


// Activation functions
inline double relu(double x) { return std::max(0.0, x); }
inline double relu_derivative(double x) { return x > 0 ? 1.0 : 0.0; }

void softmax(std::vector<double>& outputs) {
    double max_val = *std::max_element(outputs.begin(), outputs.end());
    double sum = 0.0;
    for (double& val : outputs) {
        val = std::exp(val - max_val); // Subtract max_val for numerical stability
        sum += val;
    }
    for (double& val : outputs) val /= sum;
}

// Cross-entropy loss
double cross_entropy_loss(const std::vector<double>& predictions, const std::vector<double>& labels) {
    double loss = 0.0;
    for (size_t i = 0; i < predictions.size(); ++i) {
        loss -= labels[i] * std::log(predictions[i] + 1e-15); // Avoid log(0)
    }
    return loss;
}

// Layer class
class Layer {
public:
    std::vector<std::vector<double>> weights;
    std::vector<double> biases;
    std::vector<double> outputs;
    std::vector<double> inputs;
    std::vector<std::vector<double>> weight_gradients;
    std::vector<double> bias_gradients;

    Layer(int input_size, int output_size)
        : weights(output_size, std::vector<double>(input_size)),
        biases(output_size, 0.0),
        outputs(output_size, 0.0),
        weight_gradients(output_size, std::vector<double>(input_size, 0.0)),
        bias_gradients(output_size, 0.0) {
        initialize_weights();
    }

    void initialize_weights() {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<> dist(0.0, 0.1); // Gaussian initialization
        for (auto& row : weights)
            for (auto& weight : row)
                weight = dist(gen);
    }

    void forward(const std::vector<double>& input) {
        inputs = input;
        for (size_t i = 0; i < outputs.size(); ++i) {
            outputs[i] = biases[i];
            for (size_t j = 0; j < inputs.size(); ++j) {
                outputs[i] += inputs[j] * weights[i][j];
            }
        }
    }

    void backward(const std::vector<double>& output_gradients, double learning_rate) {
        for (size_t i = 0; i < weights.size(); ++i) {
            for (size_t j = 0; j < inputs.size(); ++j) {
                weight_gradients[i][j] += output_gradients[i] * inputs[j];
            }
            bias_gradients[i] += output_gradients[i];
        }
        for (size_t i = 0; i < inputs.size(); ++i) {
            double gradient = 0.0;
            for (size_t j = 0; j < weights.size(); ++j) {
                gradient += output_gradients[j] * weights[j][i];
            }
            inputs[i] = gradient * relu_derivative(inputs[i]);
        }
        update_weights(learning_rate);
    }

    void update_weights(double learning_rate) {
        for (size_t i = 0; i < weights.size(); ++i) {
            for (size_t j = 0; j < weights[i].size(); ++j) {
                weights[i][j] -= learning_rate * weight_gradients[i][j];
                weight_gradients[i][j] = 0.0; // Reset gradients
            }
            biases[i] -= learning_rate * bias_gradients[i];
            bias_gradients[i] = 0.0; // Reset gradients
        }
    }
};

// Neural Network
class NeuralNetwork {
public:
    std::vector<Layer> layers;

    NeuralNetwork(const std::vector<int>& architecture) {
        for (size_t i = 1; i < architecture.size(); ++i) {
            layers.emplace_back(architecture[i - 1], architecture[i]);
        }
    }

    std::vector<double> forward(const std::vector<double>& input) {
        std::vector<double> activations = input;
        for (size_t i = 0; i < layers.size(); ++i) {
            layers[i].forward(activations);
            activations = layers[i].outputs;
            if (i < layers.size() - 1) { // Hidden layers
                std::transform(activations.begin(), activations.end(), activations.begin(), relu);
            }
            else { // Output layer
                softmax(activations);
            }
        }
        return activations;
    }

    void backward(const std::vector<double>& predictions, const std::vector<double>& labels, double learning_rate) {
        std::vector<double> output_gradients(predictions.size());
        for (size_t i = 0; i < predictions.size(); ++i) {
            output_gradients[i] = predictions[i] - labels[i];
        }
        for (int i = layers.size() - 1; i >= 0; --i) {
            layers[i].backward(output_gradients, learning_rate);
            output_gradients = layers[i].inputs; // Propagate to previous layer
        }
    }

    void train(const std::vector<std::vector<double>>& data,
        const std::vector<std::vector<double>>& labels,
        int epochs, int batch_size, double learning_rate) {
        for (int epoch = 1; epoch <= epochs; ++epoch) {
            double total_loss = 0.0;
            for (size_t i = 0; i < data.size(); i += batch_size) {
                int batch_end = std::min(i + batch_size, data.size());
                for (size_t j = i; j < batch_end; ++j) {
                    std::vector<double> predictions = forward(data[j]);
                    total_loss += cross_entropy_loss(predictions, labels[j]);
                    backward(predictions, labels[j], learning_rate);
                }
            }
            std::cout << "Epoch " << epoch << " - Loss: " << total_loss / data.size() << std::endl;
        }
    }
};

int predict(NeuralNetwork& model, const std::vector<double>& input) {
    std::vector<double> output = model.forward(input);
    return std::distance(output.begin(), std::max_element(output.begin(), output.end())); // Return index of max value
}

// Function to evaluate the model on a dataset
double evaluate(NeuralNetwork& model, const std::vector<std::vector<double>>& images,
    const std::vector<std::vector<double>>& labels) {
    int correct = 0;

    for (size_t i = 0; i < images.size(); ++i) {
        int predicted_class = predict(model, images[i]);
        int true_class = std::distance(labels[i].begin(), std::max_element(labels[i].begin(), labels[i].end()));

        if (predicted_class == true_class) {
            ++correct;
        }
    }

    return static_cast<double>(correct) / images.size(); // Return accuracy as a fraction
}

// Function to save the model to a file
void save_model(const NeuralNetwork& model, const std::string& filepath) {
    std::ofstream file(filepath, std::ios::out | std::ios::binary);
    if (!file.is_open()) throw std::runtime_error("Cannot open file to save model: " + filepath);

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
    std::cout << "Model saved to " << filepath << std::endl;
}

// Function to load the model from a file
void load_model(NeuralNetwork& model, const std::string& filepath) {
    std::ifstream file(filepath, std::ios::in | std::ios::binary);
    if (!file.is_open()) throw std::runtime_error("Cannot open file to load model: " + filepath);

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
    std::cout << "Model loaded from " << filepath << std::endl;
}

// Helper function to display an image (assumes 28x28 grayscale)
void display_image(const std::vector<double>& image) {
    const int size = 28;
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            if (image[i * size + j] > 0.5) std::cout << "##"; // Threshold for visual representation
            else std::cout << "..";
        }
        std::cout << std::endl;
    }
}

// Demonstration: Predict and display the first 10 samples
void demonstrate_predictions(NeuralNetwork& model, const std::vector<std::vector<double>>& images,
    const std::vector<std::vector<double>>& labels) {
    const int num_samples = 20;

    for (int i = 0; i < num_samples; ++i) {
        const auto& image = images[i];
        const auto& label = labels[i];

        int predicted_class = predict(model, image);
        int true_class = std::distance(label.begin(), std::max_element(label.begin(), label.end()));

        std::cout << "Sample " << i + 1 << ":" << std::endl;
        display_image(image);
        std::cout << "Predicted Label: " << predicted_class << ", Actual Label: " << true_class << std::endl;
        std::cout << "-----------------------------" << std::endl;
    }
}


int main() {

    std::vector<std::vector<double>> train_images, train_labels;
    std::vector<std::vector<double>> test_images, test_labels;

    const std::string train_images_path = "x_train.csv";
    const std::string train_labels_path = "y_train.csv";
    const std::string test_images_path = "x_test.csv";
    const std::string test_labels_path = "y_test.csv";

    const std::string model_path = "nn_model.bin";

    // Load and preprocess the dataset
    load_and_preprocess_dataset_csv(train_images_path, train_labels_path, test_images_path, test_labels_path,
        train_images, train_labels, test_images, test_labels);
    //// Example setup: Define the architecture
    //NeuralNetwork nn({ 784, 128, 128, 10 });
    //// Load and preprocess Fashion MNIST dataset here.
    //// Use nn.train() to train the network.
    //nn.train(train_images, train_labels, 10, 32, 0.01);



    //// Save the trained model
    //save_model(nn, model_path);

    //// Load the model to demonstrate persistence
    NeuralNetwork loaded_nn({ 784, 128, 128, 10 });
    load_model(loaded_nn, model_path);
    

    double train_accuracy = evaluate(loaded_nn, train_images, train_labels);
    double test_accuracy = evaluate(loaded_nn, test_images, test_labels);

    std::cout << "train accuracy: " << train_accuracy * 100.0 << "%" << std::endl;
    std::cout << "test accuracy: " << test_accuracy * 100.0 << "%" << std::endl;

    // Demonstrate predictions
    demonstrate_predictions(loaded_nn, test_images, test_labels);

    return 0;
}
