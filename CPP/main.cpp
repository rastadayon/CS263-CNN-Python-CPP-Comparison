#include <iostream>
#include <vector>
#include <cmath>

using namespace std;

class ConvolutionalLayer {
public:
    ConvolutionalLayer() {
        cerr<<"wtf??"<<endl;
    }
    ConvolutionalLayer(int inputChannels, int outputChannels, int kernelSize)
    {
        inputChannels = inputChannels;
        outputChannels = outputChannels;
        kernelSize = kernelSize;
        // Initialize weights and biases randomly (you may want to use more sophisticated initialization)
        weights.resize(outputChannels, vector<vector<double>>(inputChannels, vector<double>(kernelSize, 0.1)));
        biases.resize(outputChannels, 0.1);
    }

    vector<vector<double>> forward(const vector<vector<double>>& input) const {
        int inputSize = input.size();
        int outputSize = inputSize - kernelSize + 1;

        vector<vector<double>> output(outputChannels, vector<double>(outputSize, 0.0));

        for (int i = 0; i < outputChannels; ++i) {
            for (int j = 0; j < outputSize; ++j) {
                for (int k = 0; k < kernelSize; ++k) {
                    for (int l = 0; l < inputChannels; ++l) {
                        output[i][j] += input[l][j + k] * weights[i][l][k];
                    }
                }
                output[i][j] += biases[i];
                output[i][j] = max(0.0, output[i][j]);  // ReLU activation
            }
        }

        return output;
    }

private:
    int inputChannels;
    int outputChannels;
    int kernelSize;
    vector<vector<vector<double>>> weights;
    vector<double> biases;
};

class FullyConnectedLayer {
public:
    FullyConnectedLayer()
    {
        cerr << "The actual hell????" <<endl;
    }
    FullyConnectedLayer(int inputSize, int outputSize)
        : inputSize(inputSize), outputSize(outputSize)
    {
        // Initialize weights and biases randomly (you may want to use more sophisticated initialization)
        weights.resize(outputSize, vector<double>(inputSize, 0.1));
        biases.resize(outputSize, 0.1);
    }

    vector<double> forward(const vector<double>& input) const {
        vector<double> output(outputSize, 0.0);

        for (int i = 0; i < outputSize; ++i) {
            for (int j = 0; j < inputSize; ++j) {
                output[i] += input[j] * weights[i][j];
            }
            output[i] += biases[i];
            output[i] = 1.0 / (1.0 + exp(-output[i]));  // Sigmoid activation
        }

        return output;
    }

private:
    int inputSize;
    int outputSize;
    vector<vector<double>> weights;
    vector<double> biases;
};

class SimpleCNN {
public:
    SimpleCNN() {
        // Define the architecture
        convLayer = new ConvolutionalLayer(1, 32, 3);  // Input channels: 1, Output channels: 32, Kernel size: 3
        fcLayer = new FullyConnectedLayer(10, 2);  // Input size: 10, Output size: 2
    }

    vector<double> forward(const vector<vector<double>>& input) {
        // Forward pass through the layers
        vector<vector<double>> convOutput = convLayer.forward(input);
        vector<double> fcInput;
        for (const auto& row : convOutput) {
            fcInput.insert(fcInput.end(), row.begin(), row.end());
        }
        return fcLayer.forward(fcInput);
    }

private:
    ConvolutionalLayer convLayer;
    FullyConnectedLayer fcLayer;
};

int main() {
    // Create a SimpleCNN object
    SimpleCNN cnn;

    // Create a simple 2x5 input matrix
    vector<vector<double>> input = {
        {1, 0, 1, 0, 1},
        {0, 1, 0, 1, 0}
    };

    // Perform a forward pass
    vector<double> output = cnn.forward(input);

    // Display the output
    cout << "Output: [";
    for (double val : output) {
        cout << val << " ";
    }
    cout << "]\n";

    return 0;
}
