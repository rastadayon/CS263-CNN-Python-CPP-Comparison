# CNN-Python-CPP-Comparison
Comparison of runtimes of Convolutional Neural Networks implemented in Python vs C++ programming languages

Survey Notes:

https://www.preprints.org/manuscript/202012.0516/v1

## Motivation
---
When it comes to training deep neural networks, the go to programming language is usually Python. However, we know that Python being an interpreted language suffers from a lot slowdowns due to lack of compiler optimizations. In this project we want to verify if using a statically typed and compiled program could improve the runtime of train and testing a convolutional neural network (CNN).

