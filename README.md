# CNN-Python-CPP-Comparison
Comparison of runtimes of Convolutional Neural Networks implemented in Python vs C++ programming languages

Link to report: [Report](https://drive.google.com/file/d/1p3hMmK8sMm-Dq2joJolGr2qwJgRZ4siV/view?usp=sharing)

---
## Motivation
When it comes to training deep neural networks, the go-to programming language is usually Python. However, we know that Python being an interpreted language suffers from a lot of slowdowns due to a lack of compiler optimizations. In this project, we want to verify if using a statically typed and compiled program could improve the runtime of the train and testing a convolutional neural network (CNN).


---
## Approach
This project is divided into two distinct sections. In the first part, we focus on measuring the runtimes for training and testing a Convolutional Neural Network (CNN) implemented in both Python and C++. We explore various hyperparameters and examine how the runtime varies across different parameter settings. The hyperparameters and their corresponding values are as follows:
- Batch Sizes: 8, 16, 32, 64, 128, 256
- Number of Epochs: 5, 10, 20, 40, 70, 100
- Kernel Size Values: 3, 5, 8, 10, 15
- Number of Layers: 3, 4, 5, 6
- Optimizers: Adam, SGD

In the second part, we employ different profilers for C++ and Python to gain insights into the underlying functions called during the execution of the code. The specifics of the profilers are as follows:
- Python:
  1. Pytorch Profiler
  2. Py-spy
 
- C++:
  1. ValGrind used with the Callgrind tool
 
--- 
## Steps for building the project
If you would like to reproduce our results, you can use the CPP - Python Runtime and Profile notebook provided in the repository. Once downloaded, you will need to clone this repository and use the code inside it. The notebook is partitioned into 6 main sections. 
1. Preparing the environment
  1. Install Programs and GPU
  2. Clone Repo
  3. System Information

2. Python Runtime Measuring
  - In this section, you would need to change the directory to the Python directory in this repository
  - From there you can call the run_experiments with different parameters. You can do so by running this command: <code>!python3 /content/CS263-CNN-Python-CPP-Comparison/Python/runtime_experiment.py times_run <times_run> num_epochs <num_epochs> batch_size <batch_size> kernel_size <kernel_size> optimizer <optimizer> num_layers <num_layers></code>
  - After running this Python script you will have your runtimes for training and testing with specific hyperparameters.

3. C++ Runtime Measuring
  - For running the C++ code, you would first need to build it. The project is being built in 2 separate parts. To build it you would need to run all cells under the titles Generate Build System  and Build Tutorials. After compilation, you can run the executable file by running the following command: <code>!./convolutional-neural-network 0 <batch_size> <num_epochs> <kernel_size> <num_layers> <optimizer> </code>
  - Running the code as described will output the runtimes for training and testing with specific given hyperparameters.

4. CPP Profiler
   - To run the Valgrind profiler with the Callgrind tool you would need to run the cells under section CPP Profiler in the notebook. The steps to do this are:
       1. Install valgrind by running the command: <code> !apt-get install valgrind kcachegrind</code>
       2. Adding the -g flag to the compilation instructions. To do this you would need uncomment lines 7 and 8 in the CMakeLists.txt in the directory CPP/tutorials/intermediate/convolutional_neural_network/CMakeLists.txt. Then rebuild the project.
       3. Run the command <code>!valgrind --tool=callgrind --instr-atstart=yes --cache-sim=no ./convolutional-neural-network </code> present in one of the cells.
       4. Finally download the resulting file and install kcachegrind and run the command <code>kcachegrind callgrind.out.<pid></code> locally.
    
5. Python Profiler
   1. Simply change directory to CS263-CNN-Python-CPP-Comparison/Python/ and run the main.py file to run the Pytorch profiler.
   2. If you would like to profile using pyspy you can install the module and then run the command <code>!py-spy record -o profile.json -f speedscope -- python main.py</code> 
