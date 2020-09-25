# NEAT-Pathfinder

This is a simple pathfinding algorithm that utilizes the NEAT (NeuroEvolution of Augmenting Topologies) algorithm library to genetically evolve a neural network connection. The problem solved in this project is a simple pathfinder, where an input and output node is defined, and the shortest path is constructed between them. The neural network has a simple 4-input and 4-output topology. The inputs to it are the distances of the next point to the target node, whilst the outputs are the direction of traversal (up, down, left, right). 

Initially, the randomly-weighted neural network performs poorly, but as the fitness training proceeds for each population and the best performing genomes are proceeded to the next generation, the neural network gradually learns how to solve the problem. This is performed in the script nn_pathfinder_train.py. The trained neural network will then be saved onto the pickle file winner.pkl.

The image below shows the training phase of the neural network, whereby a population of 25 is used for each generation:

<img src='/images/training.JPG' width="60%">

After the neural network is trained and meets a certain fitness criterion, it can be tested on an arbitrary start and end node using the script nn_pathfinder_applied.py. An example of the testing phase is shown below:

<img src='/images/applied.JPG' width="30%">
