# MNIST Digit Classification with PyTorch

This project implements a simple neural network to classify handwritten digits from the MNIST dataset using PyTorch. The model is trained from scratch and includes evaluation and visualization tools.

## Features

- Custom `SimpleMLP` model (Multi-Layer Perceptron)
- Training with mini-batch gradient descent using Adam
- Evaluation function with test accuracy per epoch
- Visualization of:
  - Misclassified digits
  - Random test samples with predictions
- Modular code (`model.py`, `utils.py`, `visualize.py`)
- Clean training loop and reusable utilities
- Final model saved as `.pth` file

## Project Structure

```
mnist-classification/
├── train.py
├── model.py
├── utils.py
├── visualize.py
├── requirements.txt
├── README.md
└── data/  (MNIST data will be downloaded here)
```

## Requirements

Install dependencies with:

```bash
pip install -r requirements.txt
```

Or manually:

```bash
pip install torch torchvision matplotlib
```

## How to Run

Train the model and visualize predictions:

```bash
python train.py
```

After training, the script will:
- Print accuracy per epoch
- Show a sample of wrong predictions
- Show a random batch of test images with predicted labels
- Save the model as `mnist_model.pth`

## Results

You should see a final test accuracy around **97–98%**, depending on hyperparameters.

