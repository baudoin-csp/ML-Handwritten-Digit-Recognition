# Handwritten Mathematical Symbol Recognition

This machine learning project aims to recognize handwritten mathematical symbols using the HASYv2 dataset. The dataset consists of hand-drawn mathematical symbols, and our goal is to classify these symbols into their respective categories. The project is divided into two parts: Part 1 and Part 2, each using different machine learning techniques.

## Dataset

The [HASYv2 dataset](https://arxiv.org/abs/1701.08380) is a collection of hand-drawn mathematical symbols. It includes various symbols used in mathematics and engineering, such as numbers, operators, and special symbols. The dataset is organized into training and test sets.

### Dataset overview

<img width="421" alt="Screenshot 2023-09-05 at 17 20 43" src="https://github.com/baudoin-csp/Handwritten-Math-Symbols-Recognition/assets/118212739/057c0b07-c9fc-4378-863c-aa00c66ffb49">

## Installation

1. Clone this repository to your local machine:

   ```
   git clone https://github.com/baudoin-csp/math-symbol-recognition.git
   ```

2. Navigate to the project directory:

   ```
   cd math-symbol-recognition
   ```

## Usage

### Part 1: K-Means, Logistic Regression, and SVM

- To run K-Means clustering (e.g., with K=3), use the following command:

  ```
  python part1/main.py --data data/hasy-data --method kmeans --K 3
  ```

- To run Logistic Regression (e.g., with lr=0.00001 and max_iters=100), use the following command:

  ```
  python part1/main.py --data data/hasy-data --method logistic_regression --lr 1e-5 --max_iters 100
  ```

- To run SVM (e.g., with C=1, an RBF kernel, and gamma=0.01), use the following command:

  ```
  python part1/main.py --data data/hasy-data --method svm --svm_c 1. --svm_kernel rbf --svm_gamma 0.01
  ```

### Part 2: Deep Networks with PyTorch (MLP and CNN)

- To run the MLP model in Part 2, use the following command:

  ```
  python part2/main.py --data data/hasy-data --method mlp
  ```

- To run the CNN model in Part 2, use the following command:

  ```
  python part2/main.py --data data/hasy-data --method cnn
  ```

In both parts, you can add the `--test` argument to predict on the actual test set; otherwise, your code should use a validation set.

## Reports

The project reports for both Part 1 and Part 2 are available in the their respective directory as PDF files. These reports provide detailed information about our methodology and a summary of the results achieved using different methods.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributors

- [Baudoin](https://github.com/baudoin-csp)
- [Gustave](https://github.com/GustaveCharles)

## Feedback and Contributions

We welcome feedback, bug reports, and contributions. Please feel free to create issues or pull requests if you have any suggestions or improvements to make.
