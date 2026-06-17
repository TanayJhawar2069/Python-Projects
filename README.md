# Python Projects

A comprehensive collection of Python projects showcasing machine learning, classification algorithms, and recommendation systems. This repository demonstrates practical applications of popular Python libraries and data science techniques.

## Projects Overview

### MovieRec.py
A movie recommendation engine powered by collaborative filtering using the LightFM library. This project trains multiple recommendation models using different loss functions (WARP, BPR, WARP-KOS, and Logistic) on the MovieLens dataset. The system compares predictions across all models and provides personalized movie recommendations by identifying films users are likely to enjoy based on their preferences and viewing history. It showcases ensemble techniques by combining scores from multiple algorithms to produce robust recommendations.

### Gender.py
A machine learning classification system that predicts gender based on physical measurements (height, weight, and shoe size). This project implements and compares nine different classification algorithms from scikit-learn, including Decision Trees, SVM, Neural Networks, Random Forests, K-Nearest Neighbors, Gaussian Naive Bayes, and more. It demonstrates model training, prediction capabilities, and provides insights into how different machine learning approaches handle the same classification task.

## Technologies Used

- **Machine Learning**: scikit-learn, LightFM
- **Data Processing**: NumPy
- **Algorithms**: Classification, Collaborative Filtering, Ensemble Methods

## Getting Started

Each Python file is standalone and can be executed directly. Ensure you have the required dependencies installed:

```bash
pip install scikit-learn lightfm numpy
```

## Purpose

These projects serve as practical examples of machine learning implementation, perfect for learning, reference, or as a foundation for more advanced applications.

---

**Repository Status**: Active | Open to Contributions | Continuously Evolving
