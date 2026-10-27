# Predictive Mulsemedia Modality Selection for 6G Networks

**Final Year Electrical Engineering Project — EEE4022S**

**University of Cape Town (UCT)**

**Author:** Oxolo Madlongolwana
**Department:** Electrical Engineering
**Year:** 2026

---

## 1. Project Overview

Sixth-generation (6G) wireless networks are expected to support immersive communication applications that extend beyond conventional audio and video. Mulsemedia communication incorporates additional sensory modalities, including haptics, smell and taste, to provide more interactive communication experiences.

However, transmitting every available sensory modality during each communication session may introduce unnecessary communication overhead.

This project investigates a **context-aware predictive framework for mulsemedia modality selection**, using historical communication records to predict appropriate sensory-modality combinations.

Two supervised machine-learning algorithms were developed and evaluated:

- **Stochastic Gradient Descent (SGD)** with complete context-interaction features.
- **Random Forest (RF)** classification.

Both models predict complete sensory-modality combinations from contextual information. Their performance is compared using controlled historical datasets containing 250, 500, 750 and 1000 communication records.

## 2. Project Objectives

The primary objective is to develop and evaluate a machine-learning framework capable of predicting sensory-modality combinations for future 6G mulsemedia communication.

The investigation focuses on:

- Predicting complete sensory-modality combinations from historical contextual information.
- Developing and evaluating SGD and Random Forest classifiers.
- Investigating the influence of increasing historical dataset size.
- Analysing classification accuracy and errors across modality combinations.
- Implementing a periodic batch-retraining mechanism.
- Selecting the most suitable classifier based on experimental performance.

## 3. Contextual Inputs and Sensory Modalities

### Contextual Features

The predictive framework uses four categorical contextual variables.

| Feature      | Possible Values                         |
| ------------ | --------------------------------------- |
| Location     | Home, Work, Travel, Social              |
| Relationship | Family, Friend, Colleague, Acquaintance |
| Time         | Morning, Afternoon, Evening, Night      |
| Day          | Weekday, Weekend                        |

These variables produce **128 possible contextual states**.

### Sensory Modalities

The framework considers five sensory modalities:

1. Audio
2. Visual
3. Haptics
4. Smell
5. Taste

The five modalities produce 31 theoretically possible non-empty combinations.

The experimental datasets contain eight observed modality-combination classes. Each complete combination is treated as one classification output, rather than predicting individual modalities independently.

## 4. Experimental Datasets

The experimental investigation uses a controlled, synthetically generated master dataset containing 1000 historical communication records. AI assistance was used in constructing the dataset.

Four cumulative datasets were extracted from the master dataset, ensuring that each smaller dataset is contained within the subsequent larger dataset.

| Dataset    | Training Records (80%) | Testing Records (20%) | Observed Contexts |
| ---------- | ---------------------: | --------------------: | ----------------: |
| 250 calls  |                    200 |                    50 |                64 |
| 500 calls  |                    400 |                   100 |                80 |
| 750 calls  |                    600 |                   150 |               104 |
| 1000 calls |                    800 |                   200 |               128 |

Each record contains the contextual inputs and its associated sensory-modality combination.

The controlled datasets are intended for evaluating the predictive framework and do not represent communication records collected from actual 6G users.

## 5. Machine-Learning Models

### 5.1 Stochastic Gradient Descent (SGD)

The SGD classifier was implemented using scikit-learn and logistic loss.

The model incorporates one-hot encoded contextual variables and an additional complete context-interaction feature.

This interaction feature represents the combination of Location, Relationship, Time and Day as a distinct contextual state, allowing the classifier to learn relationships associated with complete communication situations.

The implementation uses averaged model parameters and a fixed random state of 42.

### 5.2 Random Forest (RF)

The Random Forest classifier was implemented using scikit-learn's `RandomForestClassifier`.

It combines predictions from multiple decision trees and learns conditional relationships between the contextual features.

The implementation investigates hyperparameters including the number of decision trees, maximum tree depth, split criteria, minimum sample requirements and feature-selection strategies.

Unlike the SGD implementation, Random Forest does not require an explicitly constructed complete context-interaction feature.

### 5.3 Periodic Batch Retraining

Both implementations incorporate a periodic batch-retraining mechanism.

Newly completed communication records are validated and accumulated in a buffer. After **10 valid new calls** have accumulated, the records are incorporated into the historical dataset and the model is retrained using the expanded historical information.

This approach allows additional communication records to contribute to future predictions without retraining after every individual call.

## 6. Experimental Results

The classifiers were evaluated using classification accuracy, Macro Precision, Macro Recall, Macro F1-score, Weighted Precision, Weighted Recall and Weighted F1-score.

### Classification Accuracy

| Historical Calls | SGD Accuracy | Random Forest Accuracy |
| ---------------- | -----------: | ---------------------: |
| 250              |       82.00% |                 84.00% |
| 500              |       88.00% |                 91.00% |
| 750              |       84.67% |                 86.00% |
| 1000             |       86.00% |                 87.00% |

### F1-Score Comparison

| Calls | SGD Macro F1 | RF Macro F1 | SGD Weighted F1 | RF Weighted F1 |
| ----- | -----------: | ----------: | --------------: | -------------: |
| 250   |       0.6894 |      0.8413 |          0.8122 |         0.8375 |
| 500   |       0.8999 |      0.9428 |          0.8796 |         0.9093 |
| 750   |       0.8534 |      0.7590 |          0.8435 |         0.8549 |
| 1000  |       0.8545 |      0.7493 |          0.8581 |         0.8643 |

### Key Findings

- Random Forest achieved higher classification accuracy and Weighted F1-scores across all four datasets.
- Both classifiers achieved their highest accuracy using 500 historical calls.
- Increasing dataset size did not produce continuous improvements in predictive performance.
- SGD achieved higher Macro F1-scores at 750 and 1000 calls, indicating stronger class-balanced performance in these evaluations.
- Classification errors were frequently observed between `Audio + Visual` and `Audio + Visual + Haptics`.

**Random Forest was selected as the preferred classifier** based on its consistently stronger overall classification accuracy and Weighted F1-scores.

## 7. Software Requirements

The project was developed using Python and Jupyter Notebook.

The principal libraries are:

- Python
- Jupyter Notebook
- NumPy
- pandas
- scikit-learn
- Matplotlib

Install the required packages using:

`python -m pip install numpy pandas scikit-learn matplotlib notebook`

The final repository should document the library versions used to reproduce the reported results.

## 8. Running the Models

1. Clone or download this repository.
2. Install the Python dependencies.
3. Ensure that the historical dataset files are available in the paths expected by the notebooks.
4. Open Jupyter Notebook using `jupyter notebook`.
5. Open the SGD or Random Forest notebook.
6. Run the notebook cells in their intended order.

Each model implementation includes data preprocessing, classifier training, prediction and performance evaluation.

To reproduce the reported experiments, use the original controlled datasets, fixed random seeds and final model configurations. Matching training and testing record assignments should be verified when comparing the two classifiers.

## 9. Project Limitations

The implemented framework is a proof of concept for predictive mulsemedia modality selection.

The experiments were conducted using controlled datasets rather than real user communication records. Only eight of the 31 theoretically possible modality combinations were represented.

The framework also does not directly incorporate network traffic conditions, available bandwidth, device capabilities or explicit user preferences as predictive inputs.

Although the classifiers achieved relatively high prediction accuracy, the experiments did not measure actual 6G transmission performance, latency, bandwidth consumption or user-perceived Quality of Experience.

Consequently, the findings demonstrate predictive classification performance under controlled experimental conditions rather than improvements in an operational 6G communication network.

## 10. Future Work

Future development could extend this project through:

- Evaluation using larger datasets collected from real users.
- Incorporation of network conditions, device capabilities and explicit user preferences.
- Improving classification performance for less frequently represented modality combinations.
- Repeated experimental evaluations to establish statistical reliability.
- Measuring computational costs associated with model inference and retraining.
- Integration with a simulated 6G network to investigate Quality of Service and Quality of Experience.

## 11. Academic Information

This repository accompanies the final-year project:

**Predictive Mulsemedia Modality Selection for 6G Networks**

Submitted in partial fulfilment of the requirements for a Bachelor of Science degree in Electrical Engineering at the University of Cape Town.

**Author:** Oxolo Madlongolwana
**Institution:** University of Cape Town
**Year:** 2026

The experimental datasets and results are intended for academic research and evaluation.

AI tools, including OpenAI ChatGPT, were used to support code development and the construction of the controlled experimental dataset. The accompanying project report provides further information about AI assistance and the investigation's limitations.
