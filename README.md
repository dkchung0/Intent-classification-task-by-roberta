# Intent Classification Task - Repository Overview

This repository contains code and resources for an intent classification task using the Banking77 dataset from Hugging Face, fine-tuning transformer-based models like BERT and RoBERTa, along with error analysis and preprocessing. Notably, the fine-tuning of the [FacebookAI/roberta-large](https://huggingface.co/FacebookAI/roberta-large) model achieved an accuracy of 0.94318 on the test dataset.

## Dataset

The dataset used for this project is [Banking77](https://huggingface.co/datasets/legacy-datasets/banking77), a dataset that contains 13,083 customer service queries labeled with 77 different user intents. It is commonly used for intent classification tasks in the field of natural language processing.

## Directory and File Overview

### 1. `data_pre_eda.ipynb`
**Description**:  
This notebook handles the initial data preprocessing and exploratory data analysis (EDA). It includes data cleaning, transformations, and visualizations, providing an overview of the dataset and its key characteristics.

---

### 2. `bert_base.ipynb`
**Description**:  
This notebook is used for fine-tuning the `bert-base` model on the intent classification dataset. It contains the setup, training process, and evaluation of the model's performance.

---

### 3. `bert_large.ipynb`
**Description**:  
This notebook focuses on fine-tuning the `bert-large` model for the same task. It compares the training process and performance of `bert-large` with `bert-base` to explore any improvements.

---

### 4. `roberta_lora.ipynb`
**Description**:  
This notebook demonstrates the fine-tuning of a `RoBERTa` model using Low-Rank Adaptation (LoRA). It highlights the efficiency gains from using LoRA and evaluates the model's performance on the intent classification task.

---

### 5. `error_analysis.ipynb`
**Description**:  
This notebook performs an in-depth analysis of misclassified samples from the intent classification task. It compares true and predicted labels and highlights areas where the model struggled, suggesting potential improvements.

---

### 6. `data_test.csv`
**Description**:  
This CSV file contains the results of grid search hyperparameter tuning. It lists various configurations and their corresponding performance metrics, helping to determine the best setup for the intent classification task.

---

### 7. `misclass_df.csv`
**Description**:  
This CSV file contains the misclassified samples, including additional information like true and predicted labels, which are used in the `error_analysis.ipynb` notebook for analyzing model performance.

---

### 8. `requirements.txt`
**Description**:  
A list of Python packages and libraries required for running the notebooks in this repository. The file ensures that users can easily replicate the environment needed for training and evaluating the models.

---

## How to Use

1. Clone the repository:
    ```bash
    git clone https://github.com/dkchung0/Intent-classification-task-by-roberta.git
    ```

2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Run the notebooks in the following order:
    - `data_pre_eda.ipynb` for data preprocessing and analysis.
    - `bert_base.ipynb`, `bert_large.ipynb`, and `roberta_lora.ipynb` for model training and evaluation.
    - `error_analysis.ipynb` for analyzing misclassified data.

## Project Summary

This project compares different transformer-based models for an intent classification task using the [Banking77 dataset](https://huggingface.co/datasets/legacy-datasets/banking77). The dataset includes 77 different user intents derived from 13,083 customer service queries. Models like BERT (base and large variants) and RoBERTa with LoRA fine-tuning are used to classify these intents. The results of grid search hyperparameter tuning, error analysis, and visualizations are provided to give insights into model performance and potential improvements.
