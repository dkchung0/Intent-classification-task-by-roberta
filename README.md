# Overview

This repository contains code and resources for an intent classification task using the Banking77 dataset from Hugging Face, fine-tuning transformer-based models like BERT and RoBERTa, along with error analysis and preprocessing. Notably, the fine-tuning of the [FacebookAI/roberta-large](https://huggingface.co/FacebookAI/roberta-large) model achieved an accuracy of 0.94318 on the test dataset.

## Dataset

The dataset used for this project is [Banking77](https://huggingface.co/datasets/legacy-datasets/banking77), a dataset that contains 13,083 customer service queries labeled with 77 different user intents. It is commonly used for intent classification tasks in the field of natural language processing.

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
  
## Files 

#### 1. `data_pre_eda.ipynb`
This notebook handles the initial data preprocessing and exploratory data analysis (EDA). It includes data cleaning, transformations, and visualizations, providing an overview of the dataset and its key characteristics.

---

#### 2. `bert_base.ipynb`
This notebook is used for fine-tuning the `bert-base` model on the intent classification dataset. It contains the setup, training process, and evaluation of the model's performance.

---

#### 3. `bert_large.ipynb`
This notebook focuses on fine-tuning the `bert-large` model for the same task.

---

#### 4. `roberta_lora.ipynb`  
This notebook demonstrates the fine-tuning of a `RoBERTa` model using Low-Rank Adaptation (LoRA). It shows how LoRA can reduce computational resource requirements while still achieving competitive performance. However, it also highlights that the best results were obtained with full parameter fine-tuning.

---

#### 5. `error_analysis.ipynb`
This notebook performs an in-depth analysis of misclassified samples from the intent classification task. It compares true and predicted labels to identify areas with potential data issues and provides insights into where data labeling may need refinement. It highlights the need for re-annotation of problematic data to improve overall model performance.

---

#### 6. `data_test.csv`
This CSV file contains the results of grid search hyperparameter tuning. It lists various configurations and their corresponding performance metrics, helping to determine the best setup for the intent classification task.

---

#### 7. `misclass_df.csv`  
This CSV file contains the misclassified samples, including additional information like true and predicted labels, which are used in the `error_analysis.ipynb` notebook for analyzing model performance.

---

#### 8. `requirements.txt`
A list of Python packages and libraries required for running the notebooks in this repository. The file ensures that users can easily replicate the environment needed for training and evaluating the models.

---

## Conclusion

- **Model Selection Reflection**: The BERT model originally performed well, and the accuracy improvement when compared to the more powerful RoBERTa was limited.
  
- **Fine-Tuning Strategy**: Full parameter fine-tuning is recommended when computational resources permit. When using LoRA, careful attention should be paid to the learning rate settings.

- **Testing Results**: With a baseline accuracy of 0.932 being quite high, the actual improvement to 0.943 is modest (0.011).

- **Error Case Analysis**: It is strongly advised to re-annotate the data, especially focusing on category confusion and labeling errors.

- **NLP Task Outlook**: For more complex tasks, instruction-based learning with LLMs becomes increasingly crucial.
