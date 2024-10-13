## Overview
This project demonstrates sentiment analysis on a dataset of IMDB movie reviews. The task involves cleaning and preprocessing the text data, exploring data distributions, and implementing machine learning models to classify movie reviews into positive and negative sentiments.

## Dataset
We use the IMDB Movie Reviews dataset for sentiment classification. The dataset consists of movie reviews labeled as **positive** or **negative**.



## Setup and Execution
### 1. **Clone the Repository**

    First, clone this repository

    ```bash
    git clone https://github.com/pipipip169/Sentiment-Analysis.git
    ```
### 2. Load and Clean the Data
- **Remove duplicates**: Duplicate rows are removed to avoid overfitting or skewed results.
- **Text Preprocessing**:
  - Expanding contractions (e.g., "don't" becomes "do not").
  - Removing HTML tags, punctuation, and emojis.
  - Converting all text to lowercase.
  - Lemmatizing words to their base form.
  - Removing stopwords (common words like "the", "and", etc.).

### 3. Data Analysis
- **Visualizing Sentiment Distribution**: 
  - A pie chart is generated to show the distribution of positive and negative sentiments in the dataset.
- **Exploratory Data Analysis (EDA)**:
  - Histograms showing the distribution of word lengths in positive and negative reviews.
  - Kernel density plots comparing word lengths between positive and negative reviews.

### 4. Split the Dataset
- The dataset is split into **training** and **testing** sets (80% training, 20% testing) using the `train_test_split` function from Scikit-learn.
- Labels are encoded into numerical values (0 for negative, 1 for positive) using `LabelEncoder`.

### 5. Vectorization
- **TF-IDF Vectorization**: The text data is transformed into numerical features using the TF-IDF (Term Frequency-Inverse Document Frequency) vectorizer.
- The vectorizer is limited to the top 10,000 features for efficient computation.

### 6. Training and Evaluation
Two machine learning models are trained and evaluated on the vectorized data:
- **Random Forest Classifier**
- **Decision Tree Classifier**
  

## Results
The project uses the accuracy metric to evaluate the models. The results of both classifiers are printed at the end of the script.

