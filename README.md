
# Language Detection Using Text Classification
This project focus on building a machine learning model that can detect the language of a given text by classifying it into one of 22 unique languages. 


## Dataset
- Columns: 

  • text: Sentences in various languages.
   
  • Language: The corresponding language label for each text sample.

 - Rows: 22,000 rows (1,000 samples per language).
 - Languages: Estonian, Swedish, English, Russian, Romanian, Persian, Pushto, Spanish, Hindi, Korean, Chinese, French, Portuguese, Indonesian, Urdu, Latin, Turkish, Japanese, Dutch, Tamil, Thai, Arabic.
 - Source: Kaggle
## Tools
- Pandas
- Scikit-learn
- Tqdm

## Project Workflow

 #### 1. Data Exploration:
 - Analyzed the distribution of languages.
  - Visualize text length and language frequencies.
 #### 2. Data Preprocessing:
  - Clean text data by removing punctuation and stopwords.
  - Converted text into numerical features using CountVectorizer.
 #### 3. Model Training:
   Trained four classifiers:
   - Multinomial Naive Bayes
   - Random Forest
   - Support Vector Classifier (SVC)
   - Multinomial Logistic Regression
 #### 4. Model Evaluation:
 - Evaluate performance using accuracy, precision, recall, and F1-score.