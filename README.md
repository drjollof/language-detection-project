
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
- numpy
- Scikit-learn
- Matplotlib
- Seaborn
- Tqdm


## Project Workflow

 #### 1. Data Exploration:
 - Analyzed the distribution of languages.
 
 #### 2. Data Preprocessing:
  - Converted text into numerical features using CountVectorizer.
 #### 3. Model Training:
   Trained four classifiers:
   - Multinomial Naive Bayes
   - Random Forest
   - Support Vector Classifier (SVC)
   - Multinomial Logistic Regression
 #### 4. Model Evaluation:
 - Evaluated performance using accuracy, precision, recall, confusion_matrix and F1-score.

---

##  Base Performances

 #### Multinomial Logistic Regression:
 - Accuracy: 0.9467

 - Precision: 0.9598

 - Recall: 0.9467

 - F1 Score: 0.9490


 #### Multinomial Naive Bayes:
 - Accuracy: 0.9438

 - Precision: 0.9597

 - Recall: 0.9438

 - F1 Score: 0.9430


 #### Random Forest Classifier:
 - Accuracy: 0.9195

 - Precision: 0.9519

 - Recall: 0.9195

 - F1 Score: 0.9179

 #### Support Vector Classifier:
 - Accuracy: 0.8802

 - Precision: 0.9157

 - Recall: 0.8802

 - F1 Score: 0.8841

### Best Model:
 Multinomial Naive Bayes  (0.9791)

---

####  Installation

To set up this project locally, follow these steps:

####  1. Clone the Repository:
```sh
git clone https://github.com/drjollof/language-detection-project.git
```
#### 2. Navigate to project directory

```sh
cd language-detection-project
```

#### 3. Install Dependencies
```sh
pip install -r requirements.txt
```

#### 4. Run App
```sh
streamlit run app.py
```
---

## Live App

You can access the deployed version of this app here:  **[Language Detection App](https://trad-lang-detect.streamlit.app/)**  
