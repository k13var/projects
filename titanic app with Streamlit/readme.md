# RMS Titanic survival rate

![Streamlit_app](https://user-images.githubusercontent.com/60387908/85390777-95e0d000-b549-11ea-9e7b-39d6c719ed19.png)

## Project overview
Streamlit app to predict survival rate on RMS Titanic.

## Preprocessing
1. Resolved missing age values by inputing median age per class.
2. Resolved missing cabin values by transforming into a binary indication (presence vs. absence of cabin).
3. Merged and transformed the 'SibSp' and 'Parch' columns to a binary indicators (travelling alone vs. not travelling alone).
4. Transformes the 'Sex' values from categorical to numerical.
5. Deleted 'Name', 'Ticket', 'Fare', 'PassengerId', and 'Embarked' columns.

## Predictors
1. LogisticRegression, test acc - 0.80
2. RandomForestClassifier, test acc - 0.81
3. SVC, test acc - 0.80
4. VotingClassifier, test acc - 0.80

## Tools, modules and techniques
**Python - web development:**

Streamlit

**Python - machine learning**

pandas | numpy | scikit-learn 