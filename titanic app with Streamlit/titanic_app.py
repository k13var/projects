# titanic_app.py
"""
A simple streamlit app to predict the survival rate
run the app by installing streamlit with pip and typing
> streamlit run titanic_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from PIL import Image
image = Image.open('titanic.jpg')

st.image(image, use_column_width=True)

st.title('Titanic Survival Rate')

# import the data (features + labels)
@st.cache
def get_data():
    target = "Survived"
    titanic =  pd.read_csv('titanic_processed.csv')
    y = titanic[target]# get labels
    X = titanic.drop(target, axis=1) #
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    return titanic, X_train, X_test, y_train, y_test

# split the data
titanic, X_train, X_test, y_train, y_test = get_data()

# train the model
@st.cache(allow_output_mutation=True)
def train():
    # train the model
    log_clf = LogisticRegression(solver="liblinear", penalty="l1", C=1, random_state=42)
    rnd_clf = RandomForestClassifier(n_estimators=100, bootstrap=False, max_depth=10, min_samples_leaf=2,
                                     min_samples_split=6, random_state=42)
    svm_clf = SVC(gamma=0.1, degree=3, C=10, kernel='rbf', probability=True, random_state=42)
    #print("*" * 10)
    voting_clf = VotingClassifier(
        estimators=[('lr', log_clf), ('rf', rnd_clf), ('svc', svm_clf)],
        voting='soft')

    voting_clf.fit(X_train, y_train)

    for clf in (log_clf, rnd_clf, svm_clf, voting_clf):
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

    return log_clf, rnd_clf, svm_clf, voting_clf

log_clf, rnd_clf, svm_clf, voting_clf = train()

# make predictions

# select the class
p_class_ = st.sidebar.radio("Passenger class", ('1st class', '2nd class', '3rd class'))
if p_class_ == '1st class':
    class_ = 1
elif p_class_ == '2nd class':
    class_ = 2
else:
    class_ = 3

# select the sex
p_sex_ = st.sidebar.radio("Passenger sex", ('male', 'female'))
if p_sex_ == 'male':
    sex_ = 1
else:
    sex_ = 0

# select the age
age_ = st.sidebar.slider('Passenger age', 0, 80, 29)

# travelling alone
p_travel_ = st.sidebar.radio("Passenger travelling with family", ('yes', 'no'))
if p_travel_ == 'yes':
    not_alone_ = 1
else:
    not_alone_ = 0

# has cabin
p_cabin_ = st.sidebar.radio("Passenger has booked a cabin", ('yes', 'no'))
if p_cabin_ == 'yes':
    cabin_ = 1
else:
    cabin_ = 0

selected_data = np.array([class_, sex_, age_, not_alone_, cabin_]).reshape(1, -1)

# choose model
p_model_ = st.sidebar.selectbox('Choose an estimator',('Logistic Regression',
                                                       'Random Forest',
                                                       'Support Vector Machine',
                                                       'Ensemble Learning'))
if p_model_ == 'Logistic Regression':
    model_ = log_clf
elif p_model_ == 'Random Forest':
    model_ = rnd_clf
elif p_model_ == 'Support Vector Machine':
    model_ = svm_clf
else:
    model_ = voting_clf

# display prediction

survived = model_.predict_proba(selected_data)

lucky_chap = survived[0][1]*100

f"""
## {lucky_chap:.{2}f} %
"""
if lucky_chap >50:
    with st.spinner('Lucky chap! :sunglasses: :cocktail:'):
        time.sleep(3)
else:
    with st.spinner("You've hit rock bottom...:fish::scream::fish:"):
        time.sleep(3)
