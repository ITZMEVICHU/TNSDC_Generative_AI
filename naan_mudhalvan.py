# -*- coding: utf-8 -*-
"""Naan_Mudhalvan.ipynb

# ***Bank-Customer-Churn-Prediction***
**95362110460**

**Vishnu A**

# Packages Import
"""

pip install numpy

pip install pandas

pip install --upgrade pandas

pip install seaborn

pip install ydata_profiling

# Commented out IPython magic to ensure Python compatibility.
# Imports
import numpy as np
import pandas as pd
import seaborn
import matplotlib.pyplot as plt
from ydata_profiling import ProfileReport
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report,ConfusionMatrixDisplay
# %matplotlib inline

# initializations
seaborn.set_theme()

"""# Reading the data"""

path = "/content/Churn_Modelling.csv"
df = pd.read_csv(path)

pip install --upgrade pandas

df.head(15),

df.tail(15),

"""# EDA"""

df.info()

"""## Basic Statistic"""

df.describe(exclude="object").T,

df.describe(include="object"),

"""## Unique values"""

df.nunique()

"""## Distribution"""

pip install matplotlib

!pip install matplotlib

seaborn.displot(data=df["Geography"])

seaborn.displot(df['Gender'])

df.skew(numeric_only = True)

"""## Handling Missing Values & Duplicates"""

df.isna().sum()

df.isnull().sum()

df.duplicated().sum()

df = df.dropna()

df.isnull().sum()

df.isna().sum()

"""## Dropping Unnecessary Columns"""

df = df.drop(['RowNumber','CustomerId', 'Surname'], axis=1)
shadow = df.copy()

df.head(20),

"""## Outliers and Boxplot"""

df.shape

list = df.columns.tolist()
list.remove("Geography")
list.remove("Age")
list.remove("Gender")
list

fig, axs = plt.subplots(nrows=4, ncols=2, figsize=(16,9))
axs = axs.flatten()
for idx,column in enumerate(list):
    seaborn.boxplot(data=df[column], ax=axs[idx])
    axs[idx].set_title(column)
plt.tight_layout()

"""Not much of outliers are present. Only in credit score column, there are some very low values which makes sense cause low credit score is a normal phenomena. Also in NumOfProducts column, the outlier value is 4 which is also normal because a person can use multiple products from a bank if he/she deems it necessary. Other than these, there are not much outliers according to this boxplot illustration. So, moving on.

## Correlations
"""

corr = shadow.drop(['Gender', "Geography"], axis=1)
corr_matrix = corr.corr()
seaborn.heatmap(corr_matrix, annot=True, fmt=".0%")

"""# Encoding Necessary Features"""

df = pd.get_dummies(data=df)

df.head(20),

"""# Model Training & Evaluation"""

features = df.drop(["Exited"], axis=1)
label = df["Exited"]

feature_train, feature_test, label_train, label_test = train_test_split(features, label, test_size=0.2, random_state=0)

"""## Model Training"""

LR = LogisticRegression()
CT = DecisionTreeClassifier()
RF = RandomForestClassifier()

"""### Logistic Regression"""

LR.fit(feature_train, label_train)

lr_prediction = LR.predict(feature_test)

accuracy_score(label_test, lr_prediction) * 100

print(classification_report(label_test, lr_prediction))

ConfusionMatrixDisplay(confusion_matrix(label_test, lr_prediction),display_labels=LR.classes_).plot()

confusion_matrix(label_test, lr_prediction)

"""### Decision Tree"""

CT.fit(feature_train, label_train)

ct_prediction = CT.predict(feature_test)

accuracy_score(label_test, ct_prediction) * 100

print(classification_report(label_test, ct_prediction))

ConfusionMatrixDisplay(confusion_matrix(label_test, ct_prediction),display_labels=CT.classes_).plot()

"""### Random Forest"""

RF.fit(feature_train, label_train)

rf_prediction = RF.predict(feature_test)

accuracy_score(label_test, rf_prediction) * 100

print(classification_report(label_test, rf_prediction))

ConfusionMatrixDisplay(confusion_matrix(label_test, rf_prediction),display_labels=RF.classes_).plot()
