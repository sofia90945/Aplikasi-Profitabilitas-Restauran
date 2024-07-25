# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
url = '/content/restaurant_menu_optimization_data.csv'
df = pd.read_csv(url)

# Display the first few rows of the dataset
print(df.head())

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Menghapus baris dengan nilai yang hilang
df = df.dropna()

# Memisahkan fitur dan label
X = df.drop('Profitability', axis=1)
y = df['Profitability']

# Mengidentifikasi fitur numerik dan kategorikal
numerical_features = X.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X.select_dtypes(include=['object']).columns

# Membuat preprocessor untuk menskalakan data numerik dan mengkodekan data kategorikal
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(), categorical_features)
    ])

# Membagi dataset menjadi data latih dan uji
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Membuat preprocessor untuk menskalakan data numerik dan mengkodekan data kategorikal
# Set sparse_output=False to return a dense array
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(sparse_output=False), categorical_features)
    ])

# Menerapkan preprocessor ke data
X_train = preprocessor.fit_transform(X_train)
X_test = preprocessor.transform(X_test)

# METODE KE-1
# RANDOM FOREST

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, make_scorer

# Splitting data into features and target
X = df.drop('Profitability', axis=1)
y = df['Profitability']

# Define which columns are categorical and which are numerical
categorical_features = X.select_dtypes(include=['object']).columns
numerical_features = X.select_dtypes(include=['int64', 'float64']).columns

# Create a ColumnTransformer to apply OneHotEncoder to categorical features and StandardScaler to numerical features
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(), categorical_features)
    ])

# Apply the transformations to the features
X = preprocessor.fit_transform(X)

# Ensure target variable is numeric if it's categorical
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define Random Forest model
rf_model = RandomForestClassifier(random_state=42)

# Define cross-validation function
def cross_validate_model(model, X, y, cv=5):
    kf = KFold(n_splits=cv, shuffle=True, random_state=42)
    accuracy_scores = cross_val_score(model, X, y, cv=kf, scoring='accuracy')
    f1_scores = cross_val_score(model, X, y, cv=kf, scoring='f1_macro')
    return accuracy_scores, f1_scores

# Random Forest
rf_accuracy_scores, rf_f1_scores = cross_validate_model(rf_model, X, y)
rf_accuracy = rf_accuracy_scores.mean()
rf_f1 = rf_f1_scores.mean()

print(f"Random Forest - Accuracy: {rf_accuracy}, F1 Score: {rf_f1}")

# METODE KE-2
# SUPPORT VECTOR MACHINE(SVM)

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, make_scorer

# Splitting data into features and target
X = df.drop('Profitability', axis=1)
y = df['Profitability']

# Define which columns are categorical and which are numerical
categorical_features = X.select_dtypes(include=['object']).columns
numerical_features = X.select_dtypes(include=['int64', 'float64']).columns

# Create a ColumnTransformer to apply OneHotEncoder to categorical features and StandardScaler to numerical features
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(), categorical_features)
    ])

# Apply the transformations to the features
X = preprocessor.fit_transform(X)

# Ensure target variable is numeric if it's categorical
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define SVM model
svm_model = SVC(random_state=42)

# Define cross-validation function
def cross_validate_model(model, X, y, cv=5):
    kf = KFold(n_splits=cv, shuffle=True, random_state=42)
    accuracy_scores = cross_val_score(model, X, y, cv=kf, scoring='accuracy')
    f1_scores = cross_val_score(model, X, y, cv=kf, scoring='f1_macro')
    return accuracy_scores, f1_scores

# SVM
svm_accuracy_scores, svm_f1_scores = cross_validate_model(svm_model, X, y)
svm_accuracy = svm_accuracy_scores.mean()
svm_f1 = svm_f1_scores.mean()

print(f"SVM - Accuracy: {svm_accuracy}, F1 Score: {svm_f1}")

# METODE KE-3
# DECISION TREE

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score, make_scorer

# Splitting data into features and target
X = df.drop('Profitability', axis=1)
y = df['Profitability']

# Define which columns are categorical and which are numerical
categorical_features = X.select_dtypes(include=['object']).columns
numerical_features = X.select_dtypes(include=['int64', 'float64']).columns

# Create a ColumnTransformer to apply OneHotEncoder to categorical features and StandardScaler to numerical features
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(), categorical_features)
    ])

# Apply the transformations to the features
X = preprocessor.fit_transform(X)

# Ensure target variable is numeric if it's categorical
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define Decision Tree model
dt_model = DecisionTreeClassifier(random_state=42)

# Define cross-validation function
def cross_validate_model(model, X, y, cv=5):
    kf = KFold(n_splits=cv, shuffle=True, random_state=42)
    accuracy_scores = cross_val_score(model, X, y, cv=kf, scoring='accuracy')
    f1_scores = cross_val_score(model, X, y, cv=kf, scoring='f1_macro')
    return accuracy_scores, f1_scores

# Decision Tree
dt_accuracy_scores, dt_f1_scores = cross_validate_model(dt_model, X, y)
dt_accuracy = dt_accuracy_scores.mean()
dt_f1 = dt_f1_scores.mean()

print(f"Decision Tree - Accuracy: {dt_accuracy}, F1 Score: {dt_f1}")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score

# Define Random Forest model
rf_model = RandomForestClassifier(random_state=42)

# Define SVM model
svm_model = SVC(random_state=42)

# Define Decision Tree model
dt_model = DecisionTreeClassifier(random_state=42)

# Define cross-validation function
def cross_validate_model(model, X, y, cv=5):
    kf = KFold(n_splits=cv, shuffle=True, random_state=42)
    accuracy_scores = cross_val_score(model, X, y, cv=kf, scoring='accuracy')
    f1_scores = cross_val_score(model, X, y, cv=kf, scoring='f1_macro')
    return accuracy_scores, f1_scores

# Random Forest
rf_accuracy_scores, rf_f1_scores = cross_validate_model(rf_model, X, y)
print(f"Random Forest Cross-Validation Scores: {rf_accuracy_scores.mean()}")

# SVM
svm_accuracy_scores, svm_f1_scores = cross_validate_model(svm_model, X, y)
print(f"SVM Cross-Validation Scores: {svm_accuracy_scores.mean()}")

# Decision Tree
dt_accuracy_scores, dt_f1_scores = cross_validate_model(dt_model, X, y)
print(f"Decision Tree Cross-Validation Scores: {dt_accuracy_scores.mean()}")

# Random Forest
rf_accuracy_scores, rf_f1_scores = cross_validate_model(rf_model, X, y)
rf_accuracy = rf_accuracy_scores.mean()
rf_f1 = rf_f1_scores.mean()

# SVM
svm_accuracy_scores, svm_f1_scores = cross_validate_model(svm_model, X, y)
svm_accuracy = svm_accuracy_scores.mean()
svm_f1 = svm_f1_scores.mean()

# Decision Tree
dt_accuracy_scores, dt_f1_scores = cross_validate_model(dt_model, X, y)
dt_accuracy = dt_accuracy_scores.mean()
dt_f1 = dt_f1_scores.mean()

# Memilih model terbaik berdasarkan metrik yang diinginkan
best_model = None
if rf_accuracy > svm_accuracy and rf_accuracy > dt_accuracy:
    best_model = rf_model
    best_model_name = "Random Forest"
    best_model_accuracy = rf_accuracy
    best_model_f1 = rf_f1
elif svm_accuracy > rf_accuracy and svm_accuracy > dt_accuracy:
    best_model = svm_model
    best_model_name = "SVM"
    best_model_accuracy = svm_accuracy
    best_model_f1 = svm_f1
else:
    best_model = dt_model
    best_model_name = "Decision Tree"
    best_model_accuracy = dt_accuracy
    best_model_f1 = dt_f1

print(f"\nModel terbaik adalah {best_model_name} dengan Accuracy: {best_model_accuracy} dan F1-Score: {best_model_f1}")

import joblib

# Simpan model
joblib.dump(rf_model, 'random_forest_model.pkl')
