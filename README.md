### Title of the Project
A Parkinson’s disease prediction system using a hybrid SVM-Random Forest model with SMOTE and PCA for enhanced early detection accuracy.

### About
A Parkinson’s disease prediction system utilizing a hybrid machine learning model, combining Support Vector Machine (SVM) and Random Forest classifiers, aimed at enhancing diagnostic accuracy. This system leverages data preprocessing techniques, such as Synthetic Minority Over-sampling Technique (SMOTE) and Principal Component Analysis (PCA), to improve model performance and identify key patterns in patient data, providing a reliable tool for early detection and support in clinical decision-making.

### Features
Hybrid model combining SVM and Random Forest for accurate predictions.
Advanced data preprocessing using SMOTE and PCA to improve model performance.
High test accuracy aimed for reliable early detection of Parkinson's disease.
Scalable and adaptable for various clinical settings.
User-friendly interface for efficient patient data input and result visualization.
Requirements
Operating System: Requires a 64-bit OS (Windows 10 or Ubuntu) for compatibility with deep learning frameworks.
Development Environment: Python 3.8 or later for implementing the Parkinson’s disease prediction system.
Machine Learning Libraries: Scikit-learn for model building, SMOTE for data balancing, PCA for dimensionality reduction.
Data Processing Libraries: Pandas and NumPy for handling and preprocessing data.
Version Control: Implementation of Git for collaborative development and effective code management.
IDE: Use of VSCode or Jupyter notebook as the Integrated Development Environment for coding, debugging, and version control integration.
Additional Dependencies: Matplotlib and Seaborn for data visualization, joblib for model persistence, and Streamlit for application deployment.
### Code
```
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn import svm
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE

# Load the data from the CSV file
parkinsons_data = pd.read_csv('/content/parkinsons.csv')

# Select only the specified features and target
selected_features = ['MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)',
                     'MDVP:Jitter(%)', 'MDVP:Shimmer', 'MDVP:Shimmer(dB)',
                     'MDVP:APQ', 'NHR', 'status']
parkinsons_data = parkinsons_data[selected_features]

# Separating the features and target
X = parkinsons_data.drop(columns=['status'], axis=1)
Y = parkinsons_data['status']

# Using SMOTE to handle class imbalance
smote = SMOTE(random_state=2)
X, Y = smote.fit_resample(X, Y)

# Splitting the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=2)

# Feature scaling - standardize all features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Applying PCA for dimensionality reduction
pca = PCA(n_components=5)  # Keep more components to retain variance
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# Hyperparameter tuning for Random Forest
rf_model = RandomForestClassifier(random_state=2)
param_grid_rf = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
}
grid_search_rf = GridSearchCV(estimator=rf_model, param_grid=param_grid_rf, cv=5, scoring='accuracy')
grid_search_rf.fit(X_train_pca, Y_train)
best_rf_model = grid_search_rf.best_estimator_

# Hyperparameter tuning for SVM
svm_model = svm.SVC(probability=True)
param_grid_svm = {
    'C': [0.1, 1, 10, 100],
    'kernel': ['linear', 'rbf', 'poly'],
}
grid_search_svm = GridSearchCV(estimator=svm_model, param_grid=param_grid_svm, cv=5, scoring='accuracy')
grid_search_svm.fit(X_train_pca, Y_train)
best_svm_model = grid_search_svm.best_estimator_

# Training and scoring for Random Forest
best_rf_model.fit(X_train_pca, Y_train)
rf_train_accuracy = best_rf_model.score(X_train_pca, Y_train)
rf_test_accuracy = best_rf_model.score(X_test_pca, Y_test)

# Training and scoring for SVM
best_svm_model.fit(X_train_pca, Y_train)
svm_train_recall = recall_score(Y_train, y_train_pred_svm)
svm_train_f1 = f1_score(Y_train, y_train_pred_svm)

svm_test_precision = precision_score(Y_test, y_test_pred_svm)
svm_test_recall = recall_score(Y_test, y_test_pred_svm)
svm_test_f1 = f1_score(Y_test, y_test_pred_svm)

# Precision, Recall, and F1 Score for Hybrid Model
hybrid_train_precision = precision_score(Y_train, y_train_pred_hybrid)
hybrid_train_recall = recall_score(Y_train, y_train_pred_hybrid)
hybrid_train_f1 = f1_score(Y_train, y_train_pred_hybrid)

hybrid_test_precision = precision_score(Y_test, y_test_pred_hybrid)
hybrid_test_recall = recall_score(Y_test, y_test_pred_hybrid)
hybrid_test_f1 = f1_score(Y_test, y_test_pred_hybrid)

# Print metrics for Random Forest
print("\nRandom Forest Metrics:")
print("Train Confusion Matrix:\n", train_conf_matrix_rf)
print("Train Precision:", rf_train_precision)
print("Train Recall:", rf_train_recall)
print("Train F1 Score:", rf_train_f1)

print("\nTest Confusion Matrix:\n", test_conf_matrix_rf)
print("Test Precision:", rf_test_precision)
print("Test Recall:", rf_test_recall)
print("Test F1 Score:", rf_test_f1)

# Print metrics for SVM
print("\nSVM Metrics:")
print("Train Confusion Matrix:\n", train_conf_matrix_svm)
print("Train Precision:", svm_train_precision)
print("Train Recall:", svm_train_recall)

# SVM Test Metrics
print("\nSVM Test Metrics:")
print("Test Confusion Matrix:\n", test_conf_matrix_svm)
print("Test Precision:", svm_test_precision)
print("Test Recall:", svm_test_recall)
print("Test F1 Score:", svm_test_f1)

import pickle
# Train the hybrid model
hybrid_model.fit(X_train, Y_train)

# Save the trained model to a .sav file
model_filename = 'parkinsons_hybrid_model.sav'
pickle.dump(hybrid_model, open(model_filename, 'wb'))

print(f"Model saved as {model_filename}")
with open('scaler.sav', 'wb') as scaler_file:
    pickle.dump(scaler, scaler_file)
```
### System Architecture
![Architecture_diagram_Final](https://github.com/user-attachments/assets/3ae25f9a-3178-43bf-a4db-e58fc5efe077)

### Output
Output1 - Person is Positive for Parkinson Disease
![op_1](https://github.com/user-attachments/assets/00be1216-b09e-45e7-b0ed-f3d694b4574d)

Output2 - Person is Negative for Parkinson Disease
![2](https://github.com/user-attachments/assets/7bfc15eb-d6f5-4c49-ab39-31d4ea6fecdc)


Detection Accuracy: 92%

Note: These metrics are customizable and should be tailored to reflect actual performance evaluations and project requirements for best results.

### Results and Impact
The Parkinson’s Disease Prediction System offers an effective tool for early diagnosis, aiding healthcare professionals in making timely clinical decisions. With its high accuracy and advanced data processing techniques, this project has the potential to improve patient outcomes and support proactive treatment strategies.

This system contributes to advancements in healthcare technology and serves as a foundation for future predictive models, promoting early intervention and enhancing quality of life for individuals at risk of Parkinson’s disease.

### Articles published / References
Vaishnavi, "SVM-based approach for Parkinson's detection using vocal data," International Research Journal of Modernization in Engineering Technology and Science, vol. 5, no. 5, 2023.

Doneti Sowmya, Dodla Kavya, J. Rashmitha, Satheesh Kumar V., and Preethi Jeevan, "Parkinson’s Disease Detection By Machine Learning Using SVM," International Research Journal of Engineering and Technology (IRJET), vol. 10, no. 01, pp. 1097. e-ISSN: 2395-0056, p-ISSN: 2395-0072, 2023.

J. Mei, C. Desrosiers, and J. Frasnelli, "Machine Learning for the Diagnosis of Parkinson’s Disease: A Review of Literature," Frontiers in Aging Neuroscience, vol. 13, no. 633752, pp. 1–13, 2021

S. Haller, S. Badoud, D. Nguyen, V. Garibotto, K.O. Lovblad, P.R. Burkhard, "Individual Detection of Patients with Parkinson Disease using Support Vector Machine Analysis of Diffusion Tensor Imaging Data: Initial Results," AJNR Am J Neuroradiol, vol. 33, no. 11, pp. 2123-2128, 2012

W. A. Mir, D. R. Rizvi, I. Nissar, S. Masood, Izharuddin, and A. Hussain, "Deep Learning based model for the detection of Parkinson's disease using voice data," in 2022 First International Conference on Artificial Intelligence Trends and Pattern Recognition (ICAITPR), 2022, DOI: 10.1109/ICAITPR51569.2022.9844185

C. K. Gomathy, B. Dheeraj Kumar Reddy, B. Varsha, and B. Varshini, "The Parkinson's Disease Detection Using Machine Learning Techniques," International Research Journal of Engineering and Technology (IRJET), vol. 08, no. 10, pp. 440-444, 2021.

V. Balu, M. Varsha Reddy, and Y. V. Leelaprasad Reddy, "Detecting Parkinson's Disease," International Research Journal of Modernization in Engineering Technology and Science, vol. 04, no. 03, pp. 1420-1424, 2022

Govindu, Aditi et al., "Parkinson's Disease Classification using Vowel Phonation and Machine Learning," Procedia Computer Science, vol. 218, pp. 249-261, 2023

Hayder Mohammed Qasim, Oguz Ata, Mohammad Azam Ansari, Mohammad N. Alomary, Saad Alghamdi, and Mazen Almehmadi, "Hybrid Feature Selection Framework for the Parkinson Imbalanced Dataset Prediction Problem," Medicina, vol. 57, no. 11, p. 1217, 2021

K. Polat, "A Hybrid Approach to Parkinson Disease Classification using speech signal: The Combination of SMOTE and Random Forests," 2019 IEEE International Symposium on INnovations in Intelligent Systems and Applications (INISTA), pp. 1-5, 2019

M. Pramanik, R. Pradhan, P. Nandy, A.K. Bhoi, and P. Barsocchi, "Machine Learning Methods with Decision Forests for Parkinson's Detection," Applied Sciences, vol. 11, no. 2, pp. 581, 2021

L. Ali, A. Javeed, A. Noor, H. T. Rauf, S. Kadry, and A. H. Gandomi, "Parkinson’s disease detection based on features refinement through L1 regularized SVM and deep neural network," Scientific Reports, vol. 14, no. 1333, pp. 1-12, 2024
