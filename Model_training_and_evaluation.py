import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib

# Load historical bug data and code metrics
data = pd.read_csv('/Users/fkhanamzm/Downloads/sample_bug_data.csv')

# Data cleaning
data.dropna(inplace=True)
#print(data)
# Feature selection
features = ['lines_of_code', 'cyclomatic_complexity', 'code_churn', 'complexity_per_loc']
X = data[features]
y = data['bug_present']

# Data normalization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
# Train a Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate model performance
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred)
'''
print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1 Score: {f1}')
print(f'ROC-AUC: {roc_auc}')
'''
# Save the trained model
joblib.dump(model, 'random_forest_model.pkl')


# Sample new data

new_data = {
    'commit_id': ['bf12f48', '8391a37'],
    'lines_of_code': [545, 549],
    'cyclomatic_complexity': [134, 134],
    'lines_added': [16, 5],
    'lines_deleted': [24, 1]
}

#new_data = pd.read_csv('/Users/fkhanamzm/new_data.csv')
# Create DataFrame for new data
new_df = pd.DataFrame(new_data)

# Feature engineering for new data
new_df['code_churn'] = new_df['lines_added'] + new_df['lines_deleted']
new_df['complexity_per_loc'] = new_df['cyclomatic_complexity'] / new_df['lines_of_code']

# Select features for prediction
features = ['lines_of_code', 'cyclomatic_complexity', 'code_churn', 'complexity_per_loc']
X_new = new_df[features]

# Load the trained model
model = joblib.load('random_forest_model.pkl')

# Data normalization
scaler = StandardScaler()
X_new_scaled = scaler.fit_transform(X_new)

# Make predictions for new data
predictions = model.predict(X_new_scaled)

# Add predictions to the DataFrame
new_df['bug_prediction'] = predictions

# Display the predictions
print(new_df[['commit_id', 'bug_prediction']])
