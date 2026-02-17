import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle

# Load Data
df = pd.read_csv("data_cts_violent_and_sexual_crime.csv")

# 1. Handle Missing Values
df = df.dropna()

# 2. Convert Categorical to Numerical
categorical_cols = df.select_dtypes(include='object').columns
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# 3. Separate features (X) and target (y)
X = df.drop('Indicator', axis=1)
y = df['Indicator']

# 4. Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# 5. Normalize/Scale Numerical Features
scaler = StandardScaler()
numerical_cols = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
if 'Year' in numerical_cols:
    numerical_cols.remove('Year') # Year might not need scaling

X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])


# 6. Train the Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 7. Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred, output_dict=True)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f"Model Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
print("Classification Report:")
print(classification_report(y_test, y_pred))

# 8. Save the model, preprocessing objects, and evaluation metrics
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('label_encoders.pkl', 'wb') as f:
    pickle.dump(label_encoders, f)

with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# Save evaluation metrics
model_metrics = {
    'accuracy': accuracy,
    'classification_report': classification_rep,
    'confusion_matrix': conf_matrix.tolist(),
    'y_test': y_test.tolist(),
    'y_pred': y_pred.tolist()
}

with open('model_metrics.pkl', 'wb') as f:
    pickle.dump(model_metrics, f)

print("Model, preprocessing objects, and evaluation metrics saved successfully!")