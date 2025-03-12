import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from flask import Flask, request, render_template, send_file
from sklearn.model_selection import train_test_split, KFold
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Initialize Flask app
app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

ALLOWED_EXTENSIONS = {'csv'}

# Check if file extension is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Preprocess dataset (handle missing values, encode categorical data, scale numeric features)
def preprocess_data(df):
    df.fillna(df.mean(numeric_only=True), inplace=True)
    df.fillna('', inplace=True)

    label_encoders = {}
    for column in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column].astype(str))
        label_encoders[column] = le

    X = df.iloc[:, :-1]  # Features
    y = df.iloc[:, -1]   # Target variable

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, df

# Function to evaluate models using K-Fold Cross Validation
def evaluate_model(model, X, y):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    acc_scores, prec_scores, rec_scores, f1_scores = [], [], [], []

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc_scores.append(accuracy_score(y_test, y_pred))
        prec_scores.append(precision_score(y_test, y_pred, average='weighted', zero_division=0))
        rec_scores.append(recall_score(y_test, y_pred, average='weighted', zero_division=0))
        f1_scores.append(f1_score(y_test, y_pred, average='weighted'))

    return {
        'accuracy': round(np.mean(acc_scores) * 100, 2),
        'precision': round(np.mean(prec_scores) * 100, 2),
        'recall': round(np.mean(rec_scores) * 100, 2),
        'f1_score': round(np.mean(f1_scores) * 100, 2)
    }

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'file' not in request.files:
        return "No file uploaded!"

    file = request.files['file']
    if file.filename == '' or not allowed_file(file.filename):
        return "Invalid file type. Only CSV files are allowed!"

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    df = pd.read_csv(filepath)
    X, y, processed_df = preprocess_data(df)

    # Define models and their parameters
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'Decision Tree': DecisionTreeClassifier(max_depth=5),
        'Random Forest': RandomForestClassifier(n_estimators=100),
        'SVM': SVC(kernel='linear'),
        'KNN': KNeighborsClassifier(n_neighbors=5),
        'Naive Bayes': GaussianNB()
    }

    results = []
    parameters = []

    for model_name, model in models.items():
        metrics = evaluate_model(model, X, y)
        results.append({'model': model_name, **metrics})
        parameters.append({'model': model_name, 'parameters': model.get_params()})

    # Train Decision Tree for visualization
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    dt_model = DecisionTreeClassifier(max_depth=5)
    dt_model.fit(X_train, y_train)

    plt.figure(figsize=(12, 6))
    plot_tree(dt_model, feature_names=processed_df.columns[:-1], filled=True, class_names=[str(cls) for cls in y.unique()])
    tree_path = os.path.join("static", "decision_tree.png")
    plt.savefig(tree_path)
    plt.close()

    return render_template('results.html', results=results, parameters=parameters, tree_image="static/decision_tree.png")

@app.route('/decision_tree')
def decision_tree():
    return send_file("static/decision_tree.png", mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True)
