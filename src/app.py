import streamlit as st
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from io import StringIO

# Streamlit app title
st.title("Regularized Model with Noise Injection")

# Upload dataset
uploaded_file = st.file_uploader("Upload your dataset (CSV format)", type="csv")
if uploaded_file is not None:
    # Load the dataset
    data = pd.read_csv(uploaded_file)

    # Display dataset
    st.write("## Dataset Preview")
    st.write(data.head())

    # Select features and target
    X = data.drop(columns=['Star category', 'Star type', 'Spectral Class', 'Star color'])
    y = data['Star category']

    # Standardize the features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    #### Train the first model on the original data ####
    original_model = DecisionTreeClassifier(max_depth=None, random_state=42)
    original_model.fit(X_train, y_train)

    # Predict on training and testing data (Original Model)
    y_train_pred_orig = original_model.predict(X_train)
    y_test_pred_orig = original_model.predict(X_test)

    # Evaluate performance of the original model
    train_accuracy_orig = accuracy_score(y_train, y_train_pred_orig)
    test_accuracy_orig = accuracy_score(y_test, y_test_pred_orig)
    
    # Display metrics for the original model in a table
    st.write("## Original Model Performance")

    # Prepare the metrics as a DataFrame
    metrics = pd.DataFrame({
        "Metric": ["Training Accuracy", "Testing Accuracy"],
        "Value": [f"{train_accuracy_orig:.2f}", f"{test_accuracy_orig:.2f}"]
    })

    # Display the metrics table
    st.table(metrics)

    # Display the classification report as a table
    st.write("**Classification Report (Test - Original Model):**")

    # Convert classification report to a DataFrame
    report = classification_report(y_test, y_test_pred_orig, output_dict=True)
    report_df = pd.DataFrame(report).transpose()

    # Display the classification report table
    st.dataframe(report_df.style.format({"precision": "{:.2f}", "recall": "{:.2f}", "f1-score": "{:.2f}"}))
    
    train_sizes, train_scores, test_scores = learning_curve(
    original_model, X, y, cv=5, scoring='accuracy', train_sizes=np.linspace(0.1, 1.0, 10), random_state=42
)

    # Calculate mean and std for plotting
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    # Plot learning curve
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(train_sizes, train_mean, label="Training Score", color="red")
    ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color="red")
    ax.plot(train_sizes, test_mean, label="Test Score", color="blue")
    ax.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1, color="blue")
    ax.set_title("Learning Curve")
    ax.set_xlabel("Training Size")
    ax.set_ylabel("Accuracy")
    ax.legend(loc="best")

    # Display the plot in Streamlit
    st.write("## Learning Curve")
    st.pyplot(fig)

    #### Augment the data by injecting noise ####
    noise = np.random.normal(loc=0, scale=0.1, size=X_train.shape)  # Weak Gaussian noise
    X_train_noisy = X_train + noise

    #### Train the second model on the noisy data ####
    regularized_model = DecisionTreeClassifier(max_depth=None, random_state=42)
    regularized_model.fit(X_train_noisy, y_train)

    # Predict on training and testing data (Regularized Model)
    y_train_pred_reg = regularized_model.predict(X_train_noisy)
    y_test_pred_reg = regularized_model.predict(X_test)

    # Evaluate performance of the regularized model
    train_accuracy_reg = accuracy_score(y_train, y_train_pred_reg)
    test_accuracy_reg = accuracy_score(y_test, y_test_pred_reg)
    
    # Display metrics for the regularized model in a table
    st.write("## Regularized Model Performance (With Noise Injection)")

    # Prepare the metrics as a DataFrame
    regularized_metrics = pd.DataFrame({
        "Metric": ["Training Accuracy", "Testing Accuracy"],
        "Value": [f"{train_accuracy_reg:.2f}", f"{test_accuracy_reg:.2f}"]
    })

    # Display the metrics table
    st.table(regularized_metrics)

    # Display the classification report as a table
    st.write("**Classification Report (Test - Regularized Model):**")

    # Convert classification report to a DataFrame
    regularized_report = classification_report(y_test, y_test_pred_reg, output_dict=True)
    regularized_report_df = pd.DataFrame(regularized_report).transpose()

    # Display the classification report table
    st.dataframe(regularized_report_df.style.format({
        "precision": "{:.2f}",
        "recall": "{:.2f}",
        "f1-score": "{:.2f}"
    }))

    #### Learning Curve for Regularized Model ####
    train_sizes, train_scores, test_scores = learning_curve(
        regularized_model, X_train_noisy, y_train, cv=5, scoring='accuracy', train_sizes=np.linspace(0.1, 1.0, 10)
    )

    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(train_sizes, train_mean, label="Training Score", color="red")
    ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color="red")
    ax.plot(train_sizes, test_mean, label="Cross-Validation Score", color="blue")
    ax.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1, color="blue")
    ax.set_title("Learning Curve (Regularized Model with Noise Injection)")
    ax.set_xlabel("Training Size")
    ax.set_ylabel("Accuracy")
    ax.legend(loc="best")
    st.pyplot(fig)

else:
    st.info("Please upload a dataset to proceed.")
