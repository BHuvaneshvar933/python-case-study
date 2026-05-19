import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.neural_network import MLPClassifier
import joblib

def run_project():
    print("Starting E-commerce Churn Prediction using Neural Network...")
    
    # Setup Output Directory
    output_dir = 'output_plots'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # 1. Data Collection
    print("\n--- 1. Data Collection ---")
    file_path = 'ecommerce_data.csv'
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found.")
        return
        
    df = pd.read_csv(file_path)
    print(f"Loaded dataset with {df.shape[0]} rows and {df.shape[1]} columns.")

    # 2. Exploratory Data Analysis (EDA)
    print("\n--- 2. Exploratory Data Analysis ---")
    
    # Plot 1: Churn Distribution (Bar)
    plt.figure(figsize=(6,4))
    sns.countplot(data=df, x='Churn', hue='Churn', palette='viridis', legend=False)
    plt.title("Churn Distribution")
    plt.savefig(f"{output_dir}/1_Churn_Distribution_Bar.png")
    plt.close()
    
    # Plot 2: Churn Distribution (Pie)
    plt.figure(figsize=(6,6))
    df['Churn'].value_counts().plot.pie(autopct='%1.1f%%', colors=['#4daf4a', '#e41a1c'])
    plt.title("Churn Percentage")
    plt.savefig(f"{output_dir}/2_Churn_Percentage_Pie.png")
    plt.close()

    # Plot 3: Age Distribution
    plt.figure(figsize=(8,5))
    sns.histplot(data=df, x='Age', kde=True, bins=20, color='skyblue')
    plt.title("Age Distribution")
    plt.savefig(f"{output_dir}/3_Age_Distribution.png")
    plt.close()
    
    # Plot 4: Annual Income Distribution
    plt.figure(figsize=(8,5))
    sns.histplot(data=df, x='AnnualIncome', kde=True, bins=20, color='salmon')
    plt.title("Annual Income Distribution")
    plt.savefig(f"{output_dir}/4_Income_Distribution.png")
    plt.close()
    
    # Plot 5: Preferred Device vs Churn
    plt.figure(figsize=(8,5))
    sns.countplot(data=df, x='PreferredDevice', hue='Churn', palette='Set2')
    plt.title("Preferred Device vs Churn")
    plt.savefig(f"{output_dir}/5_Device_vs_Churn.png")
    plt.close()
    
    # Plot 6: Spending Score vs Churn
    plt.figure(figsize=(8,5))
    sns.boxplot(data=df, x='Churn', y='SpendingScore', hue='Churn', palette='pastel', legend=False)
    plt.title("Spending Score vs Churn")
    plt.savefig(f"{output_dir}/6_SpendingScore_vs_Churn.png")
    plt.close()
    
    print(f"EDA completed. Plots saved to '{output_dir}/' directory.")

    # 3. Data Preparation
    print("\n--- 3. Data Preparation ---")
    initial_rows = df.shape[0]
    df = df.drop_duplicates().dropna()
    print(f"Removed {initial_rows - df.shape[0]} duplicate/missing rows.")
    
    if 'CustomerID' in df.columns:
        df = df.drop('CustomerID', axis=1)

    label_encoders = {}
    categorical_cols = ['Gender', 'PreferredDevice', 'IsSubscribed', 'Churn']
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    X = df.drop('Churn', axis=1)
    y = df['Churn']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    print("Data preprocessed, scaled, and split (80% train / 20% test).")

    # 4. Data Modelling (Neural Network)
    print("\n--- 4. Data Modelling ---")
    print("Training Neural Network Architecture (64 -> 32 -> 1) using MLPClassifier for 1000 epochs...")
    model = MLPClassifier(hidden_layer_sizes=(64, 32), activation='relu', solver='adam', max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    
    # Plot Training Loss
    plt.figure(figsize=(8,5))
    plt.plot(model.loss_curve_, label='Training Loss')
    plt.title('Neural Network Training Loss Curve (1000 Epochs)')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f"{output_dir}/7_Training_Loss_Curve.png")
    plt.close()
    print("Loss curve saved.")

    # 5. Model Evaluation
    print("\n--- 5. Model Evaluation ---")
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    print(f"Neural Network Accuracy: {acc * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Confusion Matrix Plot
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Stayed', 'Churned'], yticklabels=['Stayed', 'Churned'])
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig(f"{output_dir}/8_Confusion_Matrix.png")
    plt.close()
    
    model_path = 'ecommerce_nn_model.pkl'
    joblib.dump(model, model_path)
    print(f"Trained model saved to: {model_path}")
    print("Project successfully completed!")

if __name__ == "__main__":
    run_project()
