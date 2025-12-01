import pandas as pd
import gzip
import pickle
import json
import os
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_score, balanced_accuracy_score, recall_score, f1_score, confusion_matrix

# ================================ #
# Step 1: Load and clean data
# ================================ #
def clean_data(df):
    df = df.copy()
    df.drop(columns='ID', inplace=True)
    df.rename(columns={'default payment next month': 'default'}, inplace=True)
    df.dropna(inplace=True)
    df = df[(df['EDUCATION'] != 0) & (df['MARRIAGE'] != 0)]
    df.loc[df['EDUCATION'] > 4, 'EDUCATION'] = 4
    return df

# ================================ #
# Step 2: Split data
# ================================ #
def split_data(train_df, test_df):
    x_train = train_df.drop(columns="default")
    y_train = train_df["default"]
    x_test = test_df.drop(columns="default")
    y_test = test_df["default"]
    return x_train, y_train, x_test, y_test

# ============================== #
# Step 3: Create pipeline
# ============================== #
def make_pipeline():
    # Categorical and numerical variables
    categorical_features = ['SEX', 'EDUCATION', 'MARRIAGE']
    numerical_features = [
        "LIMIT_BAL", "AGE", "PAY_0", "PAY_2", "PAY_3", "PAY_4",
        "PAY_5", "PAY_6", "BILL_AMT1", "BILL_AMT2", "BILL_AMT3",
        "BILL_AMT4", "BILL_AMT5", "BILL_AMT6", "PAY_AMT1", "PAY_AMT2",
        "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6"
    ]

    # Preprocessor: one-hot for categorical, standard scaler for numerical
    preprocessor = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ("num", StandardScaler(), numerical_features),
    ])

    # Complete pipeline
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("feature_selection", SelectKBest(score_func=f_classif)),
        ("pca", PCA()),
        ("classifier", MLPClassifier(max_iter=15000, random_state=17)),
    ])
    return pipeline


# ========================================== #
# Step 4: Hyperparameter optimization
# ========================================== #
def optimize_hyperparameters(pipeline):
    param_grid = {
        'pca__n_components': [None],
        'feature_selection__k': [20],
        'classifier__hidden_layer_sizes': [(50, 30, 40, 60)],
        'classifier__alpha': [0.26],
        'classifier__learning_rate_init': [0.001],
    }
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=10,
        scoring='balanced_accuracy',
        n_jobs=-1,
        verbose=2
    )
    return grid_search


# ============================ #
# Step 5: Calculate metrics
# ============================ #
def calculate_metrics(model, x, y, dataset_name):
    y_pred = model.predict(x)
    metrics = {
        "type": "metrics",
        "dataset": dataset_name,
        "precision": round(precision_score(y, y_pred), 4),
        "balanced_accuracy": round(balanced_accuracy_score(y, y_pred), 4),
        "recall": round(recall_score(y, y_pred), 4),
        "f1_score": round(f1_score(y, y_pred), 4)
    }
    return y_pred, metrics


# ================================ #
# Step 6: Confusion matrix
# ================================ #
def calculate_confusion_matrix(x, y, dataset_name):
    cm = confusion_matrix(x, y)
    return {
        "type": "cm_matrix",
        "dataset": dataset_name,
        "true_0": {"predicted_0": int(cm[0][0]), "predicted_1": int(cm[0][1])},
        "true_1": {"predicted_0": int(cm[1][0]), "predicted_1": int(cm[1][1])}
    }


# ================================ #
# Step 7: Save model
# ================================ #
def save_model(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with gzip.open(path, 'wb') as f:
        pickle.dump(model, f)


# ================================ #
# Step 8: Save metrics
# ================================ #
def save_metrics(records, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for item in records:
            f.write(json.dumps(item) + '\n')


# ============================ #
# MAIN EXECUTION
# ============================ #
if __name__ == "__main__":
    print("Loading data...")
    train_df = pd.read_csv("files/input/train_data.csv.zip")
    test_df = pd.read_csv("files/input/test_data.csv.zip")

    print("Data loaded.")
    train_df = clean_data(train_df)
    test_df = clean_data(test_df)

    x_train, y_train, x_test, y_test = split_data(train_df, test_df)

    print("Creating pipeline...")
    pipeline = make_pipeline()

    print("Fitting model...")
    estimator = optimize_hyperparameters(pipeline)
    model = estimator.fit(x_train, y_train)

    print("Model trained.")
    save_model(estimator, 'files/models/model.pkl.gz')

    print("Calculating performance...")
    y_pred_train, train_metrics = calculate_metrics(model, x_train, y_train, "train")
    y_pred_test, test_metrics = calculate_metrics(model, x_test, y_test, "test")

    cm_train = calculate_confusion_matrix(y_train, y_pred_train, 'train')
    cm_test = calculate_confusion_matrix(y_test, y_pred_test, 'test')

    save_metrics(
        [train_metrics, test_metrics, cm_train, cm_test],
        'files/output/metrics.json'
    )

    print("Completed.")
