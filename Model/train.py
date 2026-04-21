import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, roc_auc_score,precision_recall_curve,precision_score, recall_score
import joblib
from Api.config import settings
import mlflow
import mlflow.sklearn

def clean_data():
    df = pd.read_csv("Data/raw/cs-training.csv")
    df.drop(columns=["Unnamed: 0"],inplace=True)

    df["MonthlyIncome"]=df["MonthlyIncome"].fillna(df["MonthlyIncome"].median())
    df["NumberOfDependents"]=df["NumberOfDependents"].fillna(df["NumberOfDependents"].median())
    df["RevolvingUtilizationOfUnsecuredLines"] = df["RevolvingUtilizationOfUnsecuredLines"].clip(0, 1)
    df["DebtRatio"] = df["DebtRatio"].clip(0, 1)
    df["NumberOfTime30-59DaysPastDueNotWorse"] = df["NumberOfTime30-59DaysPastDueNotWorse"].clip(0, 10)
    df["NumberOfTimes90DaysLate"] = df["NumberOfTimes90DaysLate"].clip(0, 10)
    df["NumberOfTime60-89DaysPastDueNotWorse"] = df["NumberOfTime60-89DaysPastDueNotWorse"].clip(0, 10)
    return df


def Train_Model():

    mlflow.set_experiment("creditguard-credit-risk")

    with mlflow.start_run():
    

        #load data and preprocessing
        df = clean_data()

        col_to_scale = [
            'RevolvingUtilizationOfUnsecuredLines',
            'DebtRatio',
            'MonthlyIncome',
            'age'
        ]

        preprocessor = ColumnTransformer(
            transformers=[("scaler" , StandardScaler(),col_to_scale)],
            remainder= "passthrough"
        )


        #Model Creation

        model = XGBClassifier(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=5,
                random_state=42,
                eval_metric='auc'
            )

        pipeline = Pipeline(steps=[
            ('preprocessor' , preprocessor),
            ("model" , model)
        ])

        X = df.drop(columns=["SeriousDlqin2yrs"])
        y = df["SeriousDlqin2yrs"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42 , stratify=y)

        smote = SMOTE(random_state=42)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
        pipeline.fit(X_train_resampled, y_train_resampled)


        #Model Evaluation

        y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
        precisions, recalls, thresholds = precision_recall_curve(y_test, y_pred_proba)

        f1_scores = 2 * (precisions * recalls) / (precisions + recalls)
        best_threshold = thresholds[np.argmax(f1_scores)]
        y_pred_best = (y_pred_proba >= best_threshold).astype(int)

        auc_score = roc_auc_score(y_test, y_pred_proba)
        precision_cls1 = precision_score(y_test, y_pred_best)
        recall_cls1 = recall_score(y_test, y_pred_best)


        print(f"Threshold: {best_threshold:.4f}")
        print(classification_report(y_test, y_pred_best))
        print(f"AUC-ROC: {auc_score:.4f}")


        #mlflow tracking

        mlflow.log_param("n_estimators",200)
        mlflow.log_param("learning_rate" , 0.05)
        mlflow.log_param("max_depth",5)

        mlflow.log_metric("auc_roc",float(auc_score))
        mlflow.log_metric("best_threshold",float(best_threshold))
        mlflow.log_metric("precision_class_1",precision_cls1)
        mlflow.log_metric("recall_class_1",recall_cls1)

        mlflow.sklearn.log_model(pipeline,"credit_pipeline")

        #Saving Model
        joblib.dump(pipeline, settings.credit_model_path)
        joblib.dump(best_threshold, settings.credit_threshold_path)
        print("Model Saved Successfully")

if __name__ == "__main__":
    Train_Model()