import pandas as pd
import mlflow
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import numpy as np
from sklearn.metrics import  classification_report, roc_auc_score,precision_recall_curve,precision_score, recall_score
from Api.config import settings
import joblib
import mlflow.sklearn


def load_data():

    df = pd.read_csv("Data/raw/creditcard.csv")
    df = df.dropna()

    return df

def Train_Model():

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment("creditguard-fraud")

    with mlflow.start_run() as run:

        df = load_data()
        col_to_scale = ["Amount","Time"]

        preprocessor = ColumnTransformer(
            transformers=[("scaler",StandardScaler(),col_to_scale)],
            remainder="passthrough"
        )

        model = XGBClassifier(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=5,
                random_state=42,
                eval_metric='auc'
                )
        
        pipeline = Pipeline(steps=[
            ("preprocessor",preprocessor),
            ("model",model)
        ]
        )

        X = df.drop(columns=["Class"])
        y = df["Class"]

        X_train ,X_test ,y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)

        smote = SMOTE()
        X_train_resample , y_train_resample = smote.fit_resample(X_train,y_train)
        pipeline.fit(X_train_resample,y_train_resample)
        

        y_pred_proba = pipeline.predict_proba(X_test)[:,1]
        precisions , recalls ,thresholds = precision_recall_curve(y_test , y_pred_proba)

        f1_scores = 2 * (precisions * recalls) / (precisions + recalls)
        best_threshold = thresholds[np.argmax(f1_scores)]
        y_pred_best = (y_pred_proba >= best_threshold).astype(int)

        auc_score = roc_auc_score(y_test, y_pred_proba)
        precision_cls1 = precision_score(y_test, y_pred_best)
        recall_cls1 = recall_score(y_test, y_pred_best)

        print(f"Threshold: {best_threshold:.4f}")
        print(classification_report(y_test, y_pred_best))
        print(f"AUC-ROC: {auc_score:.4f}")

        mlflow.log_param("n_estimators",200)
        mlflow.log_param("learning_rate" , 0.05)
        mlflow.log_param("max_depth",5)

        mlflow.log_metric("auc_roc",float(auc_score))
        mlflow.log_metric("best_threshold",float(best_threshold))
        mlflow.log_metric("precision_class_1",precision_cls1)
        mlflow.log_metric("recall_class_1",recall_cls1)

        mlflow.sklearn.log_model(pipeline,"fraud_pipeline")

        joblib.dump(pipeline, settings.fraud_model_path)
        joblib.dump(best_threshold, settings.fraud_threshold_path)
        print("Model Saved Successfully")
        return run.info.run_id 
    
if __name__ == "__main__":
    run_id = Train_Model()
    print(f"\n✓ Training complete. MLflow run_id: {run_id}")

