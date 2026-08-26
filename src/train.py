import joblib
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from xgboost import XGBClassifier

from src.config import (
    MLFLOW_EXPERIMENT_NAME,
    MODEL_PATH,
    N_SPLITS,
    RANDOM_STATE,
    TARGET_COL,
    TEST_SIZE,
)
from src.features import build_full_pipeline

SCORING = ["accuracy", "precision", "recall", "f1", "roc_auc"]
FINAL_MODEL_NAME = "Logistic Regression (balanced)"


def split_train_test(df_outcomes, target_col=TARGET_COL):
    X = df_outcomes.drop(columns=[target_col])
    y = df_outcomes[target_col]
    return train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)


def build_candidate_pipelines(scale_pos_weight):
    return {
        "Logistic Regression": build_full_pipeline(
            LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
        ),
        "Random Forest": build_full_pipeline(
            RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)
        ),
        "XGBoost": build_full_pipeline(
            XGBClassifier(n_estimators=100, random_state=RANDOM_STATE, eval_metric="logloss")
        ),
        "Logistic Regression (balanced)": build_full_pipeline(
            LogisticRegression(max_iter=1000, random_state=RANDOM_STATE, class_weight="balanced")
        ),
        "Random Forest (balanced)": build_full_pipeline(
            RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, class_weight="balanced")
        ),
        "XGBoost (balanced)": build_full_pipeline(
            XGBClassifier(
                n_estimators=100,
                random_state=RANDOM_STATE,
                eval_metric="logloss",
                scale_pos_weight=scale_pos_weight,
            )
        ),
    }


def run_cross_validation(pipelines, X_train, y_train):
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    return {
        name: cross_validate(
            pipeline, X_train, y_train, cv=cv, scoring=SCORING,
            return_train_score=True, n_jobs=-1,
        )
        for name, pipeline in pipelines.items()
    }


def log_cv_results_to_mlflow(pipelines, results, X_train, y_train):
    """Loga params e métricas por fold (média/desvio) de cada pipeline no MLflow.
    Cada pipeline é ajustado em X_train/y_train aqui mesmo, então o dict `pipelines`
    fica com os modelos já treinados ao final — reaproveitado para o modelo final.
    """
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    for name, pipeline in pipelines.items():
        result = results[name]
        with mlflow.start_run(run_name=name):
            mlflow.log_params(pipeline.named_steps["classifier"].get_params())
            for metric in SCORING:
                mlflow.log_metric(f"train_{metric}_mean", result[f"train_{metric}"].mean())
                mlflow.log_metric(f"train_{metric}_std", result[f"train_{metric}"].std())
                mlflow.log_metric(f"test_{metric}_mean", result[f"test_{metric}"].mean())
                mlflow.log_metric(f"test_{metric}_std", result[f"test_{metric}"].std())
            pipeline.fit(X_train, y_train)
            mlflow.sklearn.log_model(pipeline, "model", serialization_format="cloudpickle")


def train_and_select_final_model(df_outcomes, final_model_name=FINAL_MODEL_NAME):
    X_train, X_test, y_train, y_test = split_train_test(df_outcomes)

    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    pipelines = build_candidate_pipelines(scale_pos_weight)

    results = run_cross_validation(pipelines, X_train, y_train)
    log_cv_results_to_mlflow(pipelines, results, X_train, y_train)

    final_pipeline = pipelines[final_model_name]
    return final_pipeline, X_test, y_test


def save_final_model(pipeline, path=MODEL_PATH):
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path)
