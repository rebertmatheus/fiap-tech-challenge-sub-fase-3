"""Treinamento e seleção do modelo final: monta os 6 pipelines candidatos
(Logistic Regression, Random Forest e XGBoost, cada um com/sem
`class_weight`/`scale_pos_weight` balanceado), roda `StratifiedKFold` +
`cross_validate`, loga tudo no MLflow e seleciona o modelo final
(`FINAL_MODEL_NAME`) com base na análise de overfitting (ver checklist)."""

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
    """Separa treino/teste de forma estratificada.

    Args:
        df_outcomes: DataFrame com target binário já mapeado (`Target` = 0/1).
        target_col: Nome da coluna de target. Default: `TARGET_COL`.

    Returns:
        Tupla `(X_train, X_test, y_train, y_test)` — split 80/20
        (`TEST_SIZE`) estratificado por classe, `random_state` fixo.
    """
    X = df_outcomes.drop(columns=[target_col])
    y = df_outcomes[target_col]
    return train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)


def build_candidate_pipelines(scale_pos_weight):
    """Monta os 6 pipelines candidatos (3 algoritmos × com/sem balanceamento).

    Args:
        scale_pos_weight: Razão `(negativos / positivos)` em `y_train`, usada
            só pelo XGBoost balanceado para compensar o desbalanceamento leve
            (~61/39%) da classe `Desistente`.

    Returns:
        Dict `{nome_do_modelo: Pipeline não treinado}`.
    """
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
    """Roda `StratifiedKFold` + `cross_validate` (com score de treino) para
    cada pipeline candidato — o gap treino×validação por fold é a evidência
    usada na análise de overfitting/underfitting (ver checklist, item 6).

    Args:
        pipelines: Dict `{nome: Pipeline}`, como retornado por
            `build_candidate_pipelines`.
        X_train: Features de treino.
        y_train: Target de treino.

    Returns:
        Dict `{nome: resultado de cross_validate}` — cada valor tem as chaves
        `train_<metric>`/`test_<metric>` por fold, para cada métrica em `SCORING`.
    """
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
    """Orquestra o treino completo: split, CV dos 6 candidatos, log no MLflow
    e seleção do modelo final.

    Args:
        df_outcomes: DataFrame com target binário já mapeado, como retornado
            por `src.data.load_dataset`.
        final_model_name: Chave do modelo vencedor em `build_candidate_pipelines`.
            Default: `FINAL_MODEL_NAME` ("Logistic Regression (balanced)").

    Returns:
        Tupla `(final_pipeline, X_test, y_test)` — `final_pipeline` já vem
        treinado em `X_train`/`y_train` (fit acontece dentro de
        `log_cv_results_to_mlflow`); `X_test`/`y_test` ficam para a avaliação
        de holdout em `src.evaluate`.
    """
    X_train, X_test, y_train, y_test = split_train_test(df_outcomes)

    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    pipelines = build_candidate_pipelines(scale_pos_weight)

    results = run_cross_validation(pipelines, X_train, y_train)
    log_cv_results_to_mlflow(pipelines, results, X_train, y_train)

    final_pipeline = pipelines[final_model_name]
    return final_pipeline, X_test, y_test


def save_final_model(pipeline, path=MODEL_PATH):
    """Serializa o pipeline final treinado via `joblib`.

    Args:
        pipeline: Pipeline sklearn já treinado (`.fit()` chamado).
        path: Caminho de destino do `.joblib`. Default: `MODEL_PATH` (config).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path)
