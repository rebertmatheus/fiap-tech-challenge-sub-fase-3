"""Avaliação do modelo final no holdout, geração das figuras (matriz de
confusão, curva ROC), registro no MLflow Model Registry, e scoring dos alunos
`Matriculado` (sem rótulo verdadeiro, fora do fluxo de treino/teste)."""

import mlflow
import mlflow.sklearn
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    classification_report,
    f1_score,
    roc_auc_score,
)

from src.config import IMG_DIR, REGISTERED_MODEL_NAME, ROOT_DIR, TARGET_COL
from src.features import add_academic_performance_features, fix_grade_scale

TOP_RISK_PATH = ROOT_DIR / "data" / "top_risk_students.csv"


def evaluate_on_holdout(pipeline, X_test, y_test):
    """Avalia o pipeline final no conjunto de teste (holdout, nunca visto no CV).

    Args:
        pipeline: Pipeline final já treinado.
        X_test: Features de teste.
        y_test: Target de teste.

    Returns:
        Tupla `(y_pred, y_proba, metrics, report)`: `y_pred`/`y_proba` são as
        predições no holdout; `metrics` é um dict com `f1` e `roc_auc`;
        `report` é o `classification_report` formatado (precision/recall/f1
        por classe).
    """
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    report = classification_report(y_test, y_pred, target_names=["Graduado", "Desistente"])
    metrics = {
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }
    return y_pred, y_proba, metrics, report


def save_evaluation_plots(pipeline, X_test, y_test, y_pred, img_dir=IMG_DIR):
    """Gera e salva a matriz de confusão e a curva ROC do holdout em PNG.

    Args:
        pipeline: Pipeline final já treinado (usado pela curva ROC, que
            precisa de `predict_proba`).
        X_test: Features de teste.
        y_test: Target de teste.
        y_pred: Predições no holdout (de `evaluate_on_holdout`).
        img_dir: Diretório de saída. Default: `IMG_DIR` (config, `docs/img`).

    Returns:
        Tupla `(cm_path, roc_path)` com os caminhos dos PNGs salvos.
    """
    img_dir.mkdir(parents=True, exist_ok=True)

    cm_path = img_dir / "confusion_matrix_logreg_balanced.png"
    roc_path = img_dir / "roc_curve_logreg_balanced.png"

    cm_display = ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred, display_labels=["Graduado", "Desistente"]
    )
    roc_display = RocCurveDisplay.from_estimator(pipeline, X_test, y_test)

    cm_display.figure_.savefig(cm_path, dpi=150, bbox_inches="tight")
    roc_display.figure_.savefig(roc_path, dpi=150, bbox_inches="tight")

    return cm_path, roc_path


def register_final_model(
    pipeline, metrics, cm_path, roc_path,
    run_name="Logistic Regression (balanced) - Final Model",
):
    """Loga o run final no MLflow (params, métricas de holdout, figuras) e
    registra o pipeline no Model Registry (`REGISTERED_MODEL_NAME`).

    Args:
        pipeline: Pipeline final já treinado.
        metrics: Dict com `f1`/`roc_auc` de holdout (de `evaluate_on_holdout`).
        cm_path: Caminho do PNG da matriz de confusão.
        roc_path: Caminho do PNG da curva ROC.
        run_name: Nome do run no MLflow.
    """
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(pipeline.named_steps["classifier"].get_params())
        mlflow.log_metric("holdout_f1", metrics["f1"])
        mlflow.log_metric("holdout_roc_auc", metrics["roc_auc"])
        mlflow.log_artifact(str(cm_path))
        mlflow.log_artifact(str(roc_path))
        mlflow.sklearn.log_model(
            pipeline, "model", serialization_format="cloudpickle",
            registered_model_name=REGISTERED_MODEL_NAME,
        )


def score_enrolled_students(pipeline, df_enrolled, target_col=TARGET_COL):
    """Escora quem ainda está cursando (sem rótulo verdadeiro, não usado em
    fit/CV/teste). O pipeline já corrige notas e deriva features internamente
    para calcular o `risk_score`; aqui replicamos a mesma correção + derivação
    antes de montar o DataFrame de saída, senão as colunas de nota/`grade_delta`
    saem com o bug de escala (~1/3 dos matriculados tem UnidadesCurriculares*Grau
    gravado em ~1e15 em vez de 0-20).

    Args:
        pipeline: Pipeline final já treinado.
        df_enrolled: DataFrame bruto dos alunos `Matriculado` (de
            `src.data.load_dataset`), ainda com `target_col`.
        target_col: Nome da coluna de target a descartar. Default: `TARGET_COL`.

    Returns:
        DataFrame com as features derivadas + coluna `risk_score`
        (probabilidade de evasão), ordenado do maior pro menor risco.
    """
    X_enrolled = df_enrolled.drop(columns=[target_col])
    df_risk = add_academic_performance_features(fix_grade_scale(X_enrolled))
    df_risk["risk_score"] = pipeline.predict_proba(X_enrolled)[:, 1]
    return df_risk.sort_values(by="risk_score", ascending=False)


def save_top_risk_students(df_risk, n=20, path=TOP_RISK_PATH):
    """Salva os `n` alunos de maior risco (topo do `df_risk` já ordenado) em CSV.

    Args:
        df_risk: DataFrame ordenado por `risk_score` desc, de
            `score_enrolled_students`.
        n: Quantidade de linhas a salvar. Default: 20.
        path: Caminho do CSV de saída. Default: `TOP_RISK_PATH`.
    """
    df_risk.head(n).to_csv(path, index=False)
