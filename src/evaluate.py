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
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    report = classification_report(y_test, y_pred, target_names=["Graduado", "Desistente"])
    metrics = {
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }
    return y_pred, y_proba, metrics, report


def save_evaluation_plots(pipeline, X_test, y_test, y_pred, img_dir=IMG_DIR):
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
    """
    X_enrolled = df_enrolled.drop(columns=[target_col])
    df_risk = add_academic_performance_features(fix_grade_scale(X_enrolled))
    df_risk["risk_score"] = pipeline.predict_proba(X_enrolled)[:, 1]
    return df_risk.sort_values(by="risk_score", ascending=False)


def save_top_risk_students(df_risk, n=20, path=TOP_RISK_PATH):
    df_risk.head(n).to_csv(path, index=False)
