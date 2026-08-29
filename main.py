"""Entrypoint do pipeline de treino completo: carrega os dados, treina e
seleciona o modelo final via CV, avalia no holdout, registra no MLflow e
escora os alunos `Matriculado`. Roda com `uv run python main.py` (ou
`python main.py`). Não confundir com `app/streamlit_app.py` (frontend) nem
`backend/main.py` (API de serving) — este script é só para (re)treinar."""

from src.data import load_dataset
from src.evaluate import (
    evaluate_on_holdout,
    register_final_model,
    save_evaluation_plots,
    save_top_risk_students,
    score_enrolled_students,
)
from src.train import save_final_model, train_and_select_final_model


def main():
    """Roda o pipeline de treino de ponta a ponta e imprime as métricas de
    holdout no console. Efeitos colaterais: sobrescreve `models/*.joblib`,
    `docs/img/*.png` e `data/top_risk_students.csv`; cria runs no MLflow."""
    df_outcomes, df_enrolled = load_dataset()

    final_pipeline, X_test, y_test = train_and_select_final_model(df_outcomes)
    save_final_model(final_pipeline)

    y_pred, _, metrics, report = evaluate_on_holdout(final_pipeline, X_test, y_test)
    print(f"Test F1: {metrics['f1']:.3f}")
    print(f"Test ROC-AUC: {metrics['roc_auc']:.3f}")
    print(report)

    cm_path, roc_path = save_evaluation_plots(final_pipeline, X_test, y_test, y_pred)
    register_final_model(final_pipeline, metrics, cm_path, roc_path)

    df_risk = score_enrolled_students(final_pipeline, df_enrolled)
    save_top_risk_students(df_risk)


if __name__ == "__main__":
    main()
