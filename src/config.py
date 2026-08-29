"""Constantes compartilhadas pelo pipeline de ML: paths, colunas, hiperparâmetros
de reprodutibilidade e nomes usados no MLflow. Fonte única de verdade consumida
por `src/data.py`, `src/features.py`, `src/train.py`, `src/evaluate.py`,
`main.py` (treino) e `backend/main.py` (serving)."""

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
RANDOM_STATE = 42
TEST_SIZE = 0.2
N_SPLITS = 5

TARGET_COL = "Target"
TARGET_MAP = {"Desistente": 1, "Graduado": 0}

GRADE_COLS = [
    "UnidadesCurriculares1SemestreGrau",
    "UnidadesCurriculares2SemestreGrau"
]

GRADE_MAX_VALID = 20

CATEGORICAL_COLS = [
    "EstadoCivil",
    "Curso",
    "QualificacaoAnterior",
    "Nacionalidade",
    "Genero"
]

NUMERIC_COLS = [
    "QualificacaoAnteriorGrau",
    "NotaAdmissao",
    "NecessidadesEspeciais",
    "Devedor",
    "MensalidadesEmDia",
    "Bolsista",
    "International",
    "UnidadesCurriculares1SemestreCreditado",
    "UnidadesCurriculares1SemestreInscrito",
    "UnidadesCurriculares1SemestreAvaliacoes",
    "UnidadesCurriculares1SemestreAprovado",
    "UnidadesCurriculares1SemestreGrau",
    "UnidadesCurriculares1SemestreSemAvaliacoes",
    "UnidadesCurriculares2SemestreCreditado",
    "UnidadesCurriculares2SemestreInscrito",
    "UnidadesCurriculares2SemestreAvaliacoes",
    "UnidadesCurriculares2SemestreAprovado",
    "UnidadesCurriculares2SemestreGrau",
    "UnidadesCurriculares2SemestreSemAvaliacoes",
    "TaxaDesemprego",
    "TaxaInflacao",
    "PIB"
]

PATH_CSV = ROOT_DIR / "data" / "StudentsPrepared.xlsx"
MODEL_PATH = ROOT_DIR / "models" / "logistic_regression_balanced.joblib"
IMG_DIR = ROOT_DIR / "docs" / "img"

MLFLOW_EXPERIMENT_NAME = "evasao-estudantil-cv"
REGISTERED_MODEL_NAME = "evasao-estudantil-logreg-balanced"
