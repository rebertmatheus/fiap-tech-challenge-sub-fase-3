from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from src.config import CATEGORICAL_COLS, GRADE_COLS, GRADE_MAX_VALID, NUMERIC_COLS

DERIVED_NUMERIC_COLS = [
    "no_courses_full_year",
    "approval_rate_sem1",
    "approval_rate_sem2",
    "grade_delta",
    "approval_rate_delta",
]


def fix_grade_scale(df, grade_cols=GRADE_COLS, max_valid=GRADE_MAX_VALID):
    """Corrige o bug de escala nas colunas de nota: parte da base tem os valores
    gravados em ~1e15 (e um subgrupo em ~1e3) em vez do range esperado 0-20. Divide
    repetidamente por 10 até cada valor caber no range válido.
    """
    df = df.copy()
    for col in grade_cols:
        values = df[col]
        while (values > max_valid).any():
            values = values.where(values <= max_valid, values / 10)
        df[col] = values
    return df


def add_academic_performance_features(df):
    """Quem não se inscreveu em nenhuma matéria no ano teria taxa de aprovação 0/0
    (indefinido) — a flag `no_courses_full_year` evita confundir esse caso com quem
    reprovou em tudo. Também captura a evolução do aluno entre o 1º e o 2º semestre.
    """
    df = df.copy()

    df["no_courses_full_year"] = (
        (df["UnidadesCurriculares1SemestreInscrito"] == 0)
        & (df["UnidadesCurriculares2SemestreInscrito"] == 0)
    ).astype(int)

    df["approval_rate_sem1"] = (
        df["UnidadesCurriculares1SemestreAprovado"] / df["UnidadesCurriculares1SemestreInscrito"]
    ).fillna(0)

    df["approval_rate_sem2"] = (
        df["UnidadesCurriculares2SemestreAprovado"] / df["UnidadesCurriculares2SemestreInscrito"]
    ).fillna(0)

    df["grade_delta"] = df["UnidadesCurriculares2SemestreGrau"] - df["UnidadesCurriculares1SemestreGrau"]
    df["approval_rate_delta"] = df["approval_rate_sem2"] - df["approval_rate_sem1"]

    return df


def build_preprocessor(numeric_cols, categorical_cols):
    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    return ColumnTransformer(transformers=[
        ("num", numeric_pipeline, numeric_cols),
        ("cat", categorical_pipeline, categorical_cols),
    ])


def prepare_input(df, preprocessor):
    """Aplica a correção de notas + features derivadas e transforma com um
    preprocessor já treinado — usado para escorar dado novo (df_enrolled/app)
    fora do fluxo de fit/predict do Pipeline completo.
    """
    df = fix_grade_scale(df)
    df = add_academic_performance_features(df)
    return preprocessor.transform(df)


def build_full_pipeline(estimator, numeric_cols=None, categorical_cols=None):
    """Pipeline de ponta a ponta: corrige o bug das notas, deriva as features
    acadêmicas e pré-processa — tudo dentro do Pipeline sklearn, para que dado
    novo (bruto) baste passar por `.fit`/`.predict` sem pré-processamento manual.
    """
    numeric_cols = list(numeric_cols) if numeric_cols is not None else list(NUMERIC_COLS)
    categorical_cols = list(categorical_cols) if categorical_cols is not None else list(CATEGORICAL_COLS)
    numeric_cols_full = numeric_cols + DERIVED_NUMERIC_COLS

    return Pipeline(steps=[
        ("fix_grades", FunctionTransformer(fix_grade_scale)),
        ("add_features", FunctionTransformer(add_academic_performance_features)),
        ("preprocessor", build_preprocessor(numeric_cols_full, categorical_cols)),
        ("classifier", estimator),
    ])
