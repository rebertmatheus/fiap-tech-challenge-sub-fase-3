"""Engenharia de features e montagem do pipeline sklearn de ponta a ponta.

A correção do bug de escala das notas e a derivação das features acadêmicas
vivem aqui como `FunctionTransformer`s dentro do `Pipeline` (não aplicadas
uma vez no DataFrame bruto como na prototipação em notebook) — assim, dado
novo e cru (uma linha vinda do formulário do app, por exemplo) pode ir direto
pro `.predict()` sem nenhum pré-processamento manual.
"""

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

    Args:
        df: DataFrame contendo `grade_cols`.
        grade_cols: Colunas de nota a corrigir. Default: `GRADE_COLS` (config).
        max_valid: Valor máximo aceito na escala correta. Default: `GRADE_MAX_VALID`.

    Returns:
        Cópia do DataFrame com `grade_cols` corrigidas (valores 0-20).
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

    Args:
        df: DataFrame com as colunas `UnidadesCurriculares{1,2}Semestre{Inscrito,
            Aprovado,Grau}` (idealmente já com `fix_grade_scale` aplicado).

    Returns:
        Cópia do DataFrame com 5 colunas novas: `no_courses_full_year`,
        `approval_rate_sem1`, `approval_rate_sem2`, `grade_delta` e
        `approval_rate_delta` (nomes em `DERIVED_NUMERIC_COLS`).
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
    """Monta o `ColumnTransformer` de pré-processamento: numéricas ganham
    imputação por mediana + `StandardScaler`; categóricas ganham imputação por
    moda + `OneHotEncoder` (ignorando categorias não vistas no treino).

    Args:
        numeric_cols: Lista de colunas numéricas (já incluindo as derivadas,
            se aplicável — ver `DERIVED_NUMERIC_COLS`).
        categorical_cols: Lista de colunas categóricas.

    Returns:
        `ColumnTransformer` não treinado, pronto para entrar num `Pipeline`.
    """
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

    Args:
        df: DataFrame bruto (sem correção de notas nem features derivadas).
        preprocessor: `ColumnTransformer` já treinado (`.fit()` chamado).

    Returns:
        Array (denso ou esparso, conforme o `ColumnTransformer`) já
        pré-processado, pronto para `.predict()`/`.transform()` de um estimador.
    """
    df = fix_grade_scale(df)
    df = add_academic_performance_features(df)
    return preprocessor.transform(df)


def build_full_pipeline(estimator, numeric_cols=None, categorical_cols=None):
    """Pipeline de ponta a ponta: corrige o bug das notas, deriva as features
    acadêmicas e pré-processa — tudo dentro do Pipeline sklearn, para que dado
    novo (bruto) baste passar por `.fit`/`.predict` sem pré-processamento manual.

    Args:
        estimator: Classificador sklearn não treinado (ex.: `LogisticRegression`).
        numeric_cols: Colunas numéricas originais (sem as derivadas — elas são
            adicionadas automaticamente). Default: `NUMERIC_COLS` (config).
        categorical_cols: Colunas categóricas. Default: `CATEGORICAL_COLS` (config).

    Returns:
        `Pipeline` sklearn não treinado com os steps `fix_grades`,
        `add_features`, `preprocessor` e `classifier`.
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
