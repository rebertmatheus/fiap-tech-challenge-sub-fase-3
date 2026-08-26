import pandas as pd

from src.config import PATH_CSV, TARGET_COL, TARGET_MAP


def load_raw_data(path=PATH_CSV):
    return pd.read_excel(path)


def split_enrolled_outcomes(df, target_col=TARGET_COL):
    """Separa quem ainda está cursando (sem desfecho) de quem já concluiu
    (Graduado/Desistente) — só o segundo grupo tem rótulo verdadeiro para treino/avaliação.
    """
    df_enrolled = df[df[target_col] == "Matriculado"].copy()
    df_outcomes = df[df[target_col] != "Matriculado"].copy()
    return df_outcomes, df_enrolled


def map_target(df, target_col=TARGET_COL, target_map=TARGET_MAP):
    df = df.copy()
    df[target_col] = df[target_col].map(target_map)
    return df


def load_dataset(path=PATH_CSV):
    df = load_raw_data(path)
    df_outcomes, df_enrolled = split_enrolled_outcomes(df)
    df_outcomes = map_target(df_outcomes)
    return df_outcomes, df_enrolled
