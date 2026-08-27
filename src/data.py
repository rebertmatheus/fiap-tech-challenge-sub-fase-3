"""Carregamento e preparação inicial da base de estudantes: leitura do Excel,
separação entre quem já tem desfecho (Graduado/Desistente) e quem ainda está
cursando (Matriculado), e mapeamento do target para binário."""

import pandas as pd

from src.config import PATH_CSV, TARGET_COL, TARGET_MAP


def load_raw_data(path=PATH_CSV):
    """Lê a base bruta a partir do Excel original.

    Args:
        path: Caminho do arquivo `.xlsx`. Default: `PATH_CSV` (config).

    Returns:
        DataFrame com as 28 colunas originais, sem nenhum tratamento.
    """
    return pd.read_excel(path)


def split_enrolled_outcomes(df, target_col=TARGET_COL):
    """Separa quem ainda está cursando (sem desfecho) de quem já concluiu
    (Graduado/Desistente) — só o segundo grupo tem rótulo verdadeiro para treino/avaliação.
    """
    df_enrolled = df[df[target_col] == "Matriculado"].copy()
    df_outcomes = df[df[target_col] != "Matriculado"].copy()
    return df_outcomes, df_enrolled


def map_target(df, target_col=TARGET_COL, target_map=TARGET_MAP):
    """Converte o target categórico (`Desistente`/`Graduado`) em binário (1/0).

    Args:
        df: DataFrame contendo `target_col` (deve conter só as duas classes
            de `target_map` — chamar depois de `split_enrolled_outcomes`).
        target_col: Nome da coluna de target. Default: `TARGET_COL`.
        target_map: Mapeamento categoria → int. Default: `TARGET_MAP`
            (`Desistente`=1, `Graduado`=0).

    Returns:
        Cópia do DataFrame com `target_col` já mapeada para inteiro.
    """
    df = df.copy()
    df[target_col] = df[target_col].map(target_map)
    return df


def load_dataset(path=PATH_CSV):
    """Pipeline de carregamento completo: lê o Excel, separa matriculados dos
    concluídos e mapeia o target binário destes últimos.

    Args:
        path: Caminho do arquivo `.xlsx`. Default: `PATH_CSV` (config).

    Returns:
        Tupla `(df_outcomes, df_enrolled)`: `df_outcomes` tem o target já
        binário (0/1), pronto para split/treino; `df_enrolled` mantém o
        target original (`Matriculado`, sem rótulo verdadeiro).
    """
    df = load_raw_data(path)
    df_outcomes, df_enrolled = split_enrolled_outcomes(df)
    df_outcomes = map_target(df_outcomes)
    return df_outcomes, df_enrolled
