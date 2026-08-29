"""API de serving do modelo de evasão (FastAPI). Dono do pipeline treinado e
do dataset — calcula os defaults do formulário e roda as previsões. Consumido
só pelo frontend (`app/streamlit_app.py`), nunca diretamente pelo usuário
final; por isso não precisa (e não deveria) ficar exposto publicamente — ver
`docker-compose.yml` (`expose`, sem `ports`, no serviço `backend`).

Endpoints exigem o header `X-API-Key` (ver `verify_api_key`), exceto `/health`.
"""

import os

import joblib
import pandas as pd
from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel

from src.config import CATEGORICAL_COLS, MODEL_PATH, NUMERIC_COLS
from src.data import load_dataset
from src.features import fix_grade_scale

API_KEY = os.environ.get("API_KEY")

app = FastAPI(title="Previsão de evasão de estudantes - API")

model = joblib.load(MODEL_PATH)

_df_outcomes, _ = load_dataset()
_df_fixed = fix_grade_scale(_df_outcomes)
_numeric_defaults = _df_fixed[NUMERIC_COLS].median()
_categorical_defaults = _df_fixed[CATEGORICAL_COLS].mode().iloc[0]
_curso_options = sorted(_df_fixed["Curso"].unique())


def verify_api_key(x_api_key: str | None = Header(default=None)):
    """Dependência do FastAPI que autentica requisições via header `X-API-Key`.

    Fail-closed: se `API_KEY` não estiver configurada no backend (env var
    ausente), rejeita com 500 em vez de deixar passar sem checagem.

    Raises:
        HTTPException: 500 se `API_KEY` não estiver configurada no ambiente
            do backend; 401 se o header enviado não bater com `API_KEY`.
    """
    if not API_KEY:
        raise HTTPException(500, "Backend não configurado corretamente (API_KEY ausente).")
    if x_api_key != API_KEY:
        raise HTTPException(401, "Chave de API inválida ou ausente.")


class StudentInput(BaseModel):
    """Payload de `POST /predict` — só os campos que o formulário expõe.
    Os demais ~20 campos do modelo são preenchidos com mediana/moda do
    treino, aqui dentro do backend (o frontend nunca vê essas colunas)."""

    curso: str
    sem1_inscrito: int
    sem1_aprovado: int
    sem1_grau_0_20: float
    sem2_inscrito: int
    sem2_aprovado: int
    sem2_grau_0_20: float
    mensalidades_em_dia: bool
    bolsista: bool


@app.get("/health")
def health():
    """Healthcheck público (sem autenticação) para orquestração (Docker/EC2)."""
    return {"status": "ok"}


@app.get("/defaults", dependencies=[Depends(verify_api_key)])
def get_defaults():
    """Valores iniciais para os campos que o formulário do frontend expõe.

    Calculados uma vez, no import do módulo, a partir da mediana (numéricos)
    e moda (categóricos) de `df_outcomes` — com `fix_grade_scale` aplicado
    antes, senão o bug de escala das notas contaminaria a mediana.

    Returns:
        Dict com `curso_options`/`curso_default` e o valor default de cada
        um dos 8 campos editáveis (ver `StudentInput`).
    """
    curso_default = _categorical_defaults["Curso"]
    if curso_default not in _curso_options:
        curso_default = _curso_options[0]

    return {
        "curso_options": _curso_options,
        "curso_default": curso_default,
        "sem1_inscrito": int(round(_numeric_defaults["UnidadesCurriculares1SemestreInscrito"])),
        "sem1_aprovado": int(round(_numeric_defaults["UnidadesCurriculares1SemestreAprovado"])),
        "sem1_grau_0_20": float(_numeric_defaults["UnidadesCurriculares1SemestreGrau"]),
        "sem2_inscrito": int(round(_numeric_defaults["UnidadesCurriculares2SemestreInscrito"])),
        "sem2_aprovado": int(round(_numeric_defaults["UnidadesCurriculares2SemestreAprovado"])),
        "sem2_grau_0_20": float(_numeric_defaults["UnidadesCurriculares2SemestreGrau"]),
        "mensalidades_em_dia": bool(_numeric_defaults["MensalidadesEmDia"]),
        "bolsista": bool(_numeric_defaults["Bolsista"]),
    }


@app.post("/predict", dependencies=[Depends(verify_api_key)])
def predict(student: StudentInput):
    """Preenche os campos não editáveis com mediana/moda, roda o pipeline
    treinado e devolve a previsão de evasão.

    Args:
        student: Os 8 campos editáveis do formulário (ver `StudentInput`).
            Notas em escala 0-20 (conversão 0-10↔0-20 é responsabilidade do
            frontend, é só apresentação).

    Returns:
        Dict com `prediction` ("Desistente"/"Graduado") e `proba_dropout`
        (probabilidade da classe Desistente, 0-1).

    Raises:
        HTTPException: 422 se `aprovado` > `inscrito` em algum semestre
        (o frontend já bloqueia isso na UI; aqui é a segunda camada de defesa,
        já que a API pode ser chamada por qualquer cliente autenticado).
    """
    if student.sem1_aprovado > student.sem1_inscrito:
        raise HTTPException(
            422, "Matérias aprovadas no 1º semestre não pode ser maior que inscritas."
        )
    if student.sem2_aprovado > student.sem2_inscrito:
        raise HTTPException(
            422, "Matérias aprovadas no 2º semestre não pode ser maior que inscritas."
        )

    input_row = {**_numeric_defaults.to_dict(), **_categorical_defaults.to_dict()}
    input_row.update(
        {
            "Curso": student.curso,
            "UnidadesCurriculares1SemestreInscrito": student.sem1_inscrito,
            "UnidadesCurriculares1SemestreAprovado": student.sem1_aprovado,
            "UnidadesCurriculares1SemestreGrau": student.sem1_grau_0_20,
            "UnidadesCurriculares2SemestreInscrito": student.sem2_inscrito,
            "UnidadesCurriculares2SemestreAprovado": student.sem2_aprovado,
            "UnidadesCurriculares2SemestreGrau": student.sem2_grau_0_20,
            "MensalidadesEmDia": int(student.mensalidades_em_dia),
            "Bolsista": int(student.bolsista),
        }
    )
    X_input = pd.DataFrame([input_row])[NUMERIC_COLS + CATEGORICAL_COLS]

    prediction = model.predict(X_input)[0]
    proba_dropout = float(model.predict_proba(X_input)[0, 1])

    return {
        "prediction": "Desistente" if prediction == 1 else "Graduado",
        "proba_dropout": proba_dropout,
    }
