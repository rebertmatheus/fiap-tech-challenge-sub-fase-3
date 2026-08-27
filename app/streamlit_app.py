"""Frontend Streamlit — só UI. Não importa `joblib`/`pandas`/`src`: todo o
pipeline de ML e o dataset ficam no backend (`backend/main.py`); este script
só coleta os campos do formulário e fala HTTP com ele (`GET /defaults`,
`POST /predict`), autenticado via header `X-API-Key`. A conversão de nota
0-10 (exibida) ↔ 0-20 (escala nativa do modelo, escala portuguesa) é
responsabilidade só de apresentação e vive inteira aqui."""

import os
import time

import requests
import streamlit as st

BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000")
API_KEY = os.environ.get("API_KEY", "")
API_HEADERS = {"X-API-Key": API_KEY}

st.set_page_config(
    page_title="Previsão de evasão de estudantes",
    page_icon=":material/school:",
)


@st.cache_data(ttl="10m")
def get_defaults(retries=5, delay_seconds=2):
    """Busca os valores default do formulário em `GET /defaults` do backend.

    Args:
        retries: Tentativas antes de desistir. Default: 5.
        delay_seconds: Espera entre tentativas. Default: 2s.

    Returns:
        Dict com `curso_options`/`curso_default` e o default de cada campo
        editável (ver `backend.main.get_defaults`).

    Raises:
        requests.RequestException: Se todas as tentativas falharem — deixado
        propagar de propósito para o `try/except` no nível do módulo lidar
        com a mensagem de erro exibida ao usuário.
    """
    # No docker-compose, "depends_on" só espera o container do backend subir,
    # não a API ficar pronta — o retry cobre essa corrida no cold start.
    last_exc = None
    for attempt in range(retries):
        try:
            response = requests.get(
                f"{BACKEND_URL}/defaults", headers=API_HEADERS, timeout=10
            )
            response.raise_for_status()
            return response.json()
        except requests.RequestException as exc:
            last_exc = exc
            if attempt < retries - 1:
                time.sleep(delay_seconds)
    raise last_exc


try:
    defaults = get_defaults()
except requests.RequestException as exc:
    st.error(
        f"Não foi possível conectar ao serviço de previsão em {BACKEND_URL}: {exc}",
        icon=":material/error:",
    )
    st.stop()

st.title("Previsão de evasão de estudantes")
st.caption(
    "Preenche os campos mais preditivos do histórico acadêmico. Os demais "
    "campos são completados com a mediana/moda da base de treino."
)

with st.expander("Sobre o modelo", icon=":material/info:"):
    st.write(
        "Regressão logística com classes balanceadas, escolhida por ter o menor "
        "gap treino×validação entre os modelos testados (Logistic Regression, "
        "Random Forest, XGBoost)."
    )
    metric_cols = st.columns(2)
    metric_cols[0].metric("F1 (teste)", "0.905")
    metric_cols[1].metric("ROC-AUC (teste)", "0.976")
    st.caption(
        "A base original usa a escala portuguesa de notas (0-20). Aqui exibimos "
        "em 0-10 para facilitar a leitura; a conversão é feita automaticamente."
    )

with st.container(border=True):
    curso_options = defaults["curso_options"]
    curso_index = (
        curso_options.index(defaults["curso_default"])
        if defaults["curso_default"] in curso_options
        else 0
    )
    curso = st.selectbox("Curso", curso_options, index=curso_index)

    st.write("1º semestre")
    with st.container(horizontal=True):
        sem1_inscrito = st.number_input(
            "Matérias inscritas",
            min_value=0,
            max_value=20,
            value=defaults["sem1_inscrito"],
            key="sem1_inscrito",
        )
        sem1_aprovado = st.number_input(
            "Matérias aprovadas",
            min_value=0,
            max_value=sem1_inscrito,
            value=min(defaults["sem1_aprovado"], sem1_inscrito),
            key="sem1_aprovado",
            help="Não pode ser maior que o número de matérias inscritas.",
        )
        sem1_grau_display = st.slider(
            "Nota média (0-10)",
            0.0,
            10.0,
            defaults["sem1_grau_0_20"] / 2,
            key="sem1_grau",
            help="A base treina com a escala portuguesa 0-20; convertido automaticamente.",
        )

    st.write("2º semestre")
    with st.container(horizontal=True):
        sem2_inscrito = st.number_input(
            "Matérias inscritas",
            min_value=0,
            max_value=20,
            value=defaults["sem2_inscrito"],
            key="sem2_inscrito",
        )
        sem2_aprovado = st.number_input(
            "Matérias aprovadas",
            min_value=0,
            max_value=sem2_inscrito,
            value=min(defaults["sem2_aprovado"], sem2_inscrito),
            key="sem2_aprovado",
            help="Não pode ser maior que o número de matérias inscritas.",
        )
        sem2_grau_display = st.slider(
            "Nota média (0-10)",
            0.0,
            10.0,
            defaults["sem2_grau_0_20"] / 2,
            key="sem2_grau",
            help="A base treina com a escala portuguesa 0-20; convertido automaticamente.",
        )

    with st.container(horizontal=True):
        mensalidades_em_dia = st.checkbox(
            "Mensalidades em dia", value=defaults["mensalidades_em_dia"]
        )
        bolsista = st.checkbox("Bolsista", value=defaults["bolsista"])

    submitted = st.button("Prever", icon=":material/query_stats:")

if submitted and sem1_aprovado > sem1_inscrito:
    st.error(
        "O número de matérias aprovadas no 1º semestre não pode ser maior "
        "que o de matérias inscritas.",
        icon=":material/error:",
    )
elif submitted and sem2_aprovado > sem2_inscrito:
    st.error(
        "O número de matérias aprovadas no 2º semestre não pode ser maior "
        "que o de matérias inscritas.",
        icon=":material/error:",
    )
elif submitted:
    payload = {
        "curso": curso,
        "sem1_inscrito": sem1_inscrito,
        "sem1_aprovado": sem1_aprovado,
        "sem1_grau_0_20": sem1_grau_display * 2,
        "sem2_inscrito": sem2_inscrito,
        "sem2_aprovado": sem2_aprovado,
        "sem2_grau_0_20": sem2_grau_display * 2,
        "mensalidades_em_dia": mensalidades_em_dia,
        "bolsista": bolsista,
    }
    try:
        response = requests.post(
            f"{BACKEND_URL}/predict", json=payload, headers=API_HEADERS, timeout=10
        )
        response.raise_for_status()
        result = response.json()
    except requests.RequestException as exc:
        st.error(
            f"Não foi possível falar com o serviço de previsão: {exc}",
            icon=":material/error:",
        )
    else:
        st.subheader("Resultado")
        result_cols = st.columns(2)
        with result_cols[0]:
            st.metric("Previsão", result["prediction"])
        with result_cols[1]:
            st.metric("Probabilidade de evasão", f"{result['proba_dropout']:.1%}")
        st.progress(result["proba_dropout"])
