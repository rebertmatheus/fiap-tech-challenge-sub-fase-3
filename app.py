import joblib
import pandas as pd
import streamlit as st

from src.config import CATEGORICAL_COLS, MODEL_PATH, NUMERIC_COLS
from src.data import load_dataset
from src.features import fix_grade_scale

st.set_page_config(
    page_title="Previsão de evasão de estudantes",
    page_icon=":material/school:",
)


@st.cache_resource
def get_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def get_defaults():
    df_outcomes, _ = load_dataset()
    df_outcomes = fix_grade_scale(df_outcomes)
    numeric_defaults = df_outcomes[NUMERIC_COLS].median()
    categorical_defaults = df_outcomes[CATEGORICAL_COLS].mode().iloc[0]
    curso_options = sorted(df_outcomes["Curso"].unique())
    return numeric_defaults, categorical_defaults, curso_options


model = get_model()
numeric_defaults, categorical_defaults, curso_options = get_defaults()

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
    curso_index = (
        curso_options.index(categorical_defaults["Curso"])
        if categorical_defaults["Curso"] in curso_options
        else 0
    )
    curso = st.selectbox("Curso", curso_options, index=curso_index)

    st.write("1º semestre")
    with st.container(horizontal=True):
        sem1_inscrito = st.number_input(
            "Matérias inscritas",
            min_value=0,
            max_value=20,
            value=int(round(numeric_defaults["UnidadesCurriculares1SemestreInscrito"])),
            key="sem1_inscrito",
        )
        sem1_aprovado = st.number_input(
            "Matérias aprovadas",
            min_value=0,
            max_value=sem1_inscrito,
            value=min(
                int(round(numeric_defaults["UnidadesCurriculares1SemestreAprovado"])),
                sem1_inscrito,
            ),
            key="sem1_aprovado",
            help="Não pode ser maior que o número de matérias inscritas.",
        )
        sem1_grau_display = st.slider(
            "Nota média (0-10)",
            0.0,
            10.0,
            float(numeric_defaults["UnidadesCurriculares1SemestreGrau"]) / 2,
            key="sem1_grau",
            help="A base treina com a escala portuguesa 0-20; convertido automaticamente.",
        )

    st.write("2º semestre")
    with st.container(horizontal=True):
        sem2_inscrito = st.number_input(
            "Matérias inscritas",
            min_value=0,
            max_value=20,
            value=int(round(numeric_defaults["UnidadesCurriculares2SemestreInscrito"])),
            key="sem2_inscrito",
        )
        sem2_aprovado = st.number_input(
            "Matérias aprovadas",
            min_value=0,
            max_value=sem2_inscrito,
            value=min(
                int(round(numeric_defaults["UnidadesCurriculares2SemestreAprovado"])),
                sem2_inscrito,
            ),
            key="sem2_aprovado",
            help="Não pode ser maior que o número de matérias inscritas.",
        )
        sem2_grau_display = st.slider(
            "Nota média (0-10)",
            0.0,
            10.0,
            float(numeric_defaults["UnidadesCurriculares2SemestreGrau"]) / 2,
            key="sem2_grau",
            help="A base treina com a escala portuguesa 0-20; convertido automaticamente.",
        )

    with st.container(horizontal=True):
        mensalidades_em_dia = st.checkbox(
            "Mensalidades em dia", value=bool(numeric_defaults["MensalidadesEmDia"])
        )
        bolsista = st.checkbox("Bolsista", value=bool(numeric_defaults["Bolsista"]))

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
    input_row = {**numeric_defaults.to_dict(), **categorical_defaults.to_dict()}
    input_row.update(
        {
            "Curso": curso,
            "UnidadesCurriculares1SemestreInscrito": sem1_inscrito,
            "UnidadesCurriculares1SemestreAprovado": sem1_aprovado,
            "UnidadesCurriculares1SemestreGrau": sem1_grau_display * 2,
            "UnidadesCurriculares2SemestreInscrito": sem2_inscrito,
            "UnidadesCurriculares2SemestreAprovado": sem2_aprovado,
            "UnidadesCurriculares2SemestreGrau": sem2_grau_display * 2,
            "MensalidadesEmDia": int(mensalidades_em_dia),
            "Bolsista": int(bolsista),
        }
    )
    X_input = pd.DataFrame([input_row])[NUMERIC_COLS + CATEGORICAL_COLS]

    prediction = model.predict(X_input)[0]
    proba_dropout = model.predict_proba(X_input)[0, 1]

    st.subheader("Resultado")
    result_cols = st.columns(2)
    with result_cols[0]:
        if prediction == 1:
            st.metric("Previsão", "Desistente")
        else:
            st.metric("Previsão", "Graduado")
    with result_cols[1]:
        st.metric("Probabilidade de evasão", f"{proba_dropout:.1%}")
    st.progress(proba_dropout)
