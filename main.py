import streamlit as st
import pandas as pd
from transformers import pipeline
import torch

# Título de la app
st.title("🧠 Análisis de Sentimientos por Docente")

# Cargar archivo
uploaded_file = st.file_uploader("📂 Sube tu archivo CSV con comentarios", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, encoding='utf-8')
    st.success("Archivo cargado exitosamente.")
    st.dataframe(df.head())

    # Selección del ID del docente
    id_opciones = df['id_docente'].dropna().unique()
    docente_id = st.selectbox("Selecciona el ID del docente", sorted(id_opciones))

    resultados = df[df['id_docente'] == docente_id]

    if resultados.empty:
        st.error(f"No se encontró el docente con ID {docente_id}")
    else:
        st.subheader("📘 Comentarios del docente seleccionado")
        st.dataframe(resultados[['id_asignatura', 'comentarios']])

        # Limpiar y filtrar
        comentarios_invalidos = ['.', '-', '', ' ']
        resultados['comentario_valido'] = ~resultados['comentarios'].astype(str).str.strip().isin(comentarios_invalidos)
        comentarios_validos = resultados[resultados['comentario_valido']].copy()

        st.info(f"🧾 Total comentarios: {len(resultados)}")
        st.info(f"✅ Comentarios válidos: {len(comentarios_validos)}")
        st.info(f"❌ Irrelevantes: {len(resultados) - len(comentarios_validos)}")

        # Preprocesamiento de texto
        comentarios_validos['comentario_limpio'] = (
            comentarios_validos['comentarios']
            .astype(str)
            .str.strip()
            .str.replace(r"[\.\-]", "", regex=True)
            .str.lower()
        )
        max_length = 510  # máx tokens para modelo
        comentarios_validos['comentario_limpio'] = comentarios_validos['comentario_limpio'].str[:max_length]

        # Modelo de análisis de sentimiento
        @st.cache_resource
        def cargar_modelo_sentimientos():
            device = 0 if torch.cuda.is_available() else -1
            return pipeline("sentiment-analysis",
                            model="nlptown/bert-base-multilingual-uncased-sentiment",
                            device=device)

        sentiment_pipeline = cargar_modelo_sentimientos()

        with st.spinner("🔍 Analizando sentimientos..."):
            predicciones = sentiment_pipeline(comentarios_validos['comentario_limpio'].tolist())

        # Mapear a etiquetas
        def mapear_sentimiento(label):
            estrellas = int(label.split()[0])
            if estrellas <= 2:
                return "🙁 NEG"
            elif estrellas == 3:
                return "😐 NEU"
            else:
                return "🙂 POS"

        comentarios_validos['sentimiento'] = [mapear_sentimiento(p['label']) for p in predicciones]

        # 📊 Resumen general
        st.subheader("📊 Resumen general")
        total_asignaturas = resultados['id_asignatura'].nunique()
        conteo = comentarios_validos['sentimiento'].value_counts()

        st.markdown(f"""
        - **ID del docente**: {docente_id}  
        - **Asignaturas impartidas**: {total_asignaturas}  
        - **Total comentarios válidos**: {len(comentarios_validos)}  
        - 🙁 Negativos: {conteo.get('🙁 NEG', 0)}  
        - 😐 Neutros: {conteo.get('😐 NEU', 0)}  
        - 🙂 Positivos: {conteo.get('🙂 POS', 0)}  
        """)

        # 📚 Comentarios por asignatura
        st.subheader("📚 Comentarios por asignatura")

        for asignatura in sorted(comentarios_validos['id_asignatura'].unique()):
            st.markdown(f"### 📌 Asignatura {asignatura}")
            subset = comentarios_validos[comentarios_validos['id_asignatura'] == asignatura]
            for sentimiento in ['🙁 NEG', '😐 NEU', '🙂 POS']:
                subset_sent = subset[subset['sentimiento'] == sentimiento]
                if not subset_sent.empty:
                    st.markdown(f"**{sentimiento} ({len(subset_sent)} comentarios):**")
                    for _, row in subset_sent.iterrows():
                        st.markdown(f"- {row['comentario_limpio']}")
