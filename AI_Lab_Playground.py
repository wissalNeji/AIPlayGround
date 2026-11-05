"""
AI Lab Playground — Lab 1
Streamlit application for beginners: Tokenisation, Embeddings, LLM & Prompting.
No RAG in this lab. The RAG chapter is intentionally kept for a later session.

Install:
    pip install streamlit tiktoken sentence-transformers scikit-learn groq

Run:
    streamlit run AI_Lab_Playground.py

Optional LLM configuration:
    Set GROQ_API_KEY as an environment variable, or add it to .streamlit/secrets.toml:
    GROQ_API_KEY = "your_key"
    GROQ_MODEL = "llama-3.3-70b-versatile"
"""

import os
import streamlit as st
import tiktoken
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="AI Lab Playground", page_icon="🧠", layout="wide")

st.title("🧠 AI Lab Playground")
st.caption("Lab 1 — Explore how language models process text and generate answers")
st.info("Ce laboratoire couvre la tokenisation, les embeddings et la génération par LLM. "
        "La recherche documentaire et le RAG seront étudiés dans un chapitre ultérieur.")

@st.cache_resource(show_spinner="Chargement du modèle multilingue (première utilisation)...")
def load_embedding_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

def get_tokenizer(name):
    return tiktoken.get_encoding(name)

def get_groq_client():
    try:
        from groq import Groq
        key = st.secrets.get("GROQ_API_KEY", None) if hasattr(st, "secrets") else None
        key = key or os.getenv("GROQ_API_KEY")
        if not key:
            return None, "Aucune clé API configurée."
        return Groq(api_key=key), None
    except Exception as exc:
        return None, str(exc)

def call_llm(messages, model, temperature, max_tokens):
    client, error = get_groq_client()
    if client is None:
        raise RuntimeError(error + " Configurez GROQ_API_KEY pour activer cette activité.")
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_completion_tokens=max_tokens,
    )
    return response.choices[0].message.content or "", response.usage

tab1, tab2, tab3 = st.tabs([
    "① Text Explorer — Tokenisation",
    "② Meaning Explorer — Embeddings",
    "③ AI Assistant — LLM & Prompting",
])

with tab1:
    st.header("① Text Explorer")
    st.write("Un modèle de langage ne traite pas directement les mots comme un humain : "
             "le texte est découpé en unités appelées **tokens**.")
    col1, col2 = st.columns([2, 1])
    with col1:
        text = st.text_area(
            "Saisissez un texte à explorer",
            "Bonjour, je m'appelle Sami et j'étudie l'intelligence artificielle à Tunis.",
            height=110,
            key="token_text",
        )
    with col2:
        encoding_name = st.selectbox("Tokenizer à tester", ["cl100k_base", "o200k_base"])
        st.caption("Deux vocabulaires de tokenisation utilisés par différentes familles de modèles.")
    if st.button("Découper le texte", type="primary", key="tokenize"):
        try:
            enc = get_tokenizer(encoding_name)
            ids = enc.encode(text)
            pieces = [enc.decode_single_token_bytes(i).decode("utf-8", errors="replace") for i in ids]
            st.subheader("Résultat")
            m1, m2, m3 = st.columns(3)
            m1.metric("Caractères", len(text))
            m2.metric("Mots (séparés par espaces)", len(text.split()))
            m3.metric("Tokens", len(ids))
            st.write("**Tokens (segments visibles)**")
            st.code(" | ".join(repr(p) for p in pieces))
            st.write("**Identifiants des tokens**")
            st.code(str(ids))
            st.caption("Un token peut être un mot entier, une partie de mot, un espace associé à du texte ou un signe.")
        except Exception as exc:
            st.error(f"Erreur de tokenisation : {exc}")
    st.divider()
    st.subheader("Expériences guidées")
    samples = {
        "Français": "L'intelligence artificielle transforme l'éducation.",
        "English": "Artificial intelligence is transforming education.",
        "العربية": "الذكاء الاصطناعي يغير التعليم.",
        "Arabizi tunisien": "Aslema, chnouwa a7welek? Ena mrigel.",
    }
    chosen = st.selectbox("Choisir un exemple multilingue", list(samples.keys()))
    if st.button("Analyser cet exemple"):
        enc = get_tokenizer(encoding_name)
        sample = samples[chosen]
        ids = enc.encode(sample)
        pieces = [enc.decode_single_token_bytes(i).decode("utf-8", errors="replace") for i in ids]
        st.write(f"**Texte :** {sample}")
        st.write(f"**Nombre de mots :** {len(sample.split())} · **Nombre de tokens :** {len(ids)}")
        st.code(" | ".join(repr(p) for p in pieces))
    with st.expander("Défi d'application — À faire dans votre compte rendu"):
        st.markdown("""
        1. Comparez une phrase française et sa traduction anglaise.
        2. Testez une phrase en arabe puis une phrase en Arabizi tunisien.
        3. Remplacez un nom propre, ajoutez un nombre et de la ponctuation.
        4. Comparez les résultats avec `cl100k_base` et `o200k_base`.
        5. Expliquez pourquoi le nombre de tokens peut avoir un effet sur les limites et le coût d'utilisation d'un LLM.
        """)

with tab2:
    st.header("② Meaning Explorer")
    st.write("Un **embedding** représente une phrase par une liste de nombres (un vecteur). "
             "Des phrases proches par le sens peuvent avoir des vecteurs proches.")
    st.warning("Le score de similarité est un indicateur calculé par le modèle : ce n'est ni une probabilité ni une mesure parfaite de compréhension.")
    default_pairs = [
        ("Le chat dort sur le canapé.", "The cat is sleeping on the sofa."),
        ("Le film est excellent.", "Le film est génial."),
        ("Le film est excellent.", "Le film est mauvais."),
        ("Le film est excellent.", "Le film n'est pas excellent."),
    ]
    pair_choice = st.selectbox(
        "Choisissez une expérience",
        ["Traduction", "Synonymes", "Contraires", "Négation", "Saisir mes propres phrases"]
    )
    idx_map = {"Traduction": 0, "Synonymes": 1, "Contraires": 2, "Négation": 3}
    if pair_choice == "Saisir mes propres phrases":
        a = st.text_input("Phrase A", "Les étudiants apprennent avec des projets.")
        b = st.text_input("Phrase B", "Project-based learning helps students.")
    else:
        a, b = default_pairs[idx_map[pair_choice]]
        st.text_area("Phrase A", a, disabled=True)
        st.text_area("Phrase B", b, disabled=True)
    if st.button("Comparer les deux phrases", type="primary"):
        try:
            model = load_embedding_model()
            vectors = model.encode([a, b], normalize_embeddings=True)
            score = float(np.dot(vectors[0], vectors[1]))
            st.metric("Similarité cosinus", f"{score:.3f}")
            st.progress(max(0.0, min(1.0, (score + 1) / 2)))
            if score >= 0.70:
                st.success("Les phrases sont relativement proches selon ce modèle.")
            elif score >= 0.40:
                st.info("Le modèle détecte une proximité intermédiaire.")
            else:
                st.warning("Les phrases sont relativement éloignées selon ce modèle.")
            st.caption("Interprétez le score en comparant plusieurs paires, pas isolément.")
        except Exception as exc:
            st.error(f"Impossible de charger le modèle ou de calculer les embeddings : {exc}")
    st.divider()
    st.subheader("Recherche sémantique : retrouver la phrase la plus proche")
    corpus = [
        "Les étudiants peuvent consulter leur emploi du temps sur leur espace étudiant.",
        "La bibliothèque est ouverte du lundi au samedi de 8h à 18h.",
        "Une attestation de scolarité peut être demandée auprès du service concerné.",
        "Pour réinitialiser le mot de passe, utiliser la procédure de récupération du compte.",
        "Les inscriptions aux examens suivent le calendrier communiqué par l'administration.",
    ]
    query = st.text_input("Votre question ou votre phrase", "I forgot my password. How can I access my account?")
    k = st.slider("Nombre de résultats à afficher", 1, 5, 3)
    if st.button("Trouver les phrases les plus proches"):
        try:
            model = load_embedding_model()
            vecs = model.encode(corpus, normalize_embeddings=True)
            qvec = model.encode([query], normalize_embeddings=True)
            scores = cosine_similarity(qvec, vecs)[0]
            order = np.argsort(scores)[::-1][:k]
            for rank, i in enumerate(order, start=1):
                st.write(f"**{rank}. Score : {scores[i]:.3f}**")
                st.write(corpus[i])
                st.progress(max(0.0, min(1.0, float(scores[i]))))
        except Exception as exc:
            st.error(f"Erreur : {exc}")
    with st.expander("Défi d'application — À faire dans votre compte rendu"):
        st.markdown("""
        1. Comparez les paires traduction, synonymes, contraire et négation.
        2. Saisissez deux phrases de votre choix.
        3. Posez une question en anglais alors que les phrases proposées sont en français.
        4. Modifiez la formulation de la question et observez le classement.
        5. Expliquez dans quelle situation cette recherche sémantique serait utile.
        """)

with tab3:
    st.header("③ AI Assistant — LLM & Prompting")
    st.write("Vous allez configurer un assistant en modifiant son rôle, ses instructions et ses paramètres de génération.")
    client, api_error = get_groq_client()
    if client is None:
        st.warning("Le mode LLM nécessite une clé Groq configurée par l'enseignant dans les secrets de l'application.")
    role = st.selectbox("Rôle de l'assistant", [
        "Assistant pédagogique", "Conseiller d'orientation", "Assistant technique", "Assistant créatif"
    ])
    system_prompt = st.text_area(
        "Instructions système",
        "Tu es un assistant pédagogique. Explique les notions simplement, en français, avec un exemple concret. "
        "Adapte tes explications à un étudiant débutant.",
        height=120,
    )
    user_prompt = st.text_area(
        "Votre question",
        "Explique-moi ce qu'est un embedding avec un exemple.",
        height=100,
    )
    c1, c2, c3 = st.columns(3)
    with c1:
        temperature = st.slider("Température", 0.0, 1.5, 0.3, 0.1,
                                help="Une valeur faible rend généralement les réponses plus prévisibles.")
    with c2:
        max_tokens = st.slider("Longueur maximale (tokens)", 100, 1000, 400, 50)
    with c3:
        model_name = st.text_input("Modèle Groq", os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"))
    if st.button("Générer une réponse", type="primary"):
        try:
            messages = [
                {"role": "system", "content": f"Rôle : {role}. Instructions : {system_prompt}"},
                {"role": "user", "content": user_prompt},
            ]
            with st.spinner("Le modèle génère une réponse..."):
                answer, usage = call_llm(messages, model_name, temperature, max_tokens)
            st.subheader("Réponse de l'assistant")
            st.write(answer)
            st.divider()
            st.subheader("Informations sur la génération")
            x, y = st.columns(2)
            x.metric("Tokens en entrée", usage.prompt_tokens)
            y.metric("Tokens en sortie", usage.completion_tokens)
        except Exception as exc:
            st.error(f"Appel LLM impossible : {exc}")
    st.divider()
    st.subheader("Expérience comparative")
    st.markdown("""
    **Consigne de test :** « Propose un nom original pour une startup tunisienne spécialisée en IA. Réponds par le nom seul. »

    1. Gardez la même question et le même rôle.
    2. Testez une température basse (0.0 ou 0.1), puis plus élevée (1.0 ou 1.2).
    3. Comparez la créativité et la variabilité des réponses.
    4. Modifiez ensuite les instructions : imposez une réponse en trois phrases maximum avec un exemple.
    5. Observez comment les instructions influencent la réponse.
    """)
    with st.expander("Défi final — Configurer un assistant adapté à un étudiant débutant"):
        st.markdown("""
        Configurez l'assistant pour qu'il :
        - explique une notion d'IA en trois phrases maximum ;
        - utilise un vocabulaire accessible ;
        - donne un exemple concret ;
        - évite les détails techniques inutiles.

        Testez-le avec deux questions différentes. Copiez vos instructions et justifiez vos choix.
        """)

st.divider()
st.subheader("Bilan du laboratoire")
st.markdown("""
À la fin de ce TP, vous devez pouvoir :
- expliquer comment un texte est découpé en tokens et comparer deux tokenizers ;
- utiliser des embeddings pour comparer des phrases et effectuer une recherche sémantique ;
- configurer un LLM avec un prompt système et des paramètres de génération ;
- interpréter les limites d'une réponse générée.

**Le RAG n'est pas traité dans ce laboratoire.**
""")
