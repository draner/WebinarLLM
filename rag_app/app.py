import os
import uuid

import ollama
import pandas as pd
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import CSVLoader, PyPDFLoader, TextLoader
from langchain_community.vectorstores import Chroma
from langchain_ollama import ChatOllama, OllamaEmbeddings

# Configuration et constantes
OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_EMBEDDING_MODEL = "nomic-embed-text"
DEFAULT_CHAT_MODEL = "gemma3:12b"
TEMP_DIR = "temp_docs"
DB_DIR = "chroma_db"

# Création des dossiers temporaires s'ils n'existent pas
os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(DB_DIR, exist_ok=True)

# Configuration de la page Streamlit
st.set_page_config(page_title="RAG avec Ollama", layout="wide")

# Initialisation des sessions states
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "document_list" not in st.session_state:
    st.session_state.document_list = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "selected_page" not in st.session_state:
    st.session_state.selected_page = "Chat"


# Fonction pour obtenir la liste des modèles disponibles depuis Ollama
@st.cache_data(ttl=300)
def get_available_models():
    try:
        models = ollama.list()
        model_list = models.get("models", [])
        model_names = [model["model"] for model in model_list]
        # Filtrer les modèles pour séparer les modèles de chat et d'embedding (approche simplifiée)
        chat_models = [model for model in model_names if "embed" not in model.lower()]
        embedding_models = [model for model in model_names if "embed" in model.lower()]

        # Ajouter des modèles par défaut si aucun n'est trouvé
        if not chat_models:
            chat_models = ["mistral"]  # Valeur par défaut si aucun modèle de chat n'est trouvé
        if not embedding_models:
            embedding_models = ["nomic-embed-text"]  # Valeur par défaut si aucun modèle d'embedding n'est trouvé

        return chat_models, embedding_models
    except Exception as e:
        st.error(f"Erreur lors de la récupération des modèles : {e}")
        return ["mistral"], ["nomic-embed-text"]


# Fonction pour changer de page
def change_page(page):
    st.session_state.selected_page = page


# Fonction pour traiter le fichier uploadé
def process_uploaded_file(uploaded_file):
    # Créer un nom de fichier unique
    file_extension = os.path.splitext(uploaded_file.name)[1].lower()
    unique_filename = f"{str(uuid.uuid4())}{file_extension}"
    temp_file_path = os.path.join(TEMP_DIR, unique_filename)

    # Enregistrer le fichier temporairement
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Choisir le bon loader en fonction de l'extension
    if file_extension == ".pdf":
        loader = PyPDFLoader(temp_file_path)
    elif file_extension == ".csv":
        loader = CSVLoader(temp_file_path)
    else:  # Par défaut, utiliser TextLoader pour .txt et autres
        loader = TextLoader(temp_file_path)

    # Charger et découper le document
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)

    # Ajouter le document à la liste avec ses métadonnées
    doc_info = {
        "filename": uploaded_file.name,
        "path": temp_file_path,
        "chunks": len(chunks),
        "size": uploaded_file.size,
        "id": unique_filename,
    }
    st.session_state.document_list.append(doc_info)

    return chunks


# Fonction pour initialiser/mettre à jour la base de connaissances vectorielle
def update_vectorstore(chunks=None):
    embedding_model = st.session_state.embedding_model

    # Initialiser les embeddings
    embeddings = OllamaEmbeddings(model=embedding_model, base_url=OLLAMA_BASE_URL)

    # Si le vectorstore existe déjà, ajouter les nouveaux chunks
    if st.session_state.vectorstore is not None and chunks:
        st.session_state.vectorstore.add_documents(chunks)
    # Sinon, créer une nouvelle vectorstore
    elif chunks:
        st.session_state.vectorstore = Chroma.from_documents(
            documents=chunks, embedding=embeddings, persist_directory=DB_DIR
        )
    # Si le vectorstore n'existe pas et qu'il n'y a pas de chunks, initialiser une vectorstore vide
    elif st.session_state.vectorstore is None:
        # Créer une collection vide
        st.session_state.vectorstore = Chroma(embedding_function=embeddings, persist_directory=DB_DIR)


# Fonction pour générer une réponse
def generate_response_stream(query):
    """Renvoie un générateur de tokens pour permettre le streaming de la réponse."""
    # Récupérer les documents pertinents
    if st.session_state.vectorstore is None:
        # Pas de base vectorielle : on se contente d'appeler le LLM
        llm = ChatOllama(
            model=st.session_state.chat_model,
            base_url=OLLAMA_BASE_URL,
            temperature=0.3,
            streaming=True,  # IMPORTANT pour autoriser le stream
        )
        return llm.stream(query)

    # Sinon, on récupère des chunks depuis le retriever
    retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 4})
    docs = retriever.invoke(query)

    # Stocker l’info sur les chunks utilisés (pour l’afficher ensuite)
    st.session_state.used_chunks = docs

    # Construire un prompt simple : (vous pouvez affiner l’enchaînement si besoin)
    context = "\n".join(d.page_content for d in docs)
    prompt = f"Voici le contexte :\n{context}\n\nQuestion : {query}\nRéponse :"

    # Instancier le LLM en mode streaming
    llm = ChatOllama(model=st.session_state.chat_model, base_url=OLLAMA_BASE_URL, temperature=0.3, streaming=True)
    return llm.stream(prompt)


# Interface utilisateur Streamlit
def main():
    # Sidebar
    with st.sidebar:
        st.title("Configuration RAG")

        # Navigation
        st.subheader("Navigation")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Chat", use_container_width=True):
                change_page("Chat")
        with col2:
            if st.button("Documents", use_container_width=True):
                change_page("Documents")

        st.divider()

        # Sélection des modèles
        chat_models, embedding_models = get_available_models()

        st.subheader("Modèles")
        st.session_state.chat_model = st.selectbox(
            "Modèle de conversation",
            options=chat_models,
            index=chat_models.index(DEFAULT_CHAT_MODEL) if DEFAULT_CHAT_MODEL in chat_models else 0,
        )

        st.session_state.embedding_model = st.selectbox(
            "Modèle d'embeddings",
            options=embedding_models,
            index=embedding_models.index(DEFAULT_EMBEDDING_MODEL) if DEFAULT_EMBEDDING_MODEL in embedding_models else 0,
        )

        st.divider()

        # Upload de documents
        st.subheader("Ajouter des documents")
        uploaded_file = st.file_uploader("Choisir un fichier", type=["pdf", "txt", "csv"])

        if uploaded_file is not None and st.button("Traiter le document", use_container_width=True):
            with st.spinner("Traitement du document en cours..."):
                chunks = process_uploaded_file(uploaded_file)
                update_vectorstore(chunks)
            st.success(f"Document '{uploaded_file.name}' ajouté avec succès!")

    # Page principale (dynamique selon la sélection)
    if st.session_state.selected_page == "Chat":
        render_chat_page()
    else:  # Documents
        render_documents_page()


# Page de chat
def render_chat_page():
    st.title("Chat RAG avec Ollama")
    # Bouton "Nouvelle Conversation"
    if st.button("Nouvelle Conversation", type="primary"):
        st.session_state.chat_history = []
        st.rerun()

    # Afficher l'historique des messages
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # Zone de saisie pour le chat
    if prompt := st.chat_input("Posez votre question..."):
        # Ajouter le message utilisateur à l'historique
        st.session_state.chat_history.append({"role": "user", "content": prompt})

        # Afficher le message utilisateur
        with st.chat_message("user"):
            st.write(prompt)

        # Générer et afficher la réponse
        with st.chat_message("assistant"), st.spinner("Génération de la réponse..."):
            response = st.write_stream(generate_response_stream(prompt))
        # Affichage des chunks utilisés
        if "used_chunks" in st.session_state and st.session_state.used_chunks:
            with st.expander("Chunks utilisés"):
                for idx, doc in enumerate(st.session_state.used_chunks, start=1):
                    st.write(f"**Chunk {idx} :** {doc.page_content[:300]}...")

        # Ajouter la réponse à l'historique
        st.session_state.chat_history.append({"role": "assistant", "content": response})


# Page de documents
def render_documents_page():
    st.title("Documents dans la base de connaissances")

    if not st.session_state.document_list:
        st.info("Aucun document n'a été ajouté. Utilisez le panneau de gauche pour ajouter des documents.")
    else:
        # Créer un DataFrame pour afficher les documents
        doc_data = []
        for doc in st.session_state.document_list:
            doc_data.append(
                {
                    "Nom": doc["filename"],
                    "Taille (Ko)": round(doc["size"] / 1024, 2),
                    "Chunks": doc["chunks"],
                    "ID": doc["id"],
                }
            )

        df = pd.DataFrame(doc_data)
        st.dataframe(df, use_container_width=True)

        # Option pour supprimer tous les documents
        if st.button("Supprimer tous les documents", type="secondary"):
            st.session_state.document_list = []
            st.session_state.vectorstore = None
            for file in os.listdir(TEMP_DIR):
                os.remove(os.path.join(TEMP_DIR, file))
            st.success("Tous les documents ont été supprimés!")
            st.rerun()


if __name__ == "__main__":
    main()
