import os
import uuid

import ollama
import pandas as pd
import streamlit as st
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.chat_models import ChatOllama
from langchain_community.document_loaders import CSVLoader, PyPDFLoader, TextLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

# Configuration et constantes
OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_EMBEDDING_MODEL = "nomic-embed-text"
DEFAULT_CHAT_MODEL = "mistral"
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
@st.cache_data(ttl=300)  # Cache pour 5 minutes
def get_available_models():
    try:
        models = ollama.list()
        model_names = [model["name"] for model in models.get("models", [])]
        # Ajouter les modèles d'embedding à la liste
        embedding_models = ["nomic-embed-text", "all-minilm"]
        return model_names, embedding_models
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

    # Si la vectorstore existe déjà, ajouter les nouveaux chunks
    if st.session_state.vectorstore is not None and chunks:
        st.session_state.vectorstore.add_documents(chunks)
    # Sinon, créer une nouvelle vectorstore
    elif chunks:
        st.session_state.vectorstore = Chroma.from_documents(
            documents=chunks, embedding=embeddings, persist_directory=DB_DIR
        )
    # Si la vectorstore n'existe pas et qu'il n'y a pas de chunks, initialiser une vectorstore vide
    elif st.session_state.vectorstore is None:
        # Créer une collection vide
        st.session_state.vectorstore = Chroma(embedding_function=embeddings, persist_directory=DB_DIR)


# Fonction pour générer une réponse
def generate_response(query):
    if st.session_state.vectorstore is None:
        return "Veuillez d'abord ajouter des documents à la base de connaissances."

    chat_model = st.session_state.chat_model

    # Initialiser le modèle de chat
    llm = ChatOllama(model=chat_model, base_url=OLLAMA_BASE_URL, temperature=0.3)

    # Initialiser la mémoire
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    # Initialiser la chaîne de conversation avec récupération
    retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 4})

    chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever, memory=memory, verbose=True)

    # Générer la réponse
    response = chain.invoke({"question": query})
    return response["answer"]


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

        if uploaded_file is not None:
            if st.button("Traiter le document", use_container_width=True):
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
        with st.chat_message("assistant"):
            with st.spinner("Génération de la réponse..."):
                response = generate_response(prompt)
                st.write(response)

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
