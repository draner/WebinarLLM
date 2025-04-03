# Application RAG avec Ollama et Streamlit

Cette application démontre un système de RAG (Retrieval-Augmented Generation) utilisant des modèles de langage locaux via Ollama, une base de données vectorielle ChromaDB, et une interface utilisateur Streamlit.

## Fonctionnalités

- 🤖 Utilisation de modèles de langage locaux via Ollama
- 📚 Base de connaissance vectorielle avec ChromaDB
- 📄 Support pour l'import de documents PDF, TXT et CSV
- 🔍 Recherche sémantique sur les documents importés
- 💬 Interface de chat interactive
- 📊 Gestion et visualisation des documents

## Prérequis

- Python 3.9 ou supérieur
- [Ollama](https://github.com/ollama/ollama) installé et en cours d'exécution
- Un modèle compatible avec Ollama (ex: mistral)

## Installation

### 1. Cloner le dépôt

```bash
git clone https://github.com/votre-username/rag-streamlit-app.git
cd rag-streamlit-app
```

### 2. Configuration de l'environnement

Installez les dépendances à l'aide de uv:

```bash
# Installation de uv si vous ne l'avez pas déjà
pip install uv

# Installation des dépendances
uv sync
```

Alternatively, vous pouvez aussi installer avec pip:

```bash
uv pip install -e .
```

### 3. Préparation d'Ollama

Assurez-vous qu'Ollama est en cours d'exécution et que vous avez téléchargé les modèles nécessaires:

```bash
# Télécharger un modèle de chat (si ce n'est pas déjà fait)
ollama pull mistral

# Télécharger un modèle d'embeddings (si ce n'est pas déjà fait)
ollama pull nomic-embed-text
```

## Utilisation

### Lancer l'application

```bash
streamlit run rag_app/app.py
```

### Utilisation de l'application

1. **Ajouter des documents à la base de connaissances**:

   - Utilisez le panneau latéral pour sélectionner un fichier à uploader
   - Cliquez sur "Traiter le document" pour vectoriser et stocker le contenu

2. **Sélectionner les modèles**:

   - Choisissez un modèle de conversation dans la liste déroulante
   - Choisissez un modèle d'embeddings dans la liste déroulante

3. **Poser des questions**:

   - Utilisez l'interface de chat pour poser des questions sur vos documents
   - Le système cherchera les informations pertinentes dans votre base de connaissances

4. **Consulter vos documents**:
   - Naviguez vers la page "Documents" pour voir les fichiers chargés
   - Vous pouvez également supprimer tous les documents si nécessaire

## Structure du projet

```
rag-streamlit-app/
├── pyproject.toml       # Configuration du projet et dépendances
├── README.md            # Ce fichier
└── rag_app/             # Code source de l'application
    └── app.py           # Application Streamlit principale
```
