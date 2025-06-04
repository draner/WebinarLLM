# Application RAG avec Ollama et Streamlit

## Fonctionnalités

- 🤖 Utilisation de LLMs locaux via Ollama
- 📚 Base de connaissance vectorielle avec ChromaDB
- 📄 Support pour l'import de documents PDF, TXT et CSV
- 🔍 Recherche sémantique sur les documents importés
- 💬 Interface de chat interactive
- 📊 Gestion et visualisation des documents

## Prérequis

- Python 3.11 ou supérieur
- [Ollama](https://github.com/ollama/ollama) installé et en cours d'exécution

## Installation

### 1. Cloner le dépôt

```bash
git clone https://github.com/draner/WebinarLLM.git
cd WebinarLLM
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

### Activer l'environnement virtuel
Selon votre système d'exploitation :
```bash
# Linux / MacOS :
source .venv/bin/activate
# Windows :
.venv\Scripts\activate
```

### Lancer l'application

```bash
streamlit run rag_app/app.py
```

## Structure du projet

```sh
rag-streamlit-app/
├── pyproject.toml       # Configuration du projet et dépendances
├── README.md            # Ce fichier
└── rag_app/             # Code source de l'application
    └── app.py           # Application Streamlit principale
```
