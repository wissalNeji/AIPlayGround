# 🧠 AI Lab Playground — laboratoire complet en six étapes

Application Streamlit pour des étudiants débutants en intelligence artificielle. Les six étapes sont progressives et chacune porte des objectifs, des consignes, des expériences, des défis et des zones de réponse :

| Étape | Contenu | Source pédagogique | Services nécessaires |
|---|---|---|---|
| ① Text Explorer | Tokenisation, comparaison `cl100k_base` / `o200k_base` | Application d'origine (conservée) | Aucun |
| ② Meaning Explorer | Embeddings, similarité cosinus, recherche sémantique | Application d'origine (conservée) | Aucun (modèle téléchargé au 1er usage) |
| ③ AI Ethics Lab | Atelier « Du problème au prototype web » : 9 parties + synthèse | Support d'atelier (Chapitre 3) | Clé Groq **ou** LLM externe (copier-coller) ; Lovable ; HF facultatif |
| ④ AI Assistant | Rôle, prompt système, température, longueur, modèle, tokens | Ancienne étape ③ (déplacée) | Clé Groq |
| ⑤ Generative AI Lab | Sections 0 à 8 du notebook : coût, sampling, prompt système, BLEU/chrF, biais d'images, TTS/STT/WER, chaîne multimodale, empreinte, injection | Notebook `Lab_GenAI_2A.ipynb` | Clé Groq ; jeton HF (images) ; modèles locaux optionnels |
| ⑥ RAG Lab | Import TXT/MD/PDF, chunks, embeddings, Top-K, contexte, réponse sourcée, avec/sans RAG, limites | Nouveau chapitre | Aucun pour l'indexation ; clé Groq pour la génération |

---

## 1. Installation locale

```bash
python -m venv .venv
source .venv/bin/activate            # Windows : .venv\Scripts\activate
pip install -r requirements.txt
streamlit run AI_Lab_Playground.py
```

Python 3.11 à 3.13 : application testée avec Python 3.13 et Streamlit 1.67, résolution des dépendances vérifiée pour Python 3.12.

## 2. Configuration des clés (ne jamais les écrire dans le code)

Créez `.streamlit/secrets.toml` (à ne **pas** versionner — ajoutez-le à `.gitignore`) :

```toml
GROQ_API_KEY = "gsk_..."     # obligatoire pour les activités LLM, vision, audio
HF_TOKEN = "hf_..."          # facultatif : génération d'images (étapes ③ 6.1 et ⑤ sections 5 et 7)

# Facultatif : remplacer les modèles par défaut
# GROQ_MODEL = "openai/gpt-oss-120b"
# GROQ_VISION_MODEL = "qwen/qwen3.8-27b"
# GROQ_STT_MODEL = "whisper-large-v3-turbo"
# GROQ_TTS_MODEL = "canopylabs/orpheus-v1-english"
# HF_IMAGE_MODEL = "black-forest-labs/FLUX.1-schnell"
```

Les mêmes noms fonctionnent comme **variables d'environnement**. Un étudiant peut aussi saisir **sa propre clé** dans la barre latérale (*Utiliser ma propre clé*). Elle reste dans sa session et n'est jamais incluse dans les sauvegardes. Cela évite d'épuiser le quota gratuit de la clé de l'enseignant quand toute une classe travaille en même temps.

- Clé Groq : https://console.groq.com (offre gratuite avec limites de débit).
- Jeton Hugging Face : https://huggingface.co/settings/tokens (droit « Inference »). L'offre gratuite des Inference Providers inclut des crédits limités.

## 3. Déploiement sur Streamlit Community Cloud

1. Placez `AI_Lab_Playground.py`, `requirements.txt` et `README.md` dans un dépôt GitHub. Ne versionnez **aucune** clé.
2. Sur https://share.streamlit.io → *Create app*, choisissez le dépôt et le fichier principal `AI_Lab_Playground.py`.
3. *Advanced settings* : choisissez Python 3.12 et collez le contenu de votre `secrets.toml` dans *Secrets*.
4. Déployez. La première installation est longue, car `sentence-transformers` installe `torch`.

**Mémoire.** Les modèles locaux sont chargés **uniquement à la demande** : le modèle d'embedding (≈ 470 Mo) aux étapes ② et ⑥, et les modèles de l'étape ⑤ si l'étudiant choisit le moteur « Local ». Un seul modèle génératif local est gardé en mémoire à la fois. Sur Streamlit Cloud, préférez `gpt2`/`distilgpt2` et `SmolLM2-135M`. `gpt2-medium` et `SmolLM2-360M`, les modèles du notebook, sont plutôt destinés à une exécution locale.

## 4. Ce qui nécessite une clé ou un service externe

| Fonctionnalité | Service | Sans le service |
|---|---|---|
| Exécution des prompts de l'atelier ③, scénarios 9.x | Groq (texte) | Copier le prompt dans ChatGPT/Claude/Gemini, coller la réponse |
| Audit d'une capture (6.2), analyse du logo (9.4), légende d'image (⑤ §7) | Groq (vision, `qwen/qwen3.8-27b`, *preview*) | Utiliser un VLM externe et coller le résultat |
| Logo (6.1), images de biais (⑤ §5), chaîne multimodale (⑤ §7) | Hugging Face Inference Providers | Importer les images (outil externe ou notebook dans Colab) |
| Message vocal (6.3, 9.5) | gTTS (service Google, sans clé, FR) ou Groq Orpheus (anglais) | Importer un fichier audio |
| TTS → STT → WER (⑤ §6) | Groq Orpheus/Whisper ou gTTS | Importer les audios, saisir la transcription ; le WER reste calculé |
| Étape ④, générations Groq de ⑤, réponse RAG ⑥ | Groq (texte) | — (message d'erreur explicite) |
| Prototype (③ partie 7) | Lovable (outil externe) | — |

**Fonctionne sans aucun service :** étapes ① et ②, tokenizers et BLEU/chrF/WER/distinct-2 de l'étape ⑤, ainsi que l'import, le découpage, les embeddings et la recherche Top-K de l'étape ⑥. Les modèles locaux (GPT-2, SmolLM2, MarianMT) ne demandent que l'accès au Hub Hugging Face pour leur téléchargement.

## 5. Points d'attention

- **Modèles Groq.** `llama-3.3-70b-versatile`, le modèle par défaut de la version précédente, a été retiré des offres Free/Developer le 16 août 2026. Le modèle par défaut est désormais `openai/gpt-oss-120b`. Ce modèle de raisonnement est appelé avec un effort de raisonnement faible pour que la longueur maximale serve à la réponse. Les modèles *preview*, comme le modèle vision ou Orpheus, peuvent être retirés à court préavis. Le bouton *Lister les modèles Groq accessibles* de la barre latérale affiche le catalogue réel, et chaque identifiant reste modifiable.
- **Orpheus (TTS Groq)** : anglais et arabe saoudien uniquement, 200 caractères maximum. Il faut peut-être accepter les conditions du modèle dans la console Groq avant la première utilisation.
- **Groq ne génère pas d'images.** La génération d'images passe par Hugging Face.
- **Sauvegarde.** Streamlit ne conserve rien entre deux sessions. Les étudiants téléchargent leur progression (fichier JSON : réponses, tableaux, images) depuis la barre latérale et la rechargent pour reprendre. Les étapes ③ et ⑤ exportent aussi un compte rendu Markdown, et l'étape ③ un journal de prompts en CSV.
- **Confidentialité.** Les images et audios envoyés aux services externes quittent l'application. L'étape ③ impose une vérification avant l'envoi d'une capture d'écran au VLM.

## 6. Correspondance avec les sources

- **Étapes ① et ②** : code d'origine conservé à l'identique. Le seul ajout est un bandeau d'objectifs (`step_intro`) en tête de chaque étape.
- **Étape ④** : code d'origine de l'ancienne étape ③. Trois lignes changent : le numéro d'onglet, le titre ④ et le modèle par défaut. Deux éléments sont ajoutés : l'aperçu des messages envoyés et l'historique des générations pour l'expérience comparative.
- **Étape ③** : les 9 parties du support, dans l'ordre, avec leurs prompts mot pour mot. Les prompts sont préremplis par chaînage avec les résultats retenus. S'y ajoutent les tableaux de travail (priorisation, fonctionnalités, écrans, parcours, grille risque/guardrail/indicateur), les 8 scénarios critiques et une synthèse finale.
- **Étape ⑤** : sections 0 à 8 et conclusion du notebook, dans l'ordre. Les fonctions `distinct_n`, `evaluate_translation`, `wer` et le rapport `TokenReport` sont repris à l'identique. Chaque section indique le moteur utilisé et les adaptations, par exemple Stable Diffusion remplacé par Hugging Face ou par l'import d'images Colab, et BLIP remplacé par un VLM ou une légende saisie.

## 7. Dépannage

- *« Aucune clé API configurée »* : vérifiez `GROQ_API_KEY` dans les secrets, ou saisissez une clé dans la barre latérale.
- *Erreur 404 ou « model not found »* : le modèle a été retiré. Choisissez-en un autre dans *Modèles utilisés*.
- *Erreur 429* : limite de débit de l'offre gratuite. Patientez, réduisez le nombre de seeds, ou utilisez une clé personnelle.
- *Installation trop lourde sur Streamlit Cloud* : `torch` est installé par `sentence-transformers`. Pour une version CPU plus légère, on peut ajouter `--extra-index-url https://download.pytorch.org/whl/cpu` en tête de `requirements.txt`. Testez-le d'abord : un index supplémentaire peut modifier la résolution d'autres paquets.
- *PDF sans texte* : PDF scanné. L'OCR n'est pas géré ; utilisez un PDF texte.
