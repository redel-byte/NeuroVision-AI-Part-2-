# NeuroVision AI — Part 2

> Classification multi-classe d’images IRM cérébrales avec **MobileNetV2**, **transfer learning**, **fine-tuning**, **MLflow Model Registry**, **Grad-CAM** et **Streamlit**.
>
> **Projet pédagogique / démonstrateur de recherche.** Le système ne pose **aucun diagnostic médical**, ne remplace jamais un médecin et n est pas validé pour un usage clinique.

## 1. Contexte et continuité avec la Partie 1

Cette Partie 2 reprend **le même dataset IRM** utilisé dans NeuroVision AI Partie 1, où un **CNN from scratch** a été entraîné. L’objectif est d’évaluer dans quelle mesure un backbone `MobileNetV2` pré-entraîné sur ImageNet permet d’améliorer la généralisation, puis d’adapter ses dernières couches aux IRM. Les résultats sont comparés avec MLflow, et le meilleur modèle approuvé est exposé dans une démonstration Streamlit.

**Important :** conserver les splits et définitions de classes de la Partie 1 si leur validité est documentée. Sinon, reconstruire un split propre et **mentionner que la comparaison avec l’ancien CNN n est pas strictement équivalente**. Ne pas présenter de métriques comme des résultats acquis avant entraînement.

## 2. Objectifs

- Classer chaque IRM en **4 classes** : `glioma`, `meningioma`, `notumor`, `pituitary`.
- Mesurer la progression entre **CNN Part 1**, **MobileNetV2 gelé** et **MobileNetV2 fine-tuné**.
- Analyser accuracy, macro-F1, rappel par classe, faux négatifs tumoraux et temps d’entraînement.
- Fournir des explications visuelles exploratoires (Grad-CAM) sans prétendre à une interprétation clinique certaine.
- Suivre les expériences avec MLflow, enregistrer le candidat approuvé et l’exposer via l’alias `champion`.
- Livrer une interface Streamlit, une documentation reproductible et une planification Jira.

## 3. Pile technique

| Composant | Technologie | Usage |
|---|---|---|
| Modèle | TensorFlow / Keras + MobileNetV2 ImageNet | Classification et fine-tuning |
| Images | OpenCV + tf.data | Lecture, transformations, batching, prefetch |
| Environnement | VastAI, GPU CUDA compatible | Entraînement et stockage persistant |
| Gestion des dépendances | `uv` et `uv.lock` | Reproductibilité des versions |
| Expériences | MLflow Tracking + Model Registry | Hyperparamètres, métriques, artefacts, alias `champion` |
| Visualisation | Matplotlib / scikit-learn / Grad-CAM | Courbes, matrice de confusion, explications |
| Application | Streamlit | Upload, prédiction, scores, visualisation |
| Automatisation (bonus) | Apache Airflow | Orchestration, retries, promotion contrôlée |

## 4. Dataset

Arborescence **attendue** (réutilisée depuis Partie 1) :

```text
data/raw/
├── glioma/
├── meningioma/
├── notumor/
└── pituitary/
```

**Formats autorisés :** `.jpg`, `.jpeg`, `.bmp`, `.png` (sans distinction de casse). Les fichiers illisibles sont exclus avec `try/except` et journalisés. Ne pas versionner les images ni des données personnelles. Ne pas utiliser de données cliniques identifiantes dans les captures ou rapports.

Ordre **fixe** des labels (à garder dans la config, les rapports et l’inférence) :

```python
CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]
```

Le dataset doit être inspecté avant entraînement : taille, formats, images corrompues, répartition par classe, doublons, éventuelles images d’un même patient. Si un ID patient existe, le split doit être défini **par patient**, pas seulement par image. Éviter qu une IRM ou variante quasi identique se retrouve à la fois dans train/val/test.

**Splits** : train, validation et test **disjoints**, stratifiés si possible. Les manifestes et la seed sont conservés dans `data/splits/`. Aucun pourcentage imposé sans examen de la distribution et des splits de la Partie 1. L’ensemble de test est mis à l’écart pendant le choix des modèles; s il a déjà servi à régler les modèles en Partie 1, le mentionner explicitement et réserver un nouvel échantillon vraiment indépendant si disponible.

## 5. Organisation cible du dépôt

> Structure **prévue pour l’implémentation**, à intégrer dans le dépôt existant plutôt que de remplacer les fichiers de la Partie 1. Les fichiers indiqués ci-dessous ne sont pas fournis comme du code exécutable par le présent dossier de planification.

```text
NeuroVisionAI/
├── README.md
├── pyproject.toml
├── uv.lock
├── .gitignore
├── .env.example
├── data/
│   ├── raw/                        # 4 répertoires hérités de Part 1
│   └── splits/                     # Manifestes train/val/test versionnés, sans IRM sensibles
├── src/
│   ├── config.py                   # Classes, chemins, seeds, hyperparamètres
│   ├── reproducibility.py
│   ├── data/
│   │   ├── inspect.py              # Audit, gestion fichiers corrompus
│   │   ├── loader.py               # OpenCV, RGB, 224×224
│   │   ├── dataset.py              # tf.data
│   │   ├── preprocessing.py        # MobileNetV2 preprocessing
│   │   └── augmentation.py         # Train uniquement
│   ├── models/
│   │   └── mobilenetv2.py          # Backbone et nouvelle tête
│   ├── training/
│   │   ├── callbacks.py
│   │   ├── transfer.py
│   │   └── finetune.py
│   ├── evaluation/
│   │   └── evaluate.py
│   ├── explainability/
│   │   └── gradcam.py
│   ├── registry/
│   │   └── promote.py
│   └── inference/
│       └── predict.py
├── app/
│   └── streamlit_app.py
├── scripts/
│   ├── audit_data.py
│   ├── train_transfer.py
│   ├── train_finetune.py
│   ├── evaluate_champion.py
│   └── register_champion.py
├── tests/
├── reports/                        # Figures et rapports générés
├── artifacts/                      # Checkpoints / caches locaux, non versionnés
├── docs/
│   ├── model_card.md
│   ├── champion_policy.md
│   └── governance.md
├── dags/                           # Bonus Airflow
└── planning/
    ├── jira_neurovision_ai_part2_import.csv
    ├── JIRA_IMPORT_GUIDE.md
    └── PLANIFICATION.md
```

## 6. Installation et exécution sur VastAI

### Prérequis

- Une instance VastAI avec GPU reconnu par NVIDIA, un conteneur compatible TensorFlow/CUDA et du stockage persistant.
- Python compatible avec la version TensorFlow choisie, `uv` installé, `pyproject.toml` et `uv.lock` maintenus par le projet.
- Le dataset hérité de la Partie 1 copié dans `data/raw/` **une seule fois** sur le volume persistant.
- Un serveur MLflow accessible par les processus d’entraînement et Streamlit; Registry supporté par le stockage utilisé.

```bash
# Depuis la racine du dépôt existant
uv sync --frozen

# Vérification GPU
uv run python -c "import tensorflow as tf; print('TF:', tf.__version__); print('GPUs:', tf.config.list_physical_devices('GPU'))"
```

Les commandes ci-dessous définissent l **interface CLI cible à implémenter** dans les tickets Jira; elles **ne fonctionneront pas** tant que les scripts correspondants n auront pas été créés :

```bash
# Après implémentation des scripts listés dans l’arborescence
uv run python scripts/audit_data.py
uv run python scripts/train_transfer.py
uv run python scripts/train_finetune.py
uv run python scripts/register_champion.py
uv run python scripts/evaluate_champion.py
uv run streamlit run app/streamlit_app.py
```

Pour un MLflow local **de développement**, prévoir un backend et stockage d’artefacts persistants (éviter le stockage éphémère d’un conteneur). Exemple de lancement à adapter à la version MLflow choisie :

```bash
uv run mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlartifacts --host 127.0.0.1 --port 5000
```

Dans un environnement distant, utiliser une URI de suivi contrôlée, sécuriser accès et ports et **ne pas exposer publiquement sans authentification**. Le serveur de tracking doit rester accessible lorsque Streamlit exécute les prédictions. Configurer via les variables d’environnement (ne jamais commiter des credentials) :

```bash
export MLFLOW_TRACKING_URI="http://127.0.0.1:5000"
export MLFLOW_REGISTERED_MODEL_NAME="NeuroVisionMRIClassifier"
```

## 7. Pipeline de traitement

1. **Audit** : scanner les 4 dossiers, filtrer les extensions, relever les corruptions, compter les classes.
2. **Split gelé** : préserver l’alignement `chemin ↔ label`, créer manifests des jeux train/val/test, limiter fuites patient/doublon.
3. **Décodage** : `cv2.imread()` puis conversion **BGR → RGB**; si IRM en niveaux de gris, convertir explicitement en 3 canaux; redimensionner en `(224,224)`.
4. **Performance** : `tf.data` avec `map`, `batch` et `prefetch`; cache seulement si les ressources le permettent; `shuffle` avec seed sur train uniquement.
5. **Augmentation** : rotations légères, translations et zoom contrôlés sur train uniquement. Ne pas appliquer de flips arbitraires ou transformations pouvant modifier la signification clinique sans justification.
6. **Prétraitement** : `tf.keras.applications.mobilenet_v2.preprocess_input`, appliqué **une seule fois**, transforme les pixels à partir de `[0,255]` vers `[-1,1]`. Ne pas diviser d’abord par 255. L’inférence doit utiliser le même protocole.
7. **Encodage** : soit entiers `0..3` + `SparseCategoricalCrossentropy`, soit one-hot + `CategoricalCrossentropy` (pas de mélange).

## 8. Étapes ML — transfer learning et fine-tuning

### Étape A — MobileNetV2 gelé

- `tf.keras.applications.MobileNetV2(weights="imagenet", include_top=False, input_shape=(224,224,3))`.
- `base_model.trainable = False`; utiliser `base_model(inputs, training=False)` pour stabiliser BatchNormalization.
- Tête suggérée : `GlobalAveragePooling2D` → `Dropout` → `Dense(4, activation="softmax")`.
- Optimiseur Adam, taux d’apprentissage initial à tester sur **validation**; loss adaptée aux étiquettes.
- Mesurer le temps (`time.perf_counter`), sauvegarder history, métriques, meilleur checkpoint sur `val_loss`.

### Étape B — Fine-tuning

- Repartir du **meilleur modèle entraîné** de l’étape A, ne pas recharger un MobileNetV2 vierge de sa tête apprise.
- Dégeler seulement un nombre contrôlé de blocs profonds, garder couches précoces gelées; maintenir BatchNormalization en mode inférence.
- **Recompiler après modification des couches `trainable`**, choisir un learning rate beaucoup plus faible (par exemple `1e-5`, à comparer sur validation).
- Activer `ModelCheckpoint`, `EarlyStopping(restore_best_weights=True)` et `BackupAndRestore` avec emplacements distincts; documenter leurs rôles et la reprise après interruption.
- Comparer **sur validation**, avec rappel des classes tumorales, macro-F1, accuracy et temps. Évaluer une unique fois le champion figé sur le test final.

## 9. Expériences MLflow et alias champion

Créer une expérience claire : `NeuroVisionAI-Part2` avec trois familles de modèles :

| Run | Origine | Ce que l’on doit enregistrer |
|---|---|---|
| `cnn_part1_baseline` | Expérience historique de Partie 1 | Version, provenance, métriques et conditions de comparaison |
| `mobilenetv2_transfer` | Backbone gelé | Seed, split hash, LR, epochs, métriques validation, temps et courbes |
| `mobilenetv2_finetune` | Quelques blocs dégelés | Blocs trainables, LR réduit, métriques, checkpoints, artefacts |

La métrique de sélection principale recommandée est le **macro-F1 sur validation**. Vérifier aussi les rappels de `glioma`, `meningioma` et `pituitary`, car les faux négatifs sont sensibles. Les seuils minimum et marges acceptables doivent être définis et documentés **avant** de regarder le test. Toute comparaison trompeuse entre splits différents doit être signalée.

Le modèle retenu est **enregistré** sous le nom `NeuroVisionMRIClassifier`; l’alias `champion` désigne la version approuvée et **ne signifie pas une validation médicale**. En inférence :

```python
import os
import mlflow.tensorflow

model_name = os.getenv("MLFLOW_REGISTERED_MODEL_NAME", "NeuroVisionMRIClassifier")
model_uri = f"models:/{model_name}@champion"
model = mlflow.tensorflow.load_model(model_uri)  # Modèle réellement enregistré requis
```

Si l’application est déployée sur une autre machine, le Registry et le stockage d’artefacts doivent être accessibles à cette machine, avec versions de dépendances compatibles. Tester la relecture de la version dans une session vierge.

## 10. Évaluation et explicabilité

La version champion doit être **figée avant le test**. À produire : rapport de classification (macro F1, accuracy, precision, recall et supports), matrice de confusion, rappel par tumeur, faux négatifs et exemples d’erreurs anonymisés. Conserver nombre d’exemples et incertitude des estimations, surtout pour les petites classes.

Utiliser Grad-CAM sur une couche convolutive adaptée de MobileNetV2 pour mettre en évidence les régions qui contribuent à la prédiction. **Grad-CAM n est ni une preuve causale ni une segmentation fiable**, et ne localise pas nécessairement une tumeur. Montrer les limites dans la documentation et l’interface.

## 11. Démonstration Streamlit

L’application est conçue pour :

- accepter des IRM `.jpg/.jpeg/.png/.bmp`, rejeter proprement les fichiers invalides et corrompus;
- afficher l’image chargée, la classe la plus probable, les **quatre scores** dans l’ordre officiel et la version `champion`;
- permettre (optionnel) la visualisation Grad-CAM avec un avertissement sur son caractère exploratoire;
- garantir le même prétraitement qu à l’entraînement;
- afficher clairement **« Outil pédagogique d’aide à la décision — pas un diagnostic médical »**;
- lire le modèle via **MLflow alias**, jamais uniquement à partir d’un checkpoint local.

## 12. Bonus — Orchestration Airflow

DAG cible :

```text
validate_data --> prepare_splits --> train_transfer --> train_finetune
    --> compare_validation --> register_candidate --> gate_and_promote
```

Ajouter retries, logs, timeouts et tâches idempotentes. La promotion du candidat par `gate_and_promote` est **conditionnée à la validation** (macro-F1 et gardes de rappel), jamais aux métriques du holdout de test. Les valeurs/secrets nécessaires au DAG sont fournis via configuration sécurisée. Cette étape est **optionnelle** et ne bloque pas la livraison de la Partie 2.

## 13. Qualité et reproductibilité

- Versions figées par `uv.lock` et seed déclarée; préciser ce qui ne peut pas être parfaitement déterministe sur GPU.
- Manifestes de split, hashes de données, ordre des labels et paramètres archivés.
- Contrôles des formes et prétraitements, tests d’images malformées, smoke tests de l’app et du Registry.
- Checkpoints et métadonnées d’expériences sauvegardés sur volume persistant, jamais dans un conteneur éphémère uniquement.
- Pas de secrets, jeux de données IRM, fichiers patients ou checkpoints volumineux poussés dans Git.
- Aucune recommandation diagnostique ni revendication d’usage clinique.

## 14. Planification et livraison

La **planification détaillée** est fournie dans `planning/PLANIFICATION.md`. Le fichier `planning/jira_neurovision_ai_part2_import.csv` décrit **8 Epics et 52 tickets** (dont 5 bonus Airflow), avec critères d’acceptation et dépendances internes. Voir `planning/JIRA_IMPORT_GUIDE.md` pour le mappage Jira Cloud. La période de référence proposée est **19 octobre – 20 novembre 2026** (5 sprints d’une semaine); bonus jusqu au **27 novembre 2026**.

**Livrables attendus :** modèle transfer et fine-tuned, runs MLflow comparables, Registry `champion`, résultats du test holdout, matrice de confusion, visualisations Grad-CAM, Streamlit, tests, README et ticketing Jira. Aucun de ces livrables techniques n est présenté comme déjà implémenté dans le présent fichier de planification.

## 15. Documentation officielle

- [MobileNetV2 Keras](https://keras.io/api/applications/mobilenet/mobilenet_models/)
- [Prétraitement MobileNetV2](https://keras.io/api/applications/mobilenet/mobilenet_preprocessing/)
- [Transfer learning et fine-tuning Keras](https://keras.io/guides/transfer_learning/)
- [MLflow Model Registry et alias](https://www.mlflow.org/docs/latest/registry/)
- [MLflow TensorFlow](https://mlflow.org/docs/latest/ml/deep-learning/tensorflow)
- [Streamlit](https://docs.streamlit.io/)
- [Apache Airflow](https://airflow.apache.org/docs/apache-airflow/stable/concepts/dags.html)
- [Atlassian : préparer un CSV](https://support.atlassian.com/jira-software-cloud/docs/prepare-a-csv-file-for-import/)

---

*NeuroVision AI Part 2 — classification IRM à visée pédagogique, documentée et non clinique.*
