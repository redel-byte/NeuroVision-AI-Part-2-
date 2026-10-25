Project context
Contexte du projet
NeuroVision AI est une solution de vision par ordinateur dédiée à l'aide au diagnostic des tumeurs cérébrales. Elle classe des images IRM en quatre catégories : gliome, méningiome, tumeur de l'hypophyse et absence de tumeur.

Cette Part 2 reprend le dataset de la Part 1 (CNN from scratch) avec MobileNetV2, un réseau léger pré-entraîné sur ImageNet, puis une phase de fine-tuning pour l'adapter aux IRM.

Outils : VastAI, TensorFlow/Keras, OpenCV, MLflow et une interface interactive.

L'outil reste une aide à la décision et ne se substitue jamais au médecin.

Objectifs :
classer les IRM en 4 catégories (glioma, meningioma, pituitary, notumor) ;
obtenir de meilleures performances que le CNN from scratch ;
assister les médecins dans l'interprétation des résultats ;
expliquer les décisions du modèle (IA explicable).
Objectifs pédagogiques
préparer un jeu de données d'images pour un modèle pré-entraîné ;
expliquer le transfer learning et le fine-tuning ;
suivre et comparer des expériences avec MLflow ;
analyser les erreurs et interpréter les décisions du modèle ;
livrer un projet reproductible, documenté et planifié.
Dataset
Même dataset que la Part 1 : 4 dossiers, un par classe (nom du dossier = nom de la classe) :

data/raw/

├── glioma/

├── meningioma/

├── notumor/

└── pituitary/

Travail demandé
Découverte et préparation des données
Parcourir les dossiers, écarter les extensions non conformes (jpeg, jpg, bmp, png) et gérer les images corrompues avec try-except.
Charger et mélanger images et étiquettes (même indice), puis redimensionner avec OpenCV (224×224, 3 canaux).
Encapsuler le chargement dans un module réutilisable et l'accélérer (ex. tf.data).
Afficher la répartition par classe et des exemples d'images ; rééquilibrer si besoin.
Encoder les étiquettes et appliquer le prétraitement propre à MobileNetV2.
Diviser les données en ensembles d'entraînement, de validation et de test.
Appliquer une augmentation adaptée aux IRM (rotations légères, zoom, décalages), juste sur les données d'entraînement.
Transfer learning
Charger MobileNetV2 pré-entraîné sur ImageNet, sans sa couche de classification d'origine.
Geler la base du modèle afin de conserver les connaissances déjà apprises.
Geler : les poids de la base ne changent pas pendant l'entraînement.
Ajouter une nouvelle tête de classification adaptée aux 4 classes et justifier tes choix (couches, fonction d'activation de sortie, fonction de perte).
Exemple : pooling global, Dropout, sortie à 4 neurones softmax, entropie croisée catégorielle.
Compiler le modèle avec un optimiseur adapté (ex. Adam).
Vérifier l'architecture avec model.summary() et comprendre la différence entre paramètres entraînables et non entraînables.
Entraînables : la tête. Non entraînables : la base gelée.
Entraîner la nouvelle tête avec les données de validation et mesurer la durée de l'entraînement avec la bibliothèque time.
Visualiser les courbes d'apprentissage (accuracy / loss) et les interpréter.
Validation qui stagne alors que l'entraînement progresse : surapprentissage.
Fine-tuning
Repartir du modèle entraîné en transfer learning.
Dégeler une partie des dernières couches de MobileNetV2 pour les adapter aux IRM.
Les premières couches (motifs génériques) restent gelées ; les dernières sont adaptées aux IRM.
Recompiler avec un taux d'apprentissage beaucoup plus faible et expliquer pourquoi.
Un taux plus faible (ex. 0,0001) ajuste doucement des poids déjà bons, sans effacer ce qu'ImageNet a appris.
Réentraîner le modèle et mesurer la durée.
Comparer les performances avant et après fine-tuning.
Entraînement et suivi
Sauvegarder le meilleur modèle (ModelCheckpoint), reprendre après interruption (BackupAndRestore) et limiter le surapprentissage (EarlyStopping, Dropout).
Suivi des expériences avec MLflow
Suivre et comparer les expériences (CNN from scratch, transfer learning, fine-tuning).
MLflow garde la trace de chaque entraînement (un « run ») et permet de les comparer.
Enregistrer hyperparamètres, métriques (accuracy, loss, F1-score), durée et artefacts.
Hyperparamètres = réglages ; métriques = résultats ; artefacts = fichiers produits.
Enregistrer le meilleur modèle dans le Model Registry avec l'alias champion.
L'alias champion désigne la version à utiliser : changer de modèle revient à le déplacer.
Évaluation et analyse des erreurs
Évaluer le meilleur modèle sur l'ensemble de test, utilisé une seule fois.
Générer une matrice de confusion et un rapport de classification, avec une attention particulière aux classes tumorales (faux négatif = plus grave).
Déploiement / utilisation
Charger le modèle depuis MLflow via l'alias champion, pas depuis un fichier local.
Créer une fonction predict_image(path) qui utilise le champion, en gérant les entrées invalides.
Développer une interface Streamlit pour tester les prédictions.
Bonus : automatisation du pipeline avec Airflow
Découper le projet en étapes (données, entraînement, évaluation, enregistrement) dans un DAG Airflow.
Un DAG est le schéma des tâches, une par étape, exécutées sur la machine d'Airflow (ici l'instance Vast).
Gérer les dépendances et les échecs, planifier l'exécution et la suivre depuis Airflow.
Une tâche ne démarre que si la précédente a réussi ; sinon Airflow réessaie ou signale l'erreur.
Relier à MLflow et ne déplacer l'alias champion que si le nouveau modèle est meilleur.
Une dernière tâche compare le nouveau modèle au champion actuel avant de déplacer l'alias.
Documentation et reproductibilité
Commenter le code (Markdown si en notebook).
Fournir un README.md : description du projet, structure des fichiers, instructions d'exécution.
Planifier les tâches dans Jira sous forme d'Epics et de tickets.