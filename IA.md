# Utilisation de l'intelligence artificielle

J'ai utilisé ChatGPT comme assistant pédagogique pour m'accompagner dans la réalisation du TP4 sur le RAG avec LangChain4j.

### 1. Configuration du projet

- Aide à la création de la structure du projet Maven et à l'organisation des packages Java.
- Assistance dans la configuration du `pom.xml` et la compréhension des dépendances LangChain4j.
- Aide à la préparation de l'environnement de travail et à la vérification de la compilation du projet.
- Accompagnement dans la configuration des clés API Gemini et Tavily, sans les intégrer directement au code source.

### 2. Réalisation des tests

- **Test 1 – RAG naïf :** accompagnement dans la compréhension du fonctionnement du RAG, du chargement du fichier `rag.pdf`, de l'extraction du texte et de la recherche des passages pertinents.
- **Test 2 – Transformation de question :** explication du rôle du `CompressingQueryTransformer` et aide à comprendre son utilisation pour améliorer les requêtes.
- **Test 3 – Routage :** accompagnement dans la mise en place du routage entre deux sources documentaires. Aide à la préparation du fichier `autre.pdf`, contenant des informations sur le système solaire et l'astronomie, afin de tester la sélection de la bonne source.
- **Test 4 – Routage personnalisé :** explication du fonctionnement d'un routeur personnalisé et de la sélection entre une réponse directe et une recherche documentaire.
- **Test 5 – Recherche Web :** accompagnement dans l'intégration de Tavily et la compréhension du `LanguageModelQueryRouter` pour orienter les questions vers les documents locaux ou la recherche Web.

### 3. Débogage et vérification

- Analyse des erreurs de compilation et d'exécution.
- Aide à l'identification des problèmes de configuration et de dépendances.
- Vérification du fonctionnement des différents tests.
- Analyse des logs pour comprendre les décisions de routage et les sources sélectionnées.

## Résultats obtenus

Les cinq tests ont été réalisés et vérifiés à travers les exécutions du projet.

Pour le routage avec recherche Web, les logs ont notamment permis de vérifier :

- La sélection du document PDF pour une question concernant le fine-tuning.
- La sélection de Tavily pour une question concernant les actualités scientifiques.

## Responsabilité

J'ai utilisé les propositions et les explications fournies par ChatGPT comme support d'apprentissage. J'ai suivi les différentes étapes, intégré les modifications dans mon projet, exécuté les tests et vérifié les résultats obtenus.