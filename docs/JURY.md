# Questions du jury

Utilise ces réponses comme critères de compréhension. Reformule avec tes mots et montre le fichier concerné.

| Question | Réponse attendue |
|---|---|
| Que fait votre modèle ? | CNN pour obtenir 16 vecteurs visuels, puis décodeur autorégressif qui génère des caractères. |
| Pourquoi un CNN sans préentraînement ? | Exigence du challenge ; apprentissage de l'encodeur avec le reste du modèle. |
| Pourquoi 16 tokens ? | Grille spatiale compacte, coût d'attention plus faible ; détail perdu comme compromis. Pas de supériorité démontrée sans ablation. |
| Pourquoi 128 dimensions ? | Petit modèle adapté au budget local ; choix pratique, pas optimum prouvé. |
| Pourquoi 4 têtes ? | Division exacte en 32 dimensions par tête, plusieurs mélanges appris ; pas une tête par attribut imposée. |
| Qui prédit la première lettre ? | La sortie au dernier token visuel. |
| Pourquoi des positions ? | L'attention ne contient pas seule l'ordre spatial ou textuel. |
| Pourquoi diviser QK par sqrt(dk) ? | Stabiliser l'échelle des scores quand la dimension augmente. |
| Pourquoi softmax sur le dernier axe ? | Chaque query répartit son attention entre les clés. |
| Que voit un token visuel ? | Tous les tokens visuels, aucune lettre. |
| Peut-on calculer toutes les lettres ensemble à l'entraînement ? | Oui, le teacher forcing fournit le préfixe vrai et le masque bloque le futur. |
| Pourquoi la génération est-elle séquentielle ? | La prochaine entrée dépend de la dernière prédiction. |
| Qu'est-ce qu'un gradient ? | Dérivée de la loss par rapport à un paramètre, indiquant une direction locale de variation. |
| Pourquoi zero_grad ? | PyTorch accumule les gradients par défaut. |
| Pourquoi ignorer -100 ? | Le padding ne correspond à aucune cible réelle. |
| Pourquoi pas softmax avant la loss ? | CrossEntropyLoss traite déjà les logits de manière numériquement stable. |
| Pourquoi tester sur un seul batch ? | Isoler les bugs et vérifier la capacité à mémoriser, sans prétendre généraliser. |
| Votre attention est-elle correcte ? | Comparaison numérique à PyTorch, tests de masques, causalité et gradients. |
| À quoi sert le modèle aveugle ? | Séparer les régularités linguistiques de l'information apportée par l'image. |
| Pourquoi letter accuracy peut tromper ? | L'orthographe est prévisible même si taille, couleur ou relation sont fausses. |
| Et une faute comme cirle ? | Shape incorrecte, autres attributs reconnus conservés ; pas d'autocorrection. |
| Pourquoi validation et test séparés ? | Choisir les checkpoints avec val et conserver test pour l'évaluation. |
| GPU toujours plus rapide ? | Non : lancements, petits lots et transferts peuvent dominer ; nos mesures décident pour notre protocole. |
| Pourquoi synchroniser ? | CUDA exécute les kernels de façon asynchrone. |
| Pourquoi warm-up ? | Exclure les coûts d'initialisation et d'allocation des premières itérations. |
| Qu'est-ce qui est hors benchmark ? | DataLoader, CPU→GPU, stockage et création des tenseurs. |
| Pourquoi T² ? | Un score pour chaque paire query/key. |
| Que prouvent vos résultats ? | Répondre avec les nombres présents dans le README, leur protocole et leurs limites. |
| Qu'avez-vous fait avec l'IA ? | Décrire précisément AI_USAGE.md, puis démontrer ce que vous savez modifier vous-même. |

## Présentation de trois minutes

0:00-0:30 : problème, exemples et caractère contrôlé des données.

0:30-1:10 : pipeline, dimensions et rôle du masque.

1:10-1:40 : tests et E0, puis entraînement avec validation.

1:40-2:20 : résultats test et baseline aveugle, avec un exemple d'erreur.

2:20-2:45 : benchmark, warm-up et synchronisation.

2:45-3:00 : une limite réelle et prochaine expérience précise.

## Si tu ne sais pas

Dire ce que tu sais, ce qui manque, puis proposer une vérification. Exemple : « Je n'ai pas comparé 16 et 64 tokens. Je garde le reste fixe, j'entraîne plusieurs seeds et je compare exact-match, relations et débit. »

Une hypothèse n'est pas un résultat. Une phrase apprise sans capacité à la relier au code ne suffit pas.

## Journal personnel à compléter pendant nos séances

- Lignes relues et expliquées sans aide :
- Modification réalisée en direct :
- Test exécuté et interprété :
- Erreur de compréhension corrigée :
- Question encore incertaine :
