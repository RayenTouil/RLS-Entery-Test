# Comprendre et défendre Tiny VLM en 5 heures

Tu connais Python. L'objectif est de comprendre et modifier cette petite implémentation, pas de maîtriser tout le deep learning en cinq heures. Le temps de calcul est distinct du temps d'apprentissage. Le projet vise le niveau 1 ; les niveaux 2 et 3 ne sont pas revendiqués.

## Planning précis

| Séance | Minutes | Travail actif | Preuve de compréhension |
|---|---:|---|---|
| Aujourd'hui, 27 septembre | 0-10 | Problème, données, train/val/test | Décrire une scène et sa cible |
| Aujourd'hui | 10-25 | Tenseurs, tokenizer, padding | Encoder red à la main |
| Aujourd'hui | 25-40 | CNN, grille, adapter | Donner les dimensions à chaque étape |
| Aujourd'hui | 40-55 | Entrées et cibles décalées, génération | Dire qui prédit la première lettre |
| Aujourd'hui | 55-60 | Quiz sans regarder | Expliquer le projet en 60 secondes |
| Demain, 28 septembre | 0-20 | Q, K, V, produit matriciel, softmax | Calculer une attention miniature |
| Demain | 20-40 | Têtes, masque, positions, résidus, LayerNorm | Dessiner le masque |
| Demain | 40-60 | Loss, autograd, AdamW, validation | Expliquer les lignes de train.py |
| Demain | 60-80 | Tests et E0 | Expliquer pourquoi mémoriser un lot est utile |
| Demain | 80-105 | E1 et métriques | Distinguer orthographe et vision |
| Demain | 105-120 | Modifier et tester le nombre de têtes | Prédire ce qui casse si 128 / heads n'est pas entier |
| Après-demain, 29 septembre | 0-25 | CPU/GPU, kernels, synchronisation | Expliquer le benchmark |
| Après-demain | 25-45 | Courbes et erreurs réellement observées | Lire nos résultats sans les exagérer |
| Après-demain | 45-65 | Arborescence, commandes, Git | Reproduire un forward et les tests |
| Après-demain | 65-85 | Rapport, limites, usage de l'IA | Présentation de 3 minutes |
| Après-demain | 85-120 | Simulation du jury et modification en direct | Réponses personnelles, sans récitation |

## Séance 1 : l'image devient des nombres, puis un mot

### Le problème

Une image contient un ou deux objets. Chaque objet possède une taille, une couleur et une forme. Avec deux objets, une relation spatiale s'ajoute. La cible est un mot sans espaces, par exemple largeredcircle. Le modèle doit générer les lettres ; nous n'utilisons pas un classifieur qui choisit le mot entier dans un catalogue.

Une fonction classique possède une règle écrite par le programmeur. Un réseau possède des paramètres numériques appris à partir d'exemples. Les opérations sont écrites dans le code, leurs poids sont ajustés par l'entraînement.

### Tenseur, batch et dimension

Un tenseur est un tableau multidimensionnel. Une image a la forme (3,64,64) : trois canaux rouge/vert/bleu, 64 lignes, 64 colonnes. Un batch est un groupe d'images. (B,3,64,64) signifie B images traitées ensemble. Ce B n'est pas un paramètre appris.

Les pixels sont stockés en uint8 de 0 à 255 pour économiser la RAM. Dataset les convertit en float32 et divise par 255 pour fournir des valeurs entre 0 et 1 au modèle.

- Dataset : répond à « donne-moi l'exemple numéro i ».
- DataLoader : choisit et regroupe plusieurs exemples.
- collate : empile les images et complète les séquences à une longueur commune.

### Tokenizer

Le vocabulaire est a...z, puis eos. a vaut 0, b vaut 1, z vaut 25 et eos vaut 26. L'encodage de red est [17,4,3,26]. eos signifie « le mot est terminé ».

Aucun token de début n'est nécessaire : le dernier vecteur visuel prédit la première lettre. Aucun token de padding n'est ajouté au vocabulaire : dans les entrées, on utilise 0 ; dans les cibles, -100 indique une position à ignorer. -100 ne passe jamais dans nn.Embedding.

Le padding est à droite. Les prédictions utiles, situées avant ce padding, ne peuvent pas le voir grâce au masque causal. Les sorties situées dans la zone de padding ne comptent pas dans la loss.

Exercice : écrire les ids de blue puis expliquer pourquoi les cibles de red et blue doivent être complétées quand elles appartiennent au même batch.

### CNN

Une convolution applique le même petit filtre à plusieurs positions de l'image. Ses poids sont appris. Elle transforme les pixels en caractéristiques locales : contrastes, contours, mélanges de couleurs. Nous ne décidons pas qu'un filtre « détecte un cercle » ; cela dépend de l'apprentissage.

Un stride de 2 réduit la taille spatiale. ReLU remplace les valeurs négatives par zéro et introduit une non-linéarité. AvgPool2d(2) moyenne chaque bloc 2×2 pour réduire encore la grille.

| Étape | Forme |
|---|---|
| image | (B,3,64,64) |
| convolution 1 | (B,32,32,32) |
| convolution 2 | (B,64,16,16) |
| convolution 3 | (B,128,8,8) |
| pooling | (B,128,4,4) |
| flatten spatial | (B,128,16) |
| transpose | (B,16,128) |
| projection adapter | (B,16,128) |

Les 16 vecteurs correspondent à une grille 4×4. Conserver plusieurs positions aide à représenter les relations spatiales. La grille est petite pour limiter le calcul, au prix d'une perte de détail. Un vecteur visuel résume une région avec un champ réceptif ; il ne correspond pas forcément à un objet unique.

L'adapter est une couche Linear qui transforme les caractéristiques du CNN dans l'espace utilisé par le décodeur. Même quand les dimensions sont toutes deux 128, la projection apprend un changement de représentation.

### Entrées et cibles : le point le plus important

Pour la cible red, les positions prédisent ceci :

| Position utilisée | Information déjà visible | Cible |
|---|---|---|
| dernier token visuel | image | r |
| token r | image + r | e |
| token e | image + r + e | d |
| token d | image + r + e + d | eos |

À l'entraînement, le vrai préfixe est fourni : c'est le teacher forcing. À l'inférence, on utilise les lettres déjà prédites. Une erreur peut alors en provoquer d'autres.

Si T lettres sont données en entrée, le modèle retourne T+1 groupes de 27 scores : une sortie pour le dernier token visuel, puis une pour chaque lettre. Chaque score est un logit. Un logit n'est pas directement une probabilité.

Greedy decoding prend argmax : le token dont le score est maximal. On recommence jusqu'à eos ou 45 lettres. La convolution n'est calculée qu'une fois par image pendant cette génération ; le décodeur est recalculé sur le préfixe complet, sans cache KV, pour garder le code simple.

### Quiz de fin de séance

1. Pourquoi 27 tokens ?
2. À quoi correspondent B, 16 et 128 ?
3. Pourquoi le dernier token visuel peut-il prédire r sans token de début ?
4. Pourquoi -100 n'est-il pas un token du vocabulaire ?
5. Pourquoi ne faut-il pas choisir un modèle avec son score sur test ?

Réponses attendues : arrêt variable ; batch / positions visuelles / dimension ; sa sortie est alignée avec la première cible ; valeur ignorée seulement dans la loss ; cela adapterait nos choix au test et biaiserait la mesure.

## Séance 2 : attention et apprentissage

### L'attention

Chaque position produit trois vecteurs par trois couches Linear :
- Q, query : ce que cette position cherche ;
- K, key : ce que les autres positions proposent pour calculer une correspondance ;
- V, value : le contenu à mélanger.

Formule : softmax(Q Kᵀ / sqrt(d_head) + masque) V.

QKᵀ calcule un score pour chaque couple query/key. Diviser par sqrt(d_head) évite que la variance des scores augmente trop avec la dimension et sature le softmax. Ce n'est pas une division par la longueur de la séquence.

Le softmax est appliqué sur les clés, dernier axe. Il transforme les scores d'une query en poids positifs dont la somme vaut 1. Les positions interdites reçoivent -inf avant softmax et donc un poids nul.

Exercice : scores [0,0] donnent [0,5 ; 0,5]. Avec valeurs [2,8], la sortie vaut 5. Masquer la seconde position donne [1,0], donc la sortie vaut 2.

### Plusieurs têtes

Avec d_model=128 et 4 têtes, chaque tête a 32 composantes.
(B,T,128) devient (B,4,T,32).
Les scores ont la forme (B,4,T,T).
Après le mélange avec V, on retrouve (B,4,T,32), puis (B,T,128).

Les têtes peuvent apprendre différentes correspondances. On ne leur attribue pas à l'avance les rôles « couleur », « taille », etc. Une projection finale mélange leur information.

### Le masque et les positions

Les 16 tokens visuels se voient tous entre eux. Ils ne voient aucune lettre. Une lettre voit les tokens visuels, les lettres précédentes et elle-même. Elle doit prédire la lettre suivante, donc voir sa propre entrée est autorisé.

Le token visuel final voit toute l'image, mais aucune réponse textuelle. Cela empêche aussi une fuite indirecte : une lettre ne peut pas transmettre une lettre future à travers les tokens visuels.

L'attention seule ne sait pas quel token était à gauche ou en premier. Un embedding de position appris est ajouté à chaque vecteur. Pour la grille, l'ordre de flatten est fixe, ligne après ligne.

### Decoder block

LayerNorm normalise les composantes d'un token, pas le batch. Ici elle intervient avant attention et avant MLP : architecture pre-norm.

Le MLP est Linear(128,512), GELU, Linear(512,128). Il transforme chaque position séparément. L'attention échange l'information entre positions ; le MLP transforme l'information au sein d'une position.

Les connexions résiduelles additionnent l'entrée au résultat : x + f(x). Elles offrent un chemin direct pour l'information et les gradients. Deux blocs sont empilés.

### Loss et entraînement

CrossEntropyLoss pénalise la faible probabilité de la bonne lettre. Pour une position, elle vaut -log(p_bonne_lettre). Elle reçoit les logits directement : appliquer softmax avant CrossEntropyLoss serait une erreur.

À chaque batch :
1. zero_grad efface les anciens gradients ;
2. forward calcule les logits et la loss ;
3. backward utilise la règle de dérivation en chaîne pour calculer les gradients ;
4. clip_grad_norm limite la norme globale des gradients à 1 ;
5. AdamW met à jour les paramètres, avec une moyenne des gradients et de leurs carrés.

Le learning rate règle l'amplitude des mises à jour. Weight decay pénalise progressivement les poids pour limiter des valeurs inutiles trop grandes. Une epoch est un passage sur les 20 000 exemples d'entraînement.

model.train() et model.eval() indiquent le mode des couches concernées. torch.no_grad() ou inference_mode() empêchent de construire le graphe nécessaire au backward. Ce sont deux idées différentes.

La loss affichée est pondérée par le nombre de cibles non ignorées. Faire une moyenne simple des moyennes des batches pourrait biaiser le résultat quand le dernier lot ou les longueurs changent.

### Vérifications

Le test officiel compare l'attention à la fonction de référence de PyTorch avec les mêmes projections et le même masque. Les tests vérifient aussi que changer une lettre future ne modifie pas une sortie passée et que les gradients atteignent les poids.

E0 : mémoriser 16 exemples vérifie que les données, les cibles, le masque et l'optimisation peuvent fonctionner ensemble. Réussir E0 ne prouve pas la généralisation.

### Évaluation et baseline aveugle

Exact-match vaut 1 seulement si le mot entier est identique. C'est exigeant, mais clair.

Le parser conserve les attributs reconnus même si un autre est incorrect : largeredcirle conserve large et red, mais shape vaut None. Il ne corrige pas cirle en circle, ce qui gonflerait artificiellement le score.

Les tailles/couleurs/formes sont comptées sur les objets présents dans la référence. Les relations sont évaluées seulement pour les scènes à deux objets. Le nombre d'objets possède un score séparé. Les positions objet 1 / objet 2 sont conservées.

Letter accuracy est la proportion de lettres identiques à la même position, avec max(longueur prédite, longueur référence) au dénominateur. Elle est sensible aux décalages et n'est pas une distance d'édition.

E1 entraîne le même modèle avec des images nulles. Le texte possède beaucoup de régularités : après circ, écrire le est facile sans rien voir. C'est pourquoi une bonne exactitude des lettres ne suffit pas à prouver la compréhension visuelle.

## Séance 3 : mesures et défense

### CPU et GPU

Le CPU dispose de quelques cœurs polyvalents. Le GPU exécute beaucoup d'opérations simples en parallèle. Il est utile quand le travail est assez grand pour amortir les coûts de lancement et de transfert.

Une opération PyTorch lance un ou plusieurs kernels GPU. Les appels CUDA sont asynchrones : Python peut continuer alors que le GPU travaille encore. Sans synchronize avant et après le chronométrage, on risque de mesurer seulement les lancements.

Les premières itérations incluent des initialisations, allocations et états d'optimiseur. On les exclut par un warm-up. Les répétitions donnent une moyenne et une dispersion, pas une vérité universelle.

S1 mesure des tenseurs synthétiques déjà sur le périphérique, longueur maximale. Il inclut forward, loss, backward, clipping et AdamW. Il exclut génération des données, DataLoader, transfert et sauvegarde. Ce débit n'est donc pas celui de toute l'application.

### Coût mémoire

Une matrice de scores d'attention a B × H × T² valeurs. Doubler T multiplie cette partie par environ quatre. Cela ne signifie pas que toute la mémoire du modèle quadruple : il y a aussi paramètres, gradients, états AdamW et autres activations.

Pour 16 tokens visuels + 45 lettres, T=61. À B=64 et H=4, une seule matrice float32 contient 64×4×61²×4 octets, environ 3,63 Mio. Ce n'est pas la mémoire totale d'entraînement.

En float32 avec AdamW, paramètres + gradients + deux moments occupent environ 16 octets par paramètre, avant les activations et allocations temporaires.

### Limites spécifiques à ce dataset

Le générateur ne fixe pas un ordre canonique des deux objets. A leftof B et B rightof A peuvent décrire une scène équivalente, mais l'exact-match et les scores ordonnés pénalisent la seconde phrase si la référence contient la première. Cette ambiguïté peut limiter les scores ; son impact doit être mesuré avant toute conclusion quantitative.

Un seul seed ne permet pas d'estimer la variation entre entraînements. La dispersion S1 porte sur des répétitions de chronométrage, pas sur des seeds de précision. Nous n'inventons pas un écart-type pour n=1.

### Expliquer les fichiers

| Fichier ou dossier | Rôle et raison de la séparation |
|---|---|
| generate_data.py | Contrat de données officiel ; conservé intact |
| tokenizer.py | Conversion texte ↔ ids ; testable sans réseau |
| data.py | Chargement et batches ; indépendant de l'architecture |
| encoder.py | Extraction visuelle |
| attention.py | Opération délicate isolée pour le test de référence |
| decoder.py | Bloc réutilisable attention + MLP |
| vlm.py | Assemblage image/texte, positions, masque et logits |
| train.py | Apprentissage, validation et checkpoints |
| generate.py | Inférence sans accès à la bonne réponse |
| evaluate.py | Parser et métriques indépendants de l'optimisation |
| utils.py | Seeds, configuration, matériel et sauvegarde des courbes |
| configs/ | Réglages versionnés ; évite de changer le code pour chaque expérience |
| tests/ | Preuves de correction |
| experiments/ | Hypothèses, logs bruts, prédictions et figures |
| benchmarks/ | Mesures de performance avec protocole explicite |
| report/ | Synthèse technique courte |
| docs/ | Cours et préparation orale ; ajout pédagogique |
| AI_USAGE.md | Transparence sur l'assistance reçue |
| requirements.txt | Versions nécessaires à la reproduction |

Une séparation par responsabilité permet de changer le CNN sans réécrire le tokenizer, ou le parser sans réentraîner. Il n'y a pas de magie dans les noms : ce sont des frontières qui rendent le programme vérifiable.

## Exercices de modification en direct

1. Remplacer n_heads=4 par 8. Prédire d_head=16 et vérifier les tests.
2. Essayer 3 têtes. Expliquer le ValueError avant de regarder son texte.
3. Supprimer temporairement le masque dans une copie. Observer le test de causalité échouer, puis restaurer.
4. Modifier la longueur maximale de génération à 10. Expliquer les mots tronqués.
5. Expliquer et modifier le parseur pour un exemple mal orthographié.
6. Charger un checkpoint et générer une prédiction sans regarder sa cible.
7. Lire le CSV S1 et recalculer images/s = batch × steps / secondes.
8. Expliquer pourquoi train_loss basse et test mauvais peuvent coexister.

Ne pas publier les modifications volontairement incorrectes. Chacun de ces exercices doit produire une explication personnelle.

## Nos mesures a discuter

Le modele principal atteint 68,05 % des mots exacts sur test, contre 1,70 % pour E1. Le modele aveugle atteint pourtant 87,34 % des lettres si le prefixe vrai est fourni : distinguer cette aide de la generation libre. Le heldout tombe a 0 % : reconnaitre des combinaisons connues ne garantit pas de les recombiner.

Sur les 639 erreurs strictes du modele principal, 381 correspondent aux memes objets dans l ordre inverse avec relation inverse. Cela nuance les erreurs sans changer le score officiel. Les relations et les scenes a deux objets restent difficiles.
