# E1 - Le modele utilise-t-il l'image ?

Hypothese enregistree avant execution : le modele voyant les images depassera le modele aveugle en exact-match et en reconnaissance des attributs. Le modele aveugle peut apprendre l'orthographe des mots.

Variable modifiee : tous les pixels sont remplaces par zero, pendant entrainement ET evaluation. Architecture, initialisation, ordre des lots, seed 42, nombre d'epochs, optimiseur et splits restent identiques.

Comparaison : configs/baseline.yaml contre configs/blind.yaml. Choix du checkpoint par la perte de validation, jamais par le test.

Niveau 1 : une seed par condition. Aucun ecart-type entre seeds ne peut etre estime.

Resultats seed 42 : exact-match test 68,05 % pour le modele visuel contre 1,70 % pour E1. Lettres avec prefixe vrai : 98,52 % contre 87,34 %. Lettres libres : 68,99 % contre 16,17 %. Le contraste soutient l usage de l image, mais une seed ne quantifie pas la variance entre entrainements. La forte exactitude des lettres avec prefixe vrai montre les regularites orthographiques. Voir les JSON pour les attributs et leurs denominateurs.
