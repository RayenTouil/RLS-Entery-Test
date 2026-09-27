# E1 - Le modele utilise-t-il l'image ?

Hypothese enregistree avant execution : le modele voyant les images depassera le modele aveugle en exact-match et en reconnaissance des attributs. Le modele aveugle peut apprendre l'orthographe des mots.

Variable modifiee : tous les pixels sont remplaces par zero, pendant entrainement ET evaluation. Architecture, initialisation, ordre des lots, seed 42, nombre d'epochs, optimiseur et splits restent identiques.

Comparaison : configs/baseline.yaml contre configs/blind.yaml. Choix du checkpoint par la perte de validation, jamais par le test.

Niveau 1 : une seed par condition. Aucun ecart-type entre seeds ne peut etre estime. Aucun resultat ne sera invente. Un test complementaire peut masquer les images du modele normal a l'inference ; ce test n'est pas le baseline E1 entraine separement.
