# Journal technique

## 27 septembre 2026

- Lecture du brief et récupération du template officiel c21f9da.
- Générateur conservé octet pour octet ; SHA256 : 2E2BED4400660B117FC1B460A683843185F588C91CB134322A2AAE52118D732B.
- Architecture choisie avant entraînement : grille 4×4, dimension 128, 2 blocs, 4 têtes, 521 307 paramètres. Taille plus petite que l'ordre de grandeur indicatif du brief pour une implémentation accessible.
- Première installation CPU de PyTorch, puis détection de la GTX 1650 et remplacement par PyTorch 2.8.0+cu126.
- Premier lancement pytest : découverte en double des tests dans la copie temporaire du template. Correction avec testpaths=tests dans pytest.ini.
- Deuxième lancement pytest interrompu par le remplacement de PyTorch en cours. Relancé après installation.
- Vérification propre : 107 tests réussis, incluant le test CPU/GPU.
- E0, 600 pas sur 16 exemples : loss 0,000371725 et exact-match 16/16. Cette mesure porte uniquement sur le batch mémorisé.
- Données générées au format officiel : train 20 000, val 2 000, test 2 000, test_heldout 2 000 ; seed 42.
- Un chevauchement bref a eu lieu entre la fin de l'entraînement principal et le début de E1. Les durées par epoch du CSV sont des temps muraux observés, pas une comparaison de performances. S1 sera exécuté séparément des entraînements.
- Le générateur peut décrire deux objets dans les deux ordres, avec relation inverse. Limite relevée par lecture du code, sans modifier ni les étiquettes ni la métrique officielle.

- Verification depuis un clone local propre : 108 tests passent et le rapport est reconstructible depuis les preuves versionnees.
- Ajout de .gitattributes pour conserver les PDF/PNG comme fichiers binaires et eviter une conversion de fins de ligne sous Windows.
