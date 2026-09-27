# E0 - Memoriser un lot

Hypothese enregistree avant execution : la perte sur un lot fixe de 16 exemples doit descendre sous 0,02 et le decodage doit retrouver presque tous les mots. Sinon, verifier alignement, masque, gradients et taux d'apprentissage.

Configuration : configs/overfit.yaml. Ce test reutilise un lot du train. Il ne mesure jamais la generalisation.
