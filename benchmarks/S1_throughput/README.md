# S1 - Débit d'entraînement

Commande : python -m benchmarks.S1_throughput.benchmark --config configs/benchmark.yaml --devices cpu cuda

Quatre tailles de batch : 1, 16, 64, 256. Cinq pas de chauffe, puis cinq répétitions de dix pas. La moyenne et l'écart-type échantillon portent sur les cinq débits. La seed est fixe.

Les tenseurs synthétiques sont créés et placés sur le périphérique avant le chronométrage. Chaque pas inclut zero_grad, CNN, décodeur, loss, backward, clipping et AdamW. Le chargement de données, la conversion des images, le transfert CPU/GPU et les écritures de fichiers sont exclus.

Les entrées contiennent 45 caractères, donc 61 positions avec les 16 tokens visuels. Il s'agit d'un cas de longueur maximale, pas d'une moyenne sur les vraies longueurs du dataset.

CUDA est synchronisé juste avant et après chaque répétition. Le programme enregistre le CPU, le GPU, les versions, le nombre de threads et les temps bruts. Exécuter S1 sans autre entraînement concurrent. Les résultats décrivent cette machine et ce protocole, pas toutes les configurations.

raw.csv contient les mesures brutes, results.json les agrégats, hardware.txt le matériel et throughput.png le graphique.
