import json
from pathlib import Path

from reportlab.graphics.shapes import Drawing, Line, Rect, String
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def pct(value):
    return f"{100 * value:.2f} %"


def main():
    main_result = read("experiments/main/results_test.json")
    blind = read("experiments/E1_blind/results_test.json")
    heldout = read("experiments/main/results_test_heldout.json")
    analysis = read("experiments/main/results_test_analysis.json")
    overfit = read("experiments/E0_overfit/results.json")
    benchmark = read("benchmarks/S1_throughput/results.json")
    machine = main_result["hardware"]
    gpu = machine["gpu"] or machine["device"]
    cpu = machine["cpu"]
    by_count = analysis["by_object_count"]
    navy = colors.HexColor("#132b43")
    teal = colors.HexColor("#087f8c")
    grey = colors.HexColor("#eef3f7")
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle("BodyR", fontName="Helvetica", fontSize=9.3, leading=12.5, spaceAfter=7))
    styles.add(ParagraphStyle("SmallR", fontName="Helvetica", fontSize=7.8, leading=10, spaceAfter=5))
    styles.add(ParagraphStyle("TitleR", fontName="Helvetica-Bold", fontSize=25, leading=29, textColor=navy, spaceAfter=12))
    styles.add(ParagraphStyle("SectionR", fontName="Helvetica-Bold", fontSize=12.5, leading=16, textColor=teal, spaceBefore=9, spaceAfter=7))
    styles.add(ParagraphStyle("CaptionR", fontName="Helvetica", fontSize=8, leading=10, textColor=colors.HexColor("#46596a"), spaceAfter=8))
    styles.add(ParagraphStyle("CellR", fontName="Helvetica", fontSize=8, leading=10))
    story = []

    def paragraph(text, style="BodyR"):
        story.append(Paragraph(text, styles[style]))

    def section(text):
        paragraph(text, "SectionR")

    def table(rows, widths):
        cells = [[Paragraph(str(cell), styles["CellR"]) for cell in row] for row in rows]
        result = Table(cells, colWidths=widths, hAlign="LEFT")
        result.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), grey),
            ("TEXTCOLOR", (0, 0), (-1, 0), navy),
            ("LINEBELOW", (0, 0), (-1, 0), 0.7, teal),
            ("LINEBELOW", (0, 1), (-1, -1), 0.25, colors.HexColor("#d3dee7")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 7),
            ("RIGHTPADDING", (0, 0), (-1, -1), 7),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ]))
        story.append(result)
        story.append(Spacer(1, 8))

    paragraph("TINY VLM", "TitleR")
    paragraph("RLS Entrance Challenge | Rapport technique | Niveau 1 | 27 septembre 2026", "SmallR")
    section("1. Probleme")
    paragraph("Construire un modele qui observe une scene synthetique 64 x 64 et genere sa description en un mot, caractere par caractere. La question centrale est de distinguer la lecture de l'image de la simple prediction de l'orthographe. Les splits officiels ShapeScenes sont conserves : 20 000 / 2 000 / 2 000 / 2 000 exemples, seed 42.")
    section("2. Architecture et choix")
    diagram = Drawing(500, 177)
    labels = [
        ("Image RGB", "(B, 3, 64, 64)"),
        ("CNN : 3 Conv stride 2 + ReLU, puis AvgPool", "(B, 128, 4, 4)"),
        ("Flatten spatial + Linear adapter", "(B, 16, 128)"),
        ("Image + embeddings des lettres + positions", "(B, 16 + T, 128)"),
        ("2 blocs : attention 4 tetes + MLP", "(B, 16 + T, 128)"),
        ("Dernier visuel + lettres : LayerNorm + Linear", "(B, T + 1, 27)"),
    ]
    for index, (label, shape) in enumerate(labels):
        y = 151 - index * 29
        diagram.add(Rect(0, y, 500, 24, fillColor=grey if index % 2 == 0 else colors.white,
                         strokeColor=colors.HexColor("#cfdae3"), strokeWidth=0.5))
        diagram.add(String(9, y + 8, label, fontName="Helvetica", fontSize=8.5, fillColor=navy))
        diagram.add(String(368, y + 8, shape, fontName="Helvetica-Bold", fontSize=8.5, fillColor=teal))
        if index < len(labels) - 1:
            diagram.add(Line(250, y, 250, y - 5, strokeColor=teal))
    story.append(diagram)
    paragraph("Figure 1. Dimensions du pipeline. T est le nombre de caracteres fournis ; la premiere des T+1 sorties vient du dernier token visuel.", "CaptionR")
    paragraph("521 307 parametres, d_model=128, quatre tetes de 32 dimensions, MLP 128-512-128. Une grille 4 x 4 conserve plusieurs positions et limite le cout quadratique. Cette compacite sacrifie du detail ; aucune ablation ne prouve que ce choix est optimal. Le modele est plus petit que l'ordre de grandeur indicatif du brief.")
    section("3. Implementation et correction")
    paragraph("Attention manuelle : softmax(QK^T / sqrt(32) + masque)V, puis fusion des tetes et projection. Les 16 tokens visuels se voient tous, sans voir le texte. Chaque lettre voit l'image et son prefixe. Les positions apprises encodent l'ordre ; LayerNorm avant chaque sous-bloc et connexions residuelles facilitent l'optimisation.")
    paragraph("Vocabulaire : 26 lettres + eos. Le dernier token visuel predit la premiere lettre, sans bos. Teacher forcing : une entree r predit e, e predit d, d predit eos. Le padding des cibles vaut -100 et est ignore par CrossEntropyLoss. Les entrees completees valent 0, a droite ; elles ne peuvent influencer les positions utiles passees.")
    paragraph("108 tests passent : reference PyTorch de l'attention (tolerance usuelle 1e-5 en float32), causalite, gradients, alignement premiere lettre, padding, parser, arret eos et accord CPU/GPU. Les composants interdits ne sont pas utilises dans le modele.")
    story.append(PageBreak())

    section("4. Configuration d'entrainement")
    table([
        ["Element", "Valeur"],
        ["Donnees", "Generateur RLS inchange ; uint8 en stockage, float32 / 255 au chargement"],
        ["Architecture", "CNN 32/64/128 canaux, grille 4 x 4 ; 2 blocs, d=128, 4 tetes"],
        ["Optimisation", "AdamW, learning rate 0,001 ; weight decay 0,01 ; clipping norme 1"],
        ["Budget", "Batch 64 ; 10 epochs ; seed 42 ; float32 ; aucun preentrainement"],
        ["Selection", "Checkpoint de perte validation minimale ; test jamais utilise pour choisir"],
        ["Materiel", f"{gpu} ; {cpu} ; {machine['threads']} threads CPU ; {machine['os']}"],
        ["Logiciel", f"Python {machine['python']} ; PyTorch {machine['torch']} ; CUDA {machine['cuda']}"],
    ], [100, 400])
    section("5. Experiences et protocole")
    paragraph("<b>E0.</b> Hypothese avant execution : memoriser un batch de 16 exemples jusqu'a une loss inferieure a 0,02. Resultat apres 600 pas : loss " + f'{overfit["loss"]:.6f}' + f" nat/token et {pct(overfit['exact_match'])} de mots corrects sur le lot. Cela valide la capacite a apprendre ce lot, sans mesurer la generalisation.")
    paragraph("<b>E1.</b> Hypothese avant execution : les images reelles ameliorent exact-match et attributs par rapport aux pixels nuls. Seule variable modifiee : images mises a zero pendant entrainement et evaluation. Initialisation, ordre des lots, architecture, epochs, optimiseur et splits sont identiques. Une seed par condition : dispersion entre seeds non estimable.")
    plot = Table([[Image(str(ROOT / "experiments/main/loss.png"), width=247, height=144),
                   Image(str(ROOT / "experiments/E1_blind/loss.png"), width=247, height=144)]], colWidths=[250, 250])
    story.append(plot)
    paragraph(f"Figure 2. Loss train/validation en nats par token sur {gpu}. A gauche, modele visuel ; a droite, aveugle. Le plateau aveugle illustre l'information manquante sur la scene.", "CaptionR")
    paragraph("La perte est ponderee par le nombre de tokens non ignores. Les CSV conservent chaque epoch ; les checkpoints sauvegardent poids, optimiseur et historique. La reprise repart de la derniere epoch terminee, avec l'ordre des batches determine par seed + epoch.")
    paragraph("Le cours, les tests supplementaires et l'implementation ont ete prepares avec Codex, comme declare dans AI_USAGE.md. Le generateur et les tests officiels sont conserves. La comprehension du candidat doit etre demontree personnellement a l'entretien.")
    story.append(PageBreak())

    section("6. Resultats mesures")
    rows = [["Mesure", "Visuel test", "Aveugle test", "Visuel heldout"]]
    for name, key in [("Exact-match", "exact_match"), ("Taille", "size"), ("Couleur", "color"),
                      ("Forme", "shape"), ("Relation", "relation"), ("Nombre d'objets", "object_count"),
                      ("Lettres libres", "letter_accuracy"), ("Lettres, prefixe vrai", "teacher_forced_letter_accuracy")]:
        rows.append([name] + [pct(r["metrics"][key]) for r in (main_result, blind, heldout)])
    table(rows, [155, 115, 115, 115])
    paragraph(f"Tableau 1. Predictions des checkpoints retenus sur {gpu}, n=1 seed/condition. Pas d'ecart-type entre seeds estime. Test : 2 000 images, 3 020 objets, 1 020 relations ; heldout : 2 000 images, 2 950 objets, 950 relations.", "CaptionR")
    paragraph("Les attributs sont compares par emplacement, sans autocorrection : largeredcirle conserve taille et couleur, mais sa forme est fausse. Taille/couleur/forme sont agregees sur les objets de reference ; relation sur les scenes a deux objets. Le nombre d'objets est mesure separement.")
    paragraph("Les lettres libres sont comparees position par position, avec la longueur maximale au denominateur. La mesure avec prefixe vrai exclut eos et padding et utilise les bonnes lettres precedentes. Elle ne represente pas un decodage autonome.")
    section("7. Analyse et incertitude")
    paragraph("Le modele visuel atteint " + pct(main_result["metrics"]["exact_match"]) + " d'exact-match contre " + pct(blind["metrics"]["exact_match"]) + " pour E1. Ce contraste soutient l'utilisation de l'image dans ce run. Le baseline aveugle obtient pourtant " + pct(blind["metrics"]["teacher_forced_letter_accuracy"]) + " des lettres avec prefixe vrai : une grande part de la sequence s'explique par l'orthographe, sans vision.")
    paragraph(f"L'exact-match est de {pct(by_count['1']['exact_match'])} sur {by_count['1']['count']} scenes a un objet et {pct(by_count['2']['exact_match'])} sur {by_count['2']['count']} scenes a deux objets. Parmi " + str(analysis["strict_errors"]) + " erreurs strictes, " + str(analysis["errors_equivalent_after_reversing_objects"]) + " sont exactement equivalentes apres inversion des objets et de la relation. Exemple : A below B devient B above A. Le generateur ne fixe pas un ordre canonique ; le score officiel reste inchange.")
    paragraph(f"Les {analysis['other_errors']} autres erreurs montrent que l'ambiguite d'ordre n'explique pas tout. Les scenes a deux objets et les relations restent difficiles. Les confusions ordonnees de formes peuvent aussi inclure des permutations d'objets ; elles ne prouvent pas a elles seules une faiblesse du CNN.")
    paragraph("Heldout : aucun mot exact. Taille et couleur restent assez souvent reconnues, mais la forme chute a " + pct(heldout["metrics"]["shape"]) + ". Exemple : largeyellowcross devient largeyellowcircle. Hypothese exploratoire : associations couleur-forme apprises sans recombinaison fiable. Aucune intervention causale ni repetition sur trois seeds n'a ete faite ; niveau 2 non revendique.")
    story.append(PageBreak())

    section("8. Systeme : benchmark S1")
    rows = [["Batch", "CPU, images/s", "GPU, images/s"]]
    for batch in benchmark["config"]["batch_sizes"]:
        entries = [next(r for r in benchmark["summary"] if r["batch_size"] == batch and r["device"] == device) for device in ("cpu", "cuda")]
        rows.append([str(batch)] + [f'{r["mean_images_per_second"]:.2f} +/- {r["std_images_per_second"]:.2f}' for r in entries])
    table(rows, [80, 210, 210])
    paragraph("Tableau 2. Moyenne +/- ecart-type echantillon sur 5 repetitions de 10 pas, apres 5 pas de chauffe. CPU i5-10500H (4 threads), GPU GTX 1650. Float32, 16 tokens visuels + 45 caracteres. Donnees brutes : benchmarks/S1_throughput/raw.csv.", "CaptionR")
    story.append(Image(str(ROOT / "benchmarks/S1_throughput/throughput.png"), width=360, height=210))
    paragraph("Figure 3. Le GPU gagne pour les quatre batches mesures, sans croisement observe. Le gain est moindre a batch 1 et le debit GPU plafonne entre 64 et 256 dans ce protocole.", "CaptionR")
    paragraph("Les tenseurs sont synthetiques et deja sur le peripherique. Le timer inclut forward, loss, backward, clipping et AdamW ; il exclut DataLoader, transferts et ecritures. La chauffe exclut les initialisations ; synchronize avant et apres le timer attend le travail CUDA asynchrone. Les essais S1 sont separes des entrainements.")
    paragraph("L'augmentation du batch amortit les couts fixes et offre plus de parallelisme. Sans profiler, le plateau ne suffit pas a identifier un goulot precis. Une matrice d'attention a B x H x T^2 elements ; a B=64, H=4, T=61, elle occupe environ 3,63 Mio en float32, hors autres activations et etats d'optimiseur.")
    section("9. Limites et suite")
    paragraph("Une seule seed, scenes synthetiques, une architecture, parser strict et ordre des objets ambigu. Pas de profiling ni d'etude memoire de niveau 2. Prochaine experience : comparer grille 4 x 4 et un seul vecteur visuel, en fixant le reste et en repetant sur trois seeds. Verifier aussi la recombinaison heldout. L'assistance IA est declaree ; l'aisance orale reste a travailler.")
    section("References")
    paragraph("RLS, Project Brief et template c21f9da : github.com/RLS-ResearchLab/RLS-Entrance-Challenge.<br/>Vaswani et al., Attention Is All You Need, 2017 : arxiv.org/abs/1706.03762.<br/>PyTorch 2.8, scaled_dot_product_attention et CrossEntropyLoss : docs.pytorch.org/docs/2.8/ (liens precis dans README).", "SmallR")

    def footer(canvas, document):
        canvas.setStrokeColor(colors.HexColor("#d3dee7"))
        canvas.line(42, 35, 553, 35)
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(navy)
        canvas.drawString(42, 23, "RLS - Tiny VLM | Mesures locales, seed 42")
        canvas.drawRightString(553, 23, str(document.page))

    document = SimpleDocTemplate(str(ROOT / "report/report.pdf"), pagesize=(210 * mm, 297 * mm),
                                 leftMargin=42, rightMargin=42, topMargin=32, bottomMargin=45,
                                 title="Tiny VLM - Rapport technique RLS", author="Projet RLS - assistance IA declaree")
    document.build(story, onFirstPage=footer, onLaterPages=footer)
    print(ROOT / "report/report.pdf")


if __name__ == "__main__":
    main()
