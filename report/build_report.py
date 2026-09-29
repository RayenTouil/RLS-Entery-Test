# FR : Reconstruit le rapport à partir des résultats enregistrés.
# EN: Rebuilds the report from saved results.

import json
from pathlib import Path

from reportlab.graphics.shapes import Drawing, Line, Rect, String
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle, StyleSheet1
from reportlab.lib.units import mm
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


ROOT = Path(__file__).resolve().parents[1]


# FR : Les mesures viennent des fichiers JSON, pas de valeurs inventées dans le rapport.
# EN: Measurements come from JSON files, not values invented in the report.
def load_results(path):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def as_percent(value):
    return f"{100 * value:.2f}%"


def main():
    visual = load_results("experiments/main/results_test.json")
    blind = load_results("experiments/E1_blind/results_test.json")
    heldout = load_results("experiments/main/results_test_heldout.json")
    errors = load_results("experiments/main/results_test_analysis.json")
    overfit = load_results("experiments/E0_overfit/results.json")
    timing = load_results("benchmarks/S1_throughput/results.json")
    config = visual["config"]
    hardware = visual["hardware"]
    groups = errors["by_object_count"]
    gpu = hardware["gpu"] or hardware["device"]

    # FR : Quelques styles partagés gardent le même aspect sur les quatre pages.
    # EN: A few shared styles keep the four pages consistent.
    ink = colors.HexColor("#19334b")
    accent = colors.HexColor("#126b73")
    pale = colors.HexColor("#edf2f5")
    styles = StyleSheet1()
    for name, size, leading in [("body", 10, 14), ("small", 8, 11), ("cell", 8.5, 11),
                                ("caption", 8.5, 11), ("heading", 13, 17), ("title", 25, 30)]:
        styles.add(ParagraphStyle(
            name, fontName="Helvetica-Bold" if name in ("title", "heading") else "Helvetica",
            fontSize=size, leading=leading, spaceAfter=8,
            spaceBefore=8 if name == "heading" else 0,
            textColor=accent if name == "heading" else ink,
        ))
    story = []

    def add_text(text, style="body"):
        story.append(Paragraph(text, styles[style]))

    def add_heading(text):
        add_text(text, "heading")

    def add_table(rows, widths):
        cells = [[Paragraph(str(cell), styles["cell"]) for cell in row] for row in rows]
        table = Table(cells, colWidths=widths, hAlign="LEFT")
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), pale),
            ("LINEBELOW", (0, 0), (-1, 0), 0.7, accent),
            ("LINEBELOW", (0, 1), (-1, -1), 0.25, colors.HexColor("#cbd6df")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 7),
            ("RIGHTPADDING", (0, 0), (-1, -1), 7),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ]))
        story.extend([table, Spacer(1, 7)])

    # FR : Page 1 - problème, architecture et détails nécessaires pour comprendre le code.
    # EN: Page 1 - problem, architecture and the details needed to understand the code.
    add_text("Tiny Vision-Language Model", "title")
    add_text("Rayen Touil | RLS Entrance Challenge | Level 1", "small")
    add_heading("1. The task")
    add_text("The model sees a 64 x 64 image of one or two shapes and writes a description, "
             "one letter at a time. For example, a large red circle becomes <b>largeredcircle</b>. "
             "The main question is whether it uses the image or only learns how the words are spelled.")
    add_heading("2. Architecture")
    diagram = Drawing(500, 177)
    steps = [
        ("RGB image", "(B, 3, 64, 64)"),
        ("3 convolutions + ReLU, then average pooling", "(B, 128, 4, 4)"),
        ("Flatten the grid + linear adapter", "(B, 16, 128)"),
        ("Visual tokens + letters + learned positions", "(B, 16 + T, 128)"),
        ("2 decoder blocks, 4 attention heads", "(B, 16 + T, 128)"),
        ("Last visual token + letters -> output head", "(B, T + 1, 27)"),
    ]
    for index, (label, shape) in enumerate(steps):
        y = 151 - index * 29
        diagram.add(Rect(0, y, 500, 24, fillColor=pale if index % 2 == 0 else colors.white,
                         strokeColor=colors.HexColor("#cbd6df"), strokeWidth=0.5))
        diagram.add(String(9, y + 8, label, fontName="Helvetica", fontSize=8.4, fillColor=ink))
        diagram.add(String(370, y + 8, shape, fontName="Helvetica-Bold", fontSize=8.4, fillColor=accent))
        if index < len(steps) - 1:
            diagram.add(Line(250, y, 250, y - 5, strokeColor=accent))
    story.append(diagram)
    add_text("Figure 1. B is the batch size and T is the number of input letters. "
             "The last visual token predicts the first letter.", "caption")
    add_text("The model has <b>521,307 parameters</b>. The 4 x 4 grid keeps spatial information "
             "with only 16 visual tokens. Each token has 128 values; each of the four heads uses 32. "
             "The MLP expands 128 to 512, then returns to 128. This is a small baseline, "
             "not a claim that these settings are the best.")
    add_heading("3. Implementation")
    add_text("Attention is written by hand: <b>softmax(QK^T / sqrt(32) + mask)V</b>. "
             "Visual tokens can read each other, but cannot read the letters. A letter can read "
             "the image and its current prefix. Learned positions give the tokens an order. "
             "Each decoder block uses LayerNorm before attention and the MLP, with residual additions.")
    add_text("The vocabulary has 26 letters and eos. No start token is needed. During training, "
             "the correct previous letters are supplied (teacher forcing). Padded targets use -100 "
             "and are ignored by the loss. During generation, the model uses its own predictions "
             "and stops at eos or 45 letters.")
    add_text("The attention is checked against PyTorch's reference at about 1e-5 tolerance in float32. "
             "Other tests check gradients, future-token masking, target alignment, padding and generation. "
             "No pretrained model or built-in Transformer is used.")
    story.append(PageBreak())

    # FR : Page 2 - réglages et protocole, avec les courbes réellement enregistrées.
    # EN: Page 2 - settings and protocol, with the actual saved curves.
    add_heading("4. Training setup")
    add_table([
        ["Setting", "Value"],
        ["Data", "Official generator, unchanged; seed 42. Train 20,000; val, test and heldout 2,000 each."],
        ["Input", "RGB pixels stored as uint8, converted to float32 and divided by 255."],
        ["Training", f"Batch {config['batch_size']}; {config['epochs']} epochs; seed {config['seed']}."],
        ["Optimizer", f"AdamW; learning rate {config['learning_rate']}; weight decay {config['weight_decay']}; gradient norm clipped to 1."],
        ["Checkpoint", "Lowest validation loss. The test split is not used to select the model."],
        ["Hardware", f"{gpu}; {hardware['cpu']}; {hardware['threads']} CPU threads."],
        ["Software", f"Python {hardware['python']}; PyTorch {hardware['torch']}; CUDA {hardware['cuda']}; Windows 11."],
    ], [90, 410])
    add_heading("5. Experiments")
    add_text("<b>E0 - Overfit one batch.</b> The prediction before running was that the model should "
             "learn a fixed batch of 16 examples and reach a loss below 0.02. After 600 steps, "
             f"the loss was <b>{overfit['loss']:.6f} nats/token</b> and exact match was "
             f"<b>{as_percent(overfit['exact_match'])}</b>. This checks that learning works on that batch; "
             "it does not measure performance on new images.")
    add_text("<b>E1 - Blind baseline.</b> The prediction was that real images would give better "
             "word and attribute accuracy than zero images. Both models use the same settings, "
             "seed, initialization and batch order. The blind model receives zero images in both "
             "training and evaluation. Only one seed was run for each setup.")
    story.append(Table([[
        Image(str(ROOT / "experiments/main/loss.png"), width=247, height=144),
        Image(str(ROOT / "experiments/E1_blind/loss.png"), width=247, height=144),
    ]], colWidths=[250, 250]))
    add_text(f"Figure 2. Training and validation loss on {gpu}. Left: visual model. "
             "Right: blind model. The blind loss stops improving much earlier. Note the different vertical scales.",
             "caption")
    add_text("Loss is averaged over real target tokens, with padding excluded. "
             "CSV files keep the values behind the plots. Checkpoints save model weights, "
             "optimizer state and progress so training can resume after a completed epoch.")
    story.append(PageBreak())

    # FR : Page 3 - les pourcentages sont lus dans les résultats de l’évaluation.
    # EN: Page 3 - percentages are read from the evaluation results.
    add_heading("6. Results")
    rows = [["Metric", "Visual / test", "Blind / test", "Visual / heldout"]]
    for label, key in [
        ("Exact match", "exact_match"), ("Size", "size"), ("Color", "color"),
        ("Shape", "shape"), ("Relation", "relation"), ("Object count", "object_count"),
        ("Letters, free generation", "letter_accuracy"),
        ("Letters, true prefix", "teacher_forced_letter_accuracy"),
    ]:
        rows.append([label] + [as_percent(result["metrics"][key]) for result in (visual, blind, heldout)])
    add_table(rows, [155, 115, 115, 115])
    add_text(f"Table 1. Results on {gpu}, one training seed per setup. "
             "No standard deviation across seeds is available. Test has 2,000 images, 3,020 objects "
             "and 1,020 relations; heldout has 2,000 images, 2,950 objects and 950 relations.", "caption")
    add_text("The parser keeps correct parts of a broken word. For <b>largeredcirle</b>, size and "
             "color are correct, but shape is wrong. It does not fix spelling. Attributes are scored "
             "at their reference object position; relations are scored only for two-object scenes. "
             "Object count is inferred from size-word boundaries, so malformed outputs can confuse it.")
    add_text("Free-generation letter accuracy compares matching positions and divides by the longer "
             "word length. True-prefix accuracy supplies the correct earlier letters and excludes "
             "eos and padding. These are different tests.")
    add_heading("7. What the results show")
    add_text(f"The visual model reaches <b>{as_percent(visual['metrics']['exact_match'])}</b> exact match, "
             f"against <b>{as_percent(blind['metrics']['exact_match'])}</b> for the blind model. "
             "This supports the use of image information in this run. However, the blind model gets "
             f"<b>{as_percent(blind['metrics']['teacher_forced_letter_accuracy'])}</b> of letters right "
             "when given the true prefix. Much of the spelling is predictable without seeing the image.")
    add_text(f"Exact match is {as_percent(groups['1']['exact_match'])} on {groups['1']['count']} one-object scenes "
             f"and {as_percent(groups['2']['exact_match'])} on {groups['2']['count']} two-object scenes. "
             f"Of {errors['strict_errors']} strict errors, {errors['errors_equivalent_after_reversing_objects']} "
             "name the same objects in reverse order with the inverse relation: A below B becomes B above A. "
             "The generator does not set a fixed object order. These remain errors in the official metric.")
    add_text(f"The remaining {errors['other_errors']} errors show that word order does not explain everything. "
             f"Heldout exact match is {as_percent(heldout['metrics']['exact_match'])}; shape accuracy falls to "
             f"{as_percent(heldout['metrics']['shape'])}. For example, largeyellowcross becomes largeyellowcircle. "
             "A possible explanation is that the model learned common color-shape pairs, but did not learn "
             "to combine them freely. This needs another experiment to test.")
    story.append(PageBreak())

    # FR : Page 4 - débits, limites et sources ; les répétitions ne sont pas des seeds.
    # EN: Page 4 - speed, limits and sources; timing repeats are not training seeds.
    add_heading("8. CPU and GPU timing")
    rows = [["Batch", "CPU images/s", "GPU images/s"]]
    for batch in timing["config"]["batch_sizes"]:
        measured = [next(row for row in timing["summary"]
                         if row["batch_size"] == batch and row["device"] == device)
                    for device in ("cpu", "cuda")]
        rows.append([batch] + [
            f"{row['mean_images_per_second']:.2f} +/- {row['std_images_per_second']:.2f}" for row in measured
        ])
    add_table(rows, [80, 210, 210])
    add_text("Table 2. Mean +/- sample standard deviation across five repeats of ten training steps, "
             "after five warm-up steps. Intel i5-10500H, four threads; NVIDIA GTX 1650. "
             "Float32, 16 visual tokens plus 45 input letters.", "caption")
    story.append(Image(str(ROOT / "benchmarks/S1_throughput/throughput.png"), width=340, height=198))
    add_text("Figure 3. The GPU is faster at all four tested batch sizes. Its advantage is smaller "
             "at batch 1, and throughput changes little between 64 and 256.", "caption")
    add_text("The timer includes forward, loss, backward, gradient clipping and AdamW. "
             "It excludes data loading, transfers and file writes: synthetic tensors are already on the device. "
             "Warm-up removes initial setup costs. CUDA synchronization before and after timing "
             "waits for the asynchronous GPU work. S1 ran without another training job.")
    add_text("Larger batches can spread fixed costs across more images and provide more parallel work. "
             "No CPU/GPU crossover was observed here. A profiler would be needed to explain the exact "
             "bottleneck. Attention scores use B x H x T squared values, so longer sequences also cost memory.")
    add_heading("9. Limits and next step")
    add_text("This is one small model with one seed on synthetic images. The parser and object-order "
             "ambiguity affect the metrics. No detailed profiling, memory study or accuracy-tuning search "
             "was run. The next useful step is to test why unseen color-shape pairs fail. "
             "The heldout check is exploratory; this report does not claim Level 2.")
    add_heading("References")
    add_text("RLS: github.com/RLS-ResearchLab/RLS-Entrance-Challenge, template c21f9da.<br/>"
             "Vaswani et al., Attention Is All You Need (2017): arxiv.org/abs/1706.03762.<br/>"
             "PyTorch 2.8: scaled_dot_product_attention and CrossEntropyLoss; direct links in SOURCE.md.<br/>"
             "Learning and coding assistance are described in AI_USAGE.md.", "small")

    # FR : Numérote les pages ; les sauts ci-dessus fixent les quatre parties du rapport.
    # EN: Numbers the pages; the breaks above keep the report in four parts.
    def footer(canvas, document):
        canvas.setStrokeColor(colors.HexColor("#cbd6df"))
        canvas.line(42, 35, 553, 35)
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(ink)
        canvas.drawString(42, 23, "RLS Entry Test | Tiny VLM")
        canvas.drawRightString(553, 23, str(document.page))

    document = SimpleDocTemplate(
        str(ROOT / "report/report.pdf"), pagesize=(210 * mm, 297 * mm),
        leftMargin=42, rightMargin=42, topMargin=32, bottomMargin=45,
        title="Tiny VLM - RLS Entry Test", author="Rayen Touil",
    )
    document.build(story, onFirstPage=footer, onLaterPages=footer)
    print(ROOT / "report/report.pdf")


if __name__ == "__main__":
    main()
