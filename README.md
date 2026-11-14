# French Word Sense Classification with LLMs: In-Context Learning vs. QLoRA

Classifying French Wiktionary noun senses into coarse semantic categories (**hypersenses**) using **Qwen2.5-7B-Instruct (4-bit)**, comparing **in-context learning (ICL)** with **parameter-efficient fine-tuning (QLoRA)**.

Each word sense is represented by its **lemma** and its **definition**, and must be assigned one of 12 hypersenses (e.g. `animate`, `inanimate`, `dynamic_situation`, `information`, …).

```
Lemma: béruguette
Definition: Variété d'olive allongée à pulpe ferme.
→ inanimate   (supersense: food)
```

## Key results

Final evaluation on the held-out test sets, each method using its best configuration on dev:

| Method | freq-test Acc. | freq-test Macro-F1 | rand-test Acc. | rand-test Macro-F1 |
|---|---|---|---|---|
| ICL (k = 24, label scoring) | 53.2 | 45.4 | 69.2 | 48.6 |
| **QLoRA (classification head)** | **70.0** | **60.0** | **84.5** | **63.7** |

- **Fine-tuning beats ICL by ~15 points** of accuracy and macro-F1 on both test sets, with the same 4-bit model.
- The gain comes mostly from **abstract senses** (situations, information), which are hard to convey with a label list and a few examples.
- **ICL is very sensitive to which demonstrations are shown**: with the label distribution held constant, three demonstration seeds gave 68.0%, 38.3% and 61.3% accuracy on dev.
- **Random demonstration labels hurt a lot** (−8 pts accuracy), more than the marginal drop reported by Min et al. (2022).
- `other`, the combined classes (`*information`) and abstract senses remain the main error sources for both methods. Part of this reflects fuzzy class boundaries in the gold annotation itself.

## Data

Manually labelled French Wiktionary senses from [Angleraud et al. (2025)](https://aclanthology.org/2025.coling-main.356/). Each sense carries one of 25 **supersenses** (taxonomy from [Barque et al., 2020](https://aclanthology.org/2020.lrec-1.724/)), mapped deterministically to 12 **hypersenses**.

| Split | Size | Use |
|---|---|---|
| `train` | 10,604 | Demonstrations (ICL) / training data (QLoRA) |
| `rand-dev` | 300 of 570 (fixed random subset) | All tuning decisions |
| `freq-test` | 1,521 | Final evaluation only |
| `rand-test` | 678 | Final evaluation only |

`protect-frsemcor` splits are discarded. Label distributions differ substantially across splits and some classes are rare (~1% of training data), so **macro-F1 is reported alongside accuracy**.

The notebook downloads the dataset automatically.

## Methods

### Model
[`unsloth/Qwen2.5-7B-Instruct-bnb-4bit`](https://huggingface.co/unsloth/Qwen2.5-7B-Instruct-bnb-4bit): instruction-tuned, ≤ 8B parameters, multilingual (supports French), pre-quantized to NF4 so it fits on a free Colab T4.

### 1. In-context learning
- **Prompt:** chat template with a system instruction listing the 12 labels, each optionally described by the supersenses it groups (auto-derived from the mapping), followed by *k* demonstrations and the query.
- **Demonstrations:** stratified sampling from `train` so every hypersense appears equally often; shuffled.
- **Inference:** log-probability **label scoring** over all 12 full (multi-token) labels, with the end-of-turn token appended so prefix labels (`inanimate` vs `inanimate*information`) aren't favoured. Never produces invalid outputs.
- **KV-cache reuse:** the shared prefix (instruction + demos) is encoded once per run and its cache cropped back after each item, making label scoring fast enough for Colab.
- **Ablations** (one factor at a time vs. baseline): no lemma, *k* = 0 / 24, no label descriptions, French instruction, free generation, random demonstration labels, plus a 3-seed variance study.

### 2. QLoRA fine-tuning
- 1 epoch on the full training set, LoRA rank 8, α = 16, dropout 0.05, lr 2e-4, effective batch size 16, max length 192.
- Short prompt (lemma + definition only, no system instruction).
- **Configurations compared:**

| Config | Task type | LoRA targets | Lemma | Dev Acc. | Dev Macro-F1 |
|---|---|---|---|---|---|
| `cls_base` | Sequence classification head | q, v | ✓ | 82.3 | **60.2** |
| `causal_wide_lora` | Causal LM (generate label) | q, k, v, o | ✓ | 83.0 | 59.7 |
| `causal_base` | Causal LM (generate label) | q, v | ✓ | 82.3 | 59.4 |
| `causal_no_lemma` | Causal LM (generate label) | q, v | ✗ | 82.0 | 57.4 |

Differences between QLoRA configs are small (within a few dev items); `cls_base` was selected on dev macro-F1 and because it cannot produce invalid outputs.

### Error analysis
Per-class F1, confusion matrices, accuracy per supersense, accuracy by definition length / multiword lemma / "Synonyme de…" definitions, and ICL-vs-QLoRA agreement.

## Repository contents

```
├── Word_classification.ipynb                 # Full pipeline: data, ICL, QLoRA, evaluation, error analysis
├── Classifying_word_sense_definitions.pdf    # Report with methodology, results and discussion
└── README.md
```

## Running it

The notebook is designed for **Google Colab with a T4 GPU**.

1. Open `Word_classification.ipynb` in Colab and select a GPU runtime.
2. Install dependencies once:
   ```bash
   pip install -q --upgrade transformers accelerate bitsandbytes trl peft datasets
   ```
3. Run the cells in order. Google Drive is mounted to persist predictions, metrics and LoRA adapters (`MyDrive/lab1_icl`, `MyDrive/lab1_peft`), so long experiments survive Colab disconnects and finished runs are skipped on re-execution.

Approximate cost: ~2.4 s/item for ICL label scoring with *k* = 12; ~45 min per QLoRA training run on a T4.

### T4 compatibility notes
- The pre-quantized checkpoint ships with a `bfloat16` compute dtype, which the T4 doesn't support, and `transformers` ignores a passed `quantization_config` for pre-quantized checkpoints. The notebook patches the loaded `AutoConfig` to `float16` instead. (Using a fresh `BitsAndBytesConfig` silently switched NF4 → FP4 and pushed training from ~45 min to 4+ hours.)
- `fp16=False, bf16=False` in the trainer to avoid the GradScaler *"Attempting to unscale FP16 gradients"* error.
- For the classification head, `lm_head` and `score` are added to `llm_int8_skip_modules` so the new head isn't quantized.

## Authors

Sara Graça and Manuela Pacheco

## References

- Angleraud, Barque & Candito (2025). *Annotating the French Wiktionary with supersenses for large scale lexical analysis.* COLING 2025.
- Barque et al. (2020). *FrSemCor: Annotating a French Corpus with Supersenses.* LREC 2020.
- Min et al. (2022). *Rethinking the Role of Demonstrations: What Makes In-Context Learning Work?* EMNLP 2022.
- Zhao et al. (2021). *Calibrate Before Use: Improving Few-Shot Performance of Language Models.* ICML 2021.
- Ahuja et al. (2023). *MEGA: Multilingual Evaluation of Generative AI.* EMNLP 2023.
- Schulhoff et al. (2024). *The Prompt Report: A Systematic Survey of Prompt Engineering Techniques.* arXiv:2406.06608.
- Qwen Team (2024). *Qwen2.5-7B-Instruct* model card.
