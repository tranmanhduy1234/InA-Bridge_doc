# InA-Bridge

InA-Bridge is a vision–language model that connects **DINOv3** to **Qwen3-4B** through a **BERT-initialized Q-Former** and a **visual projector**. It learns to select information from images and transform it into representations that the language model can use to generate descriptions and answer questions.

The project implements three tasks: image–text retrieval in Stage 1, image captioning in Stage 2, and visual question answering in Stage 3. Current Stage 3 evaluation results are **75.06% POPE F1**, **874.26/2000 MME Perception**, and **263.57/800 MME Reasoning**.

## Architecture

```text
Image → DINOv3 → Patch features → Q-Former → Visual projector → Qwen3-4B → Text
                                     ↑                          ↑
                            Instruction (Stage 3)      Prompt / instruction
```

| Component | Structure and role |
| --- | --- |
| Vision encoder | `facebook/dinov3-vitl16-pretrain-lvd1689m`; extracts image patch features and remains frozen throughout all three stages |
| Q-Former | Initialized from `bert-base-uncased`, with 64 learnable queries and a hidden size of 768; cross-attends to image features every two blocks |
| Visual projector | RMSNorm, a gated SiLU branch, and a linear residual branch; maps queries to Qwen's embedding dimension |
| Language model | `Qwen/Qwen3-4B`; receives the visual prefix and text through `inputs_embeds` and is adapted with LoRA in Stage 2 |

### DINOv3: image representation

The vision encoder retains patch tokens and removes CLS and register tokens. With the configured **640 × 640** image resolution and a patch size of **16**, each image produces **1600 patch tokens**. These features serve as input to Q-Former's cross-attention layers.

DINOv3 remains frozen; image–language alignment is learned by the downstream components.

### Q-Former: selecting visual information

Q-Former uses encoder layers initialized from BERT. Each block applies self-attention to queries and text, when present, followed by query-to-image cross-attention in selected blocks. Queries and text have separate feed-forward networks (FFNs): the query FFN starts as a copy of BERT's FFN and is then trained independently. Query tokens and cross-attention layers are newly initialized.

The attention mask depends on the training objective:

| Objective | Query–text interaction |
| --- | --- |
| ITC — Image–Text Contrastive | The two branches remain separate to learn image and text representations that can be matched |
| ITM — Image–Text Matching | The two branches attend to each other bidirectionally to assess an image–text pair |
| ITG — Image-grounded Text Generation | Text attends to queries and preceding text tokens; queries do not attend to text |

In Stage 2, Q-Former receives only image features and learnable queries. In Stage 3, the instruction is encoded using BERT embeddings and passed to Q-Former with an ITM mask, allowing queries to select image information based on the question. The answer is never passed to Q-Former.

### Projector and Qwen3-4B: from vision to language

The projector preserves all 64 queries while mapping their feature dimension from 768 to Qwen's embedding dimension:

```text
x      = RMSNorm(query_features)
h      = SiLU(gate_proj(x)) × up_proj(x)
visual = skip(x) + scale × down_proj(Dropout(h))
```

All projections are bias-free linear layers; `scale` is a learnable parameter initialized to `0.1`. The projector produces **64 visual prefix tokens**, which are prepended to Qwen's text embeddings. Qwen generates text autoregressively.

Tensor flow, where `B` is the batch size, `Dv` is the DINOv3 feature dimension, `Dl` is Qwen's embedding dimension, and `T` is the number of text tokens:

```text
Image           [B, 3, 640, 640]
Patch features  [B, 1600, Dv]
Visual queries  [B, 64, 768]
Visual prefix   [B, 64, Dl]
Qwen input      [B, 64 + T, Dl]
```

BERT and Qwen use separate tokenizers and embedding tables. The visual prefix enters Qwen directly as vectors; the projector maps visual features rather than translating between the two tokenizers' vocabularies.

## Three training stages

| Stage | Objective | Updated components |
| --- | --- | --- |
| Stage 1 — Image–text alignment | Jointly learn ITC, ITM, and ITG from image–caption pairs | Q-Former, query tokens, ITC/ITM heads, and the decoding start embedding |
| Stage 2 — Image captioning | Predict captions from the visual prefix | Projector → LoRA → projector and LoRA jointly |
| Stage 3 — Instruction tuning | Answer instructions using image content | Q-Former and projector; Qwen and its LoRA adapters remain frozen |

Stage 1 uses an EMA teacher and a MoCo queue for ITC, treating multiple captions of the same image as positives. ITM learns to distinguish matching pairs from hard negatives, while ITG learns image-conditioned text generation. The vision encoder and BERT embeddings remain frozen, and the ITG language modeling head shares weights with BERT's word embeddings.

Stage 2 freezes Q-Former and learns caption generation using next-token cross-entropy. LoRA uses rank 16, alpha 32, and dropout 0.05; Qwen's base weights remain frozen. Stage 3 feeds the instruction into both Q-Former and the Qwen prompt, computing loss only on the answer and EOS.

## Evaluation results

The following results come from run **`stage3-eval-001`**, using the Stage 3 checkpoint at **5000 optimizer steps**, snapshot `e2793a45c15d4bab808a601427380cd1`. Scoring reports were generated on November 11–12, 2026 (UTC).

Evaluation uses **VLMEvalKit** at revision `c247c4d61d92be5643802aa9be5f2bc34d913d2a`, the `exact_matching` evaluator, each dataset's default prompt, and the checkpoint's chat template. The run used `do_sample=False`, a maximum of 512 new tokens, and seed 42.

| Benchmark | Metric | Result |
| --- | --- | ---: |
| POPE | F1 | **75.06%** |
| POPE | Accuracy | 70.06% |
| POPE | Precision | 64.31% |
| POPE | Recall | 90.13% |
| MME | Perception | **874.26 / 2000** |
| MME | Reasoning | **263.57 / 800** |

Inference completed for **5127/5127 POPE samples** and **2374/2374 MME samples** in the datasets loaded by the toolkit, with 100% coverage and no subset limit. MME scores are aggregate scores, not accuracy percentages.

POPE indicates that the model identifies most objects that are present but still makes many false positive claims: recall is 90.13%, while precision is 64.31%. Accuracy drops from 77.70% on the random split to 63.03% on the adversarial split. On MME, object existence (158.33/200) and scene recognition (141.50/200) score higher than position (51.67/200), color (68.33/200), and numerical calculation (50/200).

Data sources: `src/Evaluation/output/stage3-eval-001/summary.json` and `run.json` in the project copy used to compile this document. These results apply to the checkpoint identified above; no baseline evaluated under the same protocol is available to establish an improvement. MMBench_DEV_EN has no valid score in this report because the original evaluation run failed input validation.

## Scope and limitations

InA-Bridge is intended for experimentation with image captioning, single-image question answering, and research into connecting vision encoders with language models. It may describe objects that are not present, misinterpret details, or reason incorrectly. The results above do not establish performance across all languages, data domains, or question types; the capabilities of the original Qwen3 model should not be taken as verified capabilities of InA-Bridge.

## Licenses and acknowledgments

InA-Bridge uses foundation models developed by the teams listed below. Rights to the original components belong to their respective owners. Model names are used for attribution and do not imply that these teams sponsor or endorse InA-Bridge.

| Component | Source and developer | License |
| --- | --- | --- |
| DINOv3 ViT-L/16 | [Meta — facebook/dinov3-vitl16-pretrain-lvd1689m](https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m) | [DINOv3 License](https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m/blob/main/LICENSE.md) |
| Qwen3-4B | [Qwen Team, Alibaba — Qwen/Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B) | [Apache License 2.0](https://huggingface.co/Qwen/Qwen3-4B/blob/main/LICENSE) |
| BERT base uncased | [Google — google-bert/bert-base-uncased](https://huggingface.co/google-bert/bert-base-uncased) | [Apache License 2.0](https://raw.githubusercontent.com/google-research/bert/master/LICENSE) |

**DINOv3 has its own license.** Distribution of DINO Materials or derivative works covered by that license must comply with its terms and include a copy of the license. Published research using DINOv3 must acknowledge its use; the use restrictions in Section 1(b) continue to apply. See [DINOv3 License, Section 1](https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m/blob/main/LICENSE.md).

**For Apache 2.0 components**, redistribution must include the license, retain relevant rights notices, identify modified files, and preserve NOTICE content where supplied with the original work. See [Apache License 2.0, Section 4](https://www.apache.org/licenses/LICENSE-2.0).

InA-Bridge's modifications include the BERT-based Q-Former adaptations, visual projector, Qwen LoRA adaptation, and three-stage training pipeline. This document does not grant a new license for the original components. A separate license for InA-Bridge's own code and weights has not been identified in the reviewed project copy; the complete model should not be assumed to be released under Apache 2.0.

Acknowledgments and citations do not replace license compliance. When releasing checkpoints or providing inference services, the terms applicable to the components actually supplied must be followed; rights to training data and input content are separate matters. The links above provide the full terms; this section is only a summary.

## References

We acknowledge the following foundational works:

1. Oriane Siméoni et al. (2025). **DINOv3**. [arXiv:2508.10104](https://arxiv.org/abs/2508.10104).
2. An Yang et al. (2025). **Qwen3 Technical Report**. [arXiv:2505.09388](https://arxiv.org/abs/2505.09388).
3. Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova (2019). **BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding**. NAACL-HLT. [Paper](https://aclanthology.org/N19-1423/).
4. Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi (2023). **BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models**. Reference work for the Q-Former architecture. [arXiv:2301.12597](https://arxiv.org/abs/2301.12597).

Source code and technical documentation: [InA-Bridge on GitHub](https://github.com/tranmanhduy1234/InABridge).
