# VEGA

**Beyond Feature Alignment: Heterogeneous Visual Evidence Matching for Remote Sensing Image–Text Retrieval**

VEGA (Visual Evidence-Guided Alignment) is a remote sensing image–text retrieval framework that matches different textual semantics with different types of visual evidence. Instead of aligning every semantic level with one shared visual representation, VEGA explicitly models scene layout, local entities, and inter-object composition.

## Highlights

- **FG-SGE**: Frequency-Guided Structural Graph Enhancement decomposes visual features with a one-level Haar discrete wavelet transform and constructs structured evidence from low-/high-frequency cues.
- **F-SAMGA**: Frequency-Guided Semantic-Aware Multi-Granularity Alignment separates text into scene, entity, and composition semantics with learnable queries.
- **Multi-relation reasoning**: Spatial adjacency, semantic similarity, scene co-occurrence, and frequency compatibility are modeled with a dynamic graph.
- **Fine-grained matching**: Cross-modal attention and entropy-regularized optimal transport establish detailed visual–textual correspondences.
- **Efficient adaptation**: The CLIP ViT-B/32 backbone is frozen and only the newly introduced modules are optimized.

## Framework

Given an image–caption pair, VEGA follows this pipeline:

1. A CLIP ViT-B/32 backbone extracts global and patch-level features.
2. **FG-SGE** applies Haar DWT to produce `LL`, `LH`, `HL`, and `HH` subbands. Low-frequency evidence emphasizes global layouts, while high-frequency evidence captures boundaries, textures, and small objects.
3. A dynamic graph enriches visual nodes through four complementary relation types: `adj`, `sem`, `co`, and `freq`.
4. **F-SAMGA** extracts scene-, entity-, and composition-level text semantics and aligns them with layout, detail, and relational visual evidence, respectively.
5. The final bidirectional retrieval score fuses the scene, entity, and composition matching scores.

The training objective is:

```text
L = L_CLIP + L_scene + L_entity + L_composition
```

## Repository contents

```text
VEGA/
├── AVE_FGSGE_DWT4.py   # FG-SGE visual evidence construction
├── F_SAMGA.py          # F-SAMGA semantic-specific alignment
├── run.py              # training and retrieval entry point
└── README.md
```

## Datasets

The paper evaluates VEGA on:

- **RSICD**: 10,921 images, 30 scene categories, five captions per image.
- **RSITMD**: 4,743 images, 32 scene categories, five captions per image.
- **UCM-Captions**: 2,100 images, 21 land-use categories, five captions per image.

## Citation

If you find this work useful, please cite the corresponding paper:

```bibtex
@article{vega2026,
  title   = {Beyond Feature Alignment: Heterogeneous Visual Evidence Matching for Remote Sensing Image-Text Retrieval},
  journal = {Pattern Recognition},
  year    = {2026}
}
```

## License

This repository does not currently include a separate license file. Please add an appropriate license before redistributing the code.
