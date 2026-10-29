# Attribute baseline smoke test — 16 October 2026

- Model: openai/clip-vit-base-patch32.
- Pinned revision: 3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268.
- Downloaded checkpoint SHA256: a63082132ba4f97a80bea76823f544493bffa8082296d62d71581a4feff1576f (verified before loading).
- Device: CUDA, RTX 4050 Laptop GPU.
- Taxonomy: version 1.0, backend/taxonomy.py.
- Input: synthetic Component 1 garment_references/dresses/dress-02-batik-wrap.png.
- Actual POST /api/analyse-garment returned HTTP 200.
- Garment type suggestion: dress, relative score 0.9224.
- Pattern suggestion: floral, relative score 0.6387.
- Texture suggestion: smooth_matte, relative score 0.3783, ambiguous.
- Drape style: not_applicable through the non-saree rule.
- Eight automated validation/ranking tests passed. Recommendation HTTP smoke test passed.
- Frontend analysis.js passed the Node syntax check; browser visual layout has not been inspected in this milestone.

These scores are relative candidate similarities, not measured accuracy. One synthetic example establishes execution only. Reviewed real-image evaluation, local adaptation, person analysis and incorporation of corrected attributes into ranking remain outstanding.
