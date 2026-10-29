# Real drape candidate inventory — 2026-10-18

The user supplied two category folders and stated that the images are authorized for use. No needs_review directory is required or created. Folder labels are user-supplied candidate labels, not independent expert ground truth. Per-image source URLs and licence/permission references were not supplied and remain null in the inventory; user authorization is recorded separately rather than invented.

Input: data/local_attributes/drape_candidates/real. Current inventory: runtime/datasets/drape_inventory_02/inventory.json (inventory_01 was superseded after JFIF support was added).

- Nivi: 52 readable files.
- Kandyan/Osariya: 63 readable files.
- Total: 115 readable files, 114 distinct decoded RGB images.
- Unreadable/ignored files: zero.
- Originals remain unchanged. No splits, training or test evaluation have been performed.

## Duplication

Exact duplicate: kandyan_osariya/images (15).jpg and images (42).jpg. Keep only one in a future derived manifest; no original should be deleted automatically.

Perceptual-hash candidates, also visually consistent with resized/recompressed copies in the contact sheets:

1. charming-pink-black-floral-handmade-batik-saree-yobo-819x1024.webp and images (5).jpg.
2. elegant-pink-black-polka-dot-handmade-batik-saree.webp and images (6).jpg.
3. images (9).jpg and Sinhalese Traditional Dress _ Osari Design _ Sri Lanka.jpeg.

These need canonical selection or shared source groups before splitting. Perceptual hashes can miss other same-garment views; 114 exact-unique files must not be described as 114 independent physical garments. Some contact-sheet examples have similar clothing/background setups and some are cropped portraits or have partly occluded waists. This is a data-quality observation, not an automatic relabelling or expert drape validation.

The next preparation step is a derived, deduplicated manifest with garment/source grouping and a documented user-label review. Preserve real folder contents; do not assign related views across train/validation/test. The current inventory intentionally leaves source_group and split unassigned. No synthetic examples are included.
