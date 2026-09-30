# Training data sources

The packaged classifier has one output target (`mooncake`). It uses a small set of images labeled `not_mooncake` during training; all such examples share one label and are not split into cake, cookie, or other output classes.

## Mooncake positive images

- `data/mooncake/real_0001.jpg` through `real_0207.jpg`: 207 public real mooncake photographs. The source manifest records each source file and page. 184 are from DimSum50 Mooncake; 23 are from Wikimedia Commons. Dataset page: <https://www.kaggle.com/datasets/rock3yu/dimsum50-0-1>.
- `data/mooncake/ai_original/`: 998 original GPT Image 2 PNGs. They are distributed as two **uncompressed TAR** assets in GitHub Release `v1.1.0`; their PNG file bytes are preserved. Prompts and file mapping are in `data/source_manifest.csv`.
- `data/mooncake-composites/`: the earlier background-composited JPEG variants, retained for reference. The new trainer does not use them.

## Non-mooncake training examples

`data/not_mooncake/train/` contains 200 files: 100 AI-generated images and 100 public photographs. `data/not_mooncake/validation/` contains another 200 files, similarly divided. Training uses a 0.5 loss-group weight for the negative batch. All negative examples have the single label `not_mooncake`.

The subset manifest records the source category, prompt/title, author, license, source URL, original filename, and SHA-256. To avoid a known background overlap, 284 Commons photographs used as backgrounds in older mooncake composites were excluded from the negative training and validation subsets.

No user-uploaded photographs are included in these datasets.
