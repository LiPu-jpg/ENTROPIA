# HotpotQA Small Samples

These files are small HotpotQA samples exported from the official CMU HotpotQA
JSON files. They are intended for offline cluster smoke tests when Hugging Face
is unavailable.

Source URLs:

- `http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_train_v1.1.json`
- `http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json`

Files:

- `hotpot_train_1k.json`: first 1,000 examples from the official training file.
- `hotpot_dev_distractor_500.json`: first 500 examples from the official distractor dev file.
- `full/hotpot_train_full.json.gz.part-00` ... `part-02`: split gzip of the full official training file.
- `full/hotpot_dev_distractor_full.json.gz`: gzip of the full official distractor dev file.

Full training set reconstruction:

```bash
cat data/raw/hotpotqa/full/hotpot_train_full.json.gz.part-* \
  > data/raw/hotpotqa/full/hotpot_train_full.json.gz

gunzip -c data/raw/hotpotqa/full/hotpot_train_full.json.gz \
  > data/raw/hotpotqa/full/hotpot_train_full.json
```

Full distractor dev reconstruction:

```bash
gunzip -c data/raw/hotpotqa/full/hotpot_dev_distractor_full.json.gz \
  > data/raw/hotpotqa/full/hotpot_dev_distractor_full.json
```

Record counts:

- full train: 90,447 examples
- full distractor dev: 7,405 examples

Example:

```bash
python scripts/probe_searchqa.py \
  --dataset hotpotqa \
  --data_file data/raw/hotpotqa/hotpot_train_1k.json \
  --n 100
```

The files preserve the official HotpotQA record schema:

```json
{
  "_id": "...",
  "question": "...",
  "answer": "...",
  "supporting_facts": [["Title", 0]],
  "context": [["Title", ["sentence 1", "sentence 2"]]]
}
```
