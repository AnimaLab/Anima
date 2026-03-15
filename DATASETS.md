# Datasets And Model Assets

This public repository intentionally does not bundle benchmark datasets or raw
benchmark outputs.

## LoCoMo

- Upstream project: `https://github.com/snap-research/locomo`
- Public dataset URL used by the harness:
`https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json`
- The dataset is not stored in this repo because it contains third-party media
references and is treated conservatively for redistribution.

### How To Use It

- `benchmarks/locomo/run.py` and `benchmarks/locomo/run_think.py` will download
the dataset automatically if the requested path does not exist.
- Other benchmark helper scripts accept an explicit dataset path via
`--data-path` or `--locomo-path`.

## LongMemEval

- Upstream dataset URL used by the harness:
`https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json`
- `benchmarks/longmemeval/run.py` downloads the dataset automatically if it is
missing.

## Model Assets

- Local ONNX embedding assets are downloaded or supplied outside version control.
- API-backed benchmarks may also require `OPENAI_API_KEY`,
`PROCESSOR_API_KEY`, or `HUGGING_FACE_HUB_TOKEN` depending on the run mode.
