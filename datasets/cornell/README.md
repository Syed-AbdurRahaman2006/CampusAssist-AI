# Cornell Movie-Dialogs Corpus — Prepared Dataset

## Overview

This directory contains the preprocessed Cornell Movie-Dialogs Corpus, prepared for fine-tuning the `microsoft/DialoGPT-small` conversational model.

## Dataset Information

- **Source:** [Cornell Movie-Dialogs Corpus](https://convokit.cornell.edu/documentation/movie.html)
- **License:** Available for academic research use (NSF-supported)
- **Reference:** Danescu-Niculescu-Mizil & Lee, ACL 2011
- **Original Size:** 304,713 utterances from 617 movies
- **Processed Pairs:** 220,109 dialogue pairs

## Files

### Processed Datasets

- **`train.jsonl`** — 175,381 dialogue pairs (79.68%)
- **`validation.jsonl`** — 22,037 dialogue pairs (10.01%)
- **`test.jsonl`** — 22,691 dialogue pairs (10.31%)
- **`dataset_stats.json`** — Comprehensive statistics and metadata

### Format

Each line in the JSONL files contains a JSON object with:

```json
{
  "input": "What are you doing?",
  "response": "Just thinking about things."
}
```

## Statistics

### Split Distribution
- **Training:** 175,381 pairs (80%)
- **Validation:** 22,037 pairs (10%)
- **Test:** 22,691 pairs (10%)

### Length Statistics (in words)
- **Input:** Mean 10.63 words, Median 7 words (Range: 1-556)
- **Response:** Mean 10.28 words, Median 7 words (Range: 1-313)

## Data Processing

The dataset was prepared using the following process:

1. **Download:** Retrieved via ConvoKit library
2. **Pair Extraction:** Consecutive utterances converted to input-response pairs
3. **Cleaning:** Whitespace normalization, empty pair removal
4. **Deduplication:** Removed 1,173 exact duplicate pairs
5. **Splitting:** Conversation-level split to prevent data leakage
6. **Validation:** Format and content validation

### Key Features

- ✅ **No data leakage:** Conversations split at the movie level
- ✅ **Reproducible:** Fixed random seed (42) for consistent splits
- ✅ **Validated:** All records verified for format and content
- ✅ **Clean:** Normalized text with valid input-response pairs only

## Usage for DialoGPT Fine-tuning

These datasets are ready for the next phase: fine-tuning `microsoft/DialoGPT-small`.

### Recommended Approach

1. Load datasets using `datasets` library or custom data loader
2. Apply DialoGPT tokenizer with appropriate special tokens
3. Concatenate input and response with EOS tokens
4. Create attention masks and labels
5. Use causal language modeling objective

### Sequence Length Guidance

Based on statistics:
- **Recommended max length:** 128-256 tokens (covers most dialogues)
- **90th percentile:** ~20 words per turn
- **Consider truncation:** For very long outliers (max 556 words)

## Reproduction

To regenerate this dataset:

```powershell
# From project root
python scripts\prepare_dataset.py
```

The script is fully automated and reproducible.

## Citation

If using this dataset in research, cite the original paper:

```bibtex
@InProceedings{Danescu-Niculescu-Mizil+Lee:11a,
  author={Cristian Danescu-Niculescu-Mizil and Lillian Lee},
  title={Chameleons in imagined conversations: 
         A new approach to understanding coordination of linguistic style in dialogs.},
  booktitle={Proceedings of the Workshop on Cognitive Modeling and Computational Linguistics, ACL 2011},
  year={2011}
}
```

## Notes

- **Not for production:** Movie dialogues don't represent campus policies
- **Training purpose only:** For learning conversational patterns
- **Domain adaptation needed:** Fine-tuned model will require further adaptation with campus-specific data
- **Quality varies:** Movie scripts contain varied language styles and contexts

## Next Steps

1. ✅ Dataset prepared
2. ⏳ Implement DialoGPT fine-tuning pipeline
3. ⏳ Train model on prepared data
4. ⏳ Evaluate model performance
5. ⏳ Integrate with CampusAssist AI backend
