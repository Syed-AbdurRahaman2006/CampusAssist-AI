"""
Cornell Movie-Dialogs Corpus Dataset Preparation Script

This script downloads, processes, and splits the Cornell Movie-Dialogs Corpus
for future DialoGPT fine-tuning. It creates conversational pairs suitable for
causal language model training.

Dataset Source: Cornell University
Reference: Danescu-Niculescu-Mizil & Lee, ACL 2011
Available via ConvoKit: https://convokit.cornell.edu/
"""

import json
import os
import random
from collections import defaultdict
from pathlib import Path

# Configuration
RANDOM_SEED = 42
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

BASE_DIR = Path(__file__).parent.parent
DATASETS_DIR = BASE_DIR / "datasets" / "cornell"
DATASETS_DIR.mkdir(parents=True, exist_ok=True)


def download_and_load_corpus():
    """Download Cornell Movie-Dialogs Corpus using ConvoKit."""
    try:
        from convokit import Corpus, download
        
        print("Downloading Cornell Movie-Dialogs Corpus via ConvoKit...")
        print("This may take a few minutes on first run...")
        
        corpus = Corpus(filename=download("movie-corpus"))
        
        print(f"\n✓ Dataset downloaded successfully!")
        print(f"  Total utterances: {len(corpus.utterances)}")
        print(f"  Total conversations: {len(corpus.conversations)}")
        print(f"  Total speakers: {len(corpus.speakers)}")
        
        return corpus
        
    except ImportError:
        print("ERROR: ConvoKit is not installed.")
        print("Please install it with: pip install convokit")
        return None
    except Exception as e:
        print(f"ERROR downloading dataset: {e}")
        return None


def extract_dialogue_pairs(corpus):
    """
    Extract consecutive utterance pairs from conversations.
    Returns pairs grouped by conversation to prevent data leakage.
    """
    print("\nExtracting dialogue pairs from conversations...")
    
    conversation_pairs = defaultdict(list)
    total_pairs = 0
    skipped_pairs = 0
    
    for conv_id, conversation in corpus.conversations.items():
        utterances = list(conversation.iter_utterances())
        
        # Sort by timestamp to ensure correct order
        utterances.sort(key=lambda u: u.timestamp if u.timestamp else "")
        
        # Create consecutive pairs
        for i in range(len(utterances) - 1):
            input_text = utterances[i].text
            response_text = utterances[i + 1].text
            
            # Clean and validate
            input_clean = clean_text(input_text)
            response_clean = clean_text(response_text)
            
            if input_clean and response_clean:
                conversation_pairs[conv_id].append({
                    "input": input_clean,
                    "response": response_clean,
                    "movie_idx": conversation.meta.get("movie_idx", "unknown"),
                    "movie_name": conversation.meta.get("movie_name", "unknown")
                })
                total_pairs += 1
            else:
                skipped_pairs += 1
    
    print(f"  ✓ Extracted {total_pairs:,} valid dialogue pairs")
    print(f"  ✓ From {len(conversation_pairs):,} conversations")
    print(f"  ✓ Skipped {skipped_pairs:,} invalid pairs")
    
    return conversation_pairs


def clean_text(text):
    """Clean and normalize text while preserving conversational style."""
    if not text:
        return ""
    
    # Strip whitespace
    text = text.strip()
    
    # Remove empty strings
    if not text:
        return ""
    
    # Normalize multiple spaces
    text = " ".join(text.split())
    
    return text


def split_by_conversation(conversation_pairs, train_r, val_r, test_r, seed):
    """
    Split dataset by conversation to prevent leakage.
    Ensures no conversation appears in multiple splits.
    """
    print("\nSplitting dataset (by conversation to prevent leakage)...")
    
    # Get all conversation IDs
    conv_ids = list(conversation_pairs.keys())
    
    # Shuffle with fixed seed for reproducibility
    random.seed(seed)
    random.shuffle(conv_ids)
    
    # Calculate split points
    total_convs = len(conv_ids)
    train_end = int(total_convs * train_r)
    val_end = train_end + int(total_convs * val_r)
    
    # Split conversation IDs
    train_convs = conv_ids[:train_end]
    val_convs = conv_ids[train_end:val_end]
    test_convs = conv_ids[val_end:]
    
    # Flatten pairs for each split
    train_pairs = []
    for conv_id in train_convs:
        train_pairs.extend(conversation_pairs[conv_id])
    
    val_pairs = []
    for conv_id in val_convs:
        val_pairs.extend(conversation_pairs[conv_id])
    
    test_pairs = []
    for conv_id in test_convs:
        test_pairs.extend(conversation_pairs[conv_id])
    
    print(f"  Train: {len(train_pairs):,} pairs from {len(train_convs):,} conversations")
    print(f"  Val:   {len(val_pairs):,} pairs from {len(val_convs):,} conversations")
    print(f"  Test:  {len(test_pairs):,} pairs from {len(test_convs):,} conversations")
    
    return train_pairs, val_pairs, test_pairs


def remove_duplicates(pairs):
    """Remove exact duplicate input-response pairs."""
    seen = set()
    unique_pairs = []
    duplicates = 0
    
    for pair in pairs:
        key = (pair["input"], pair["response"])
        if key not in seen:
            seen.add(key)
            unique_pairs.append(pair)
        else:
            duplicates += 1
    
    if duplicates > 0:
        print(f"  Removed {duplicates:,} duplicate pairs")
    
    return unique_pairs


def compute_statistics(train_pairs, val_pairs, test_pairs):
    """Compute comprehensive dataset statistics."""
    print("\nComputing dataset statistics...")
    
    all_pairs = train_pairs + val_pairs + test_pairs
    
    # Length statistics
    input_lengths = [len(p["input"].split()) for p in all_pairs]
    response_lengths = [len(p["response"].split()) for p in all_pairs]
    
    stats = {
        "dataset_name": "Cornell Movie-Dialogs Corpus",
        "source": "https://convokit.cornell.edu/documentation/movie.html",
        "license": "Available for academic research use (NSF-supported)",
        "reference": "Danescu-Niculescu-Mizil & Lee, ACL 2011",
        "splits": {
            "train": {
                "num_pairs": len(train_pairs),
                "percentage": round(len(train_pairs) / len(all_pairs) * 100, 2)
            },
            "validation": {
                "num_pairs": len(val_pairs),
                "percentage": round(len(val_pairs) / len(all_pairs) * 100, 2)
            },
            "test": {
                "num_pairs": len(test_pairs),
                "percentage": round(len(test_pairs) / len(all_pairs) * 100, 2)
            },
            "total": len(all_pairs)
        },
        "length_statistics": {
            "input": {
                "min": min(input_lengths),
                "max": max(input_lengths),
                "mean": round(sum(input_lengths) / len(input_lengths), 2),
                "median": sorted(input_lengths)[len(input_lengths) // 2]
            },
            "response": {
                "min": min(response_lengths),
                "max": max(response_lengths),
                "mean": round(sum(response_lengths) / len(response_lengths), 2),
                "median": sorted(response_lengths)[len(response_lengths) // 2]
            }
        },
        "sample_pairs": [
            {"input": p["input"], "response": p["response"]}
            for p in random.sample(all_pairs, min(5, len(all_pairs)))
        ]
    }
    
    return stats


def save_jsonl(pairs, filepath):
    """Save pairs in JSON Lines format."""
    with open(filepath, 'w', encoding='utf-8') as f:
        for pair in pairs:
            # Save only input/response for training
            record = {
                "input": pair["input"],
                "response": pair["response"]
            }
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
    
    print(f"  ✓ Saved {len(pairs):,} pairs to {filepath.name}")


def validate_output_files():
    """Validate that output files are correctly formatted."""
    print("\nValidating output files...")
    
    files = ["train.jsonl", "validation.jsonl", "test.jsonl"]
    all_valid = True
    
    for filename in files:
        filepath = DATASETS_DIR / filename
        
        if not filepath.exists():
            print(f"  ✗ Missing: {filename}")
            all_valid = False
            continue
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                
                # Check at least one record
                if len(lines) == 0:
                    print(f"  ✗ Empty file: {filename}")
                    all_valid = False
                    continue
                
                # Validate first and last records
                for line in [lines[0], lines[-1]]:
                    record = json.loads(line)
                    if "input" not in record or "response" not in record:
                        print(f"  ✗ Invalid format in {filename}")
                        all_valid = False
                        break
                    if not record["input"].strip() or not record["response"].strip():
                        print(f"  ✗ Empty text in {filename}")
                        all_valid = False
                        break
                
                print(f"  ✓ Valid: {filename} ({len(lines):,} records)")
        
        except Exception as e:
            print(f"  ✗ Error reading {filename}: {e}")
            all_valid = False
    
    return all_valid


def main():
    """Main dataset preparation pipeline."""
    print("=" * 70)
    print("Cornell Movie-Dialogs Dataset Preparation for DialoGPT Fine-tuning")
    print("=" * 70)
    
    # Step 1: Download corpus
    corpus = download_and_load_corpus()
    if corpus is None:
        print("\n✗ Failed to download dataset. Exiting.")
        return
    
    # Step 2: Extract dialogue pairs
    conversation_pairs = extract_dialogue_pairs(corpus)
    
    # Step 3: Split by conversation
    train_pairs, val_pairs, test_pairs = split_by_conversation(
        conversation_pairs, TRAIN_RATIO, VAL_RATIO, TEST_RATIO, RANDOM_SEED
    )
    
    # Step 4: Remove duplicates
    print("\nRemoving duplicate pairs...")
    train_pairs = remove_duplicates(train_pairs)
    val_pairs = remove_duplicates(val_pairs)
    test_pairs = remove_duplicates(test_pairs)
    
    # Step 5: Compute statistics
    stats = compute_statistics(train_pairs, val_pairs, test_pairs)
    
    # Step 6: Save datasets
    print("\nSaving processed datasets...")
    save_jsonl(train_pairs, DATASETS_DIR / "train.jsonl")
    save_jsonl(val_pairs, DATASETS_DIR / "validation.jsonl")
    save_jsonl(test_pairs, DATASETS_DIR / "test.jsonl")
    
    # Step 7: Save statistics
    stats_path = DATASETS_DIR / "dataset_stats.json"
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"  ✓ Saved statistics to {stats_path.name}")
    
    # Step 8: Validate outputs
    is_valid = validate_output_files()
    
    # Final summary
    print("\n" + "=" * 70)
    print("DATASET PREPARATION COMPLETE")
    print("=" * 70)
    print(f"\n📊 Dataset Statistics:")
    print(f"  Total dialogue pairs: {stats['splits']['total']:,}")
    print(f"  Training set: {stats['splits']['train']['num_pairs']:,} ({stats['splits']['train']['percentage']}%)")
    print(f"  Validation set: {stats['splits']['validation']['num_pairs']:,} ({stats['splits']['validation']['percentage']}%)")
    print(f"  Test set: {stats['splits']['test']['num_pairs']:,} ({stats['splits']['test']['percentage']}%)")
    print(f"\n📏 Average Lengths (words):")
    print(f"  Input: {stats['length_statistics']['input']['mean']} words")
    print(f"  Response: {stats['length_statistics']['response']['mean']} words")
    print(f"\n📂 Output Files:")
    print(f"  {DATASETS_DIR / 'train.jsonl'}")
    print(f"  {DATASETS_DIR / 'validation.jsonl'}")
    print(f"  {DATASETS_DIR / 'test.jsonl'}")
    print(f"  {DATASETS_DIR / 'dataset_stats.json'}")
    
    if is_valid:
        print(f"\n✅ All validation checks passed!")
    else:
        print(f"\n⚠️  Some validation checks failed. Review the output above.")
    
    print("\n💡 Next Steps:")
    print("  1. Review dataset_stats.json for detailed statistics")
    print("  2. Examine sample pairs to verify quality")
    print("  3. Use these datasets for DialoGPT fine-tuning in the next phase")
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
