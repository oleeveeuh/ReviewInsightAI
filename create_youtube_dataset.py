#!/usr/bin/env python3
"""
Create YouTube dataset with original and cleaned transcripts.
Generates a CSV file with video metadata and transcript processing.
"""

import re
import pandas as pd
from pathlib import Path

# Paths
DATA_DIR = Path("data")
TRANSCRIPT_DIR = DATA_DIR / "youtube" / "transcripts"
OUTPUT_DIR = DATA_DIR / "youtube"
OUTPUT_FILE = OUTPUT_DIR / "youtube_dataset.csv"

def clean_transcript_line(line):
    """
    Clean a single transcript line by removing timestamps, music tags,
    filler words, and normalizing text.
    """
    # Remove timestamp patterns like [0.00] or [123.45]
    line = re.sub(r'\[\d+\.\d+\]', '', line)

    # Remove music tags
    line = re.sub(r'\[Music\]', '', line, flags=re.IGNORECASE)
    line = re.sub(r'\[music\]', '', line)

    # Remove other bracketed content (sound effects, etc.)
    line = re.sub(r'\[[^\]]*\]', '', line)

    # Convert to lowercase and strip
    line = line.lower().strip()

    # Remove common filler words and phrases
    filler_words = [
        r'\bum+\b', r'\buh+\b', r'\bah+\b', r'\ber+\b', r'\bumm+\b',
        r'\byou know\b', r'\blike\b', r'\bi mean\b', r'\byou see\b',
        r'\bkind of\b', r'\bsort of\b', r'\bjust\b', r'\bpretty\b',
        r'\bbasically\b', r'\bactually\b', r'\bliterally\b', r'\breally\b',
        r'\bsomething\b', r'\banything\b', r'\bwhatever\b'
    ]

    for filler in filler_words:
        line = re.sub(filler, '', line, flags=re.IGNORECASE)

    # Remove extra whitespace
    line = re.sub(r'\s+', ' ', line)

    # Remove repeated punctuation
    line = re.sub(r'([!?])\1+', r'\1', line)

    # Fix common transcription errors
    corrections = {
        'gonna': 'going to',
        'wanna': 'want to',
        'gotta': 'got to',
        'kinda': 'kind of',
        'sorta': 'sort of',
        'cause': 'because',
        'tho': 'though',
        'tho ': 'though',
        ' til ': ' until ',
        'u': 'you',
        'ur': 'your',
        'r': 'are',
        ' ima ': ' i am going to ',
        'dont': "don't",
        'doesnt': "doesn't",
        'wont': "won't",
        'cant': "can't",
        'didnt': "didn't",
        'couldnt': "couldn't",
        'shouldnt': "shouldn't",
        'wouldnt': "wouldn't",
        'hasnt': "hasn't",
        'havent': "haven't",
        'isnt': "isn't",
        'arent': "aren't",
        'wasnt': "wasn't",
        'werent': "weren't",
        'alot': 'a lot',
        'alright': 'all right',
    }

    for wrong, correct in corrections.items():
        line = re.sub(rf'\b{wrong}\b', correct, line, flags=re.IGNORECASE)

    # Remove empty or very short lines
    if len(line.strip()) < 3:
        return ""

    return line.strip()

def process_transcript_file(file_path, video_id):
    """
    Process a transcript file and extract both original and cleaned versions.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    original_lines = []
    cleaned_lines = []

    for line in lines:
        original_line = line.strip()
        if not original_line:
            continue

        cleaned_line = clean_transcript_line(original_line)

        original_lines.append(original_line)
        if cleaned_line:  # Only add non-empty cleaned lines
            cleaned_lines.append(cleaned_line)

    return {
        'video_id': video_id,
        'original_lines': original_lines,
        'cleaned_lines': cleaned_lines,
        'line_count': len(original_lines),
        'cleaned_count': len(cleaned_lines)
    }

def extract_video_metadata(file_path):
    """
    Extract metadata from transcript file or derive from filename.
    """
    video_id = file_path.stem

    # Try to read JSON metadata if it exists
    json_path = file_path.with_suffix('.json')
    metadata = {
        'video_id': video_id,
        'title': f'YouTube Video {video_id}',
        'source': 'youtube'
    }

    if json_path.exists():
        try:
            import json
            with open(json_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
                metadata['title'] = json_data.get('title', metadata['title'])
        except Exception:
            pass

    return metadata

def main():
    """Main function to create the YouTube dataset."""
    print("Creating YouTube dataset...")

    # Find all transcript files
    transcript_files = list(TRANSCRIPT_DIR.glob("*.txt"))

    if not transcript_files:
        print(f"No transcript files found in {TRANSCRIPT_DIR}")
        return

    print(f"Found {len(transcript_files)} transcript files")

    # Process each transcript file
    dataset_rows = []

    for transcript_file in sorted(transcript_files):
        print(f"\nProcessing: {transcript_file.name}")

        # Extract metadata
        metadata = extract_video_metadata(transcript_file)

        # Process transcript
        transcript_data = process_transcript_file(transcript_file, metadata['video_id'])

        # Create rows for each line
        originals = transcript_data['original_lines']
        cleaned_lines = transcript_data['cleaned_lines']
        padded = cleaned_lines + [''] * (len(originals) - len(cleaned_lines))
        for i, (original, cleaned) in enumerate(zip(originals, padded)):
            dataset_rows.append({
                'video_id': metadata['video_id'],
                'title': metadata['title'],
                'source': metadata['source'],
                'line_number': i + 1,
                'transcript_original': original,
                'transcript_cleaned': cleaned if cleaned else original
            })

        print(f"  Lines: {transcript_data['line_count']}, Cleaned: {transcript_data['cleaned_count']}")

    # Create DataFrame
    df = pd.DataFrame(dataset_rows)

    # Save to CSV
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8')

    print(f"\n{'='*50}")
    print("Dataset created successfully!")
    print(f"Total rows: {len(df)}")
    print(f"Unique videos: {df['video_id'].nunique()}")
    print(f"Output file: {OUTPUT_FILE}")

    # Display statistics
    print(f"\n{'='*50}")
    print("Dataset Statistics:")
    print(f"Total transcript lines: {len(df)}")
    print(f"Videos processed: {df['video_id'].nunique()}")
    print(f"Average lines per video: {len(df) / df['video_id'].nunique():.1f}")

    # Show sample data
    print(f"\n{'='*50}")
    print("Sample Data (first 5 rows):")
    print(df[['video_id', 'line_number', 'transcript_original', 'transcript_cleaned']].head())

    return df

if __name__ == "__main__":
    main()
