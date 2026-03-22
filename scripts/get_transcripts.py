#!/usr/bin/env python3
"""
Download YouTube captions for sentiment analysis.
Usage: python scripts/get_transcripts.py
"""

import csv
import os
import sys
from youtube_transcript_api import YouTubeTranscriptApi

# Paths
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
VIDEO_LIST = os.path.join(DATA_DIR, "youtube", "video_list.csv")
OUTPUT_DIR = os.path.join(DATA_DIR, "youtube", "transcripts")

def get_video_id(url_or_id):
    """Extract video ID from URL or return as-is."""
    if len(url_or_id) == 11:
        return url_or_id
    # Handle various YouTube URL formats
    if "v=" in url_or_id:
        return url_or_id.split("v=")[1].split("&")[0]
    if "youtu.be/" in url_or_id:
        return url_or_id.split("youtu.be/")[1].split("?")[0]
    return url_or_id

def download_transcript(video_id, title):
    """Download transcript for a single video."""
    try:
        api = YouTubeTranscriptApi()
        result = api.list(video_id).find_transcript(['en'])
        transcript = result.fetch()
        return transcript
    except Exception as e:
        print(f"  Error: {e}")
        return None

def save_transcript(video_id, title, transcript, output_dir):
    """Save transcript as both TXT and JSON."""
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Clean filename
    safe_title = "".join(c if c.isalnum() or c in (" ", "-", "_") else "_" for c in title)
    safe_title = safe_title[:50]  # Limit length

    # Convert dataclass objects to dict for JSON serialization
    transcript_data = [
        {"start": entry.start, "text": entry.text, "duration": entry.duration}
        for entry in transcript
    ]

    # Save as plain text
    txt_path = os.path.join(output_dir, f"{video_id}.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        for entry in transcript:
            f.write(f"[{entry.start:.2f}] {entry.text}\n")

    # Save as JSON with metadata
    import json
    json_path = os.path.join(output_dir, f"{video_id}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "video_id": video_id,
            "title": title,
            "transcript": transcript_data
        }, f, indent=2)

    print(f"  Saved: {txt_path}")
    return txt_path

def main():
    # Read video list
    videos = []
    with open(VIDEO_LIST, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        videos = list(reader)

    print(f"Found {len(videos)} videos in video_list.csv")

    success = 0
    failed = []

    for video in videos:
        video_id = video["video_id"].strip()
        title = video["title"]

        if not video_id:
            continue

        print(f"\n[{videos.index(video)+1}/{len(videos)}] {title}")
        print(f"  Video ID: {video_id}")

        transcript = download_transcript(video_id, title)
        if transcript:
            save_transcript(video_id, title, transcript, OUTPUT_DIR)
            success += 1
        else:
            failed.append((video_id, title))

    # Summary
    print(f"\n{'='*50}")
    print(f"Done! Downloaded {success}/{len(videos)} transcripts")
    print(f"Transcripts saved to: {OUTPUT_DIR}")

    if failed:
        print(f"\nFailed ({len(failed)}):")
        for vid, title in failed:
            print(f"  - {vid}: {title}")

    return 0 if success == len(videos) else 1

if __name__ == "__main__":
    sys.exit(main())
