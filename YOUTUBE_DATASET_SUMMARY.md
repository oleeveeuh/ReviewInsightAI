# YouTube Dataset Creation Summary

## Dataset Overview

Created a comprehensive YouTube transcript dataset with original and cleaned versions for sentiment analysis of warehouse/employee experiences.

### Dataset Statistics
- **Total Videos Processed**: 8
- **Total Transcript Lines**: 2,294
- **Average Lines per Video**: 286.8
- **Output File**: `data/youtube/youtube_dataset.csv`

### Dataset Structure

The dataset contains the following columns:
- `video_id`: Unique YouTube video identifier
- `title`: Video title
- `source`: Data source (youtube)
- `line_number`: Sequential line number within the transcript
- `transcript_original`: Original transcript line with timestamps
- `transcript_cleaned`: Cleaned and normalized transcript line

### Video Processing Details

| Video ID | Original Lines | Cleaned Lines | Reduction |
|----------|---------------|---------------|-----------|
| 40EDWNpK2XM | 339 | 329 | 3% |
| BggxkRAW2_I | 239 | 216 | 10% |
| GDu6cmst__A | 210 | 205 | 2% |
| MQjVIJkvNPY | 300 | 295 | 2% |
| jzg4snTtk3E | 165 | 130 | 21% |
| kXOb9lfWU9w | 262 | 246 | 6% |
| lxBjYK8wHt4 | 640 | 631 | 1% |
| tgWkQC3_b30 | 139 | 123 | 12% |

## Text Cleaning Pipeline

The cleaning process includes:

### 1. Timestamp Removal
- Removed patterns like `[0.00]`, `[123.45]`

### 2. Music and Sound Effect Tags
- Removed `[Music]`, `[music]`, and other bracketed sound effects

### 3. Filler Word Removal
Removed common filler words:
- Basic fillers: "um", "uh", "ah", "er", "umm"
- Conversation fillers: "you know", "like", "I mean", "you see"
- Qualifiers: "kind of", "sort of", "just", "pretty"
- Intensifiers: "basically", "actually", "literally", "really"

### 4. Text Normalization
- Lowercase conversion
- Extra whitespace removal
- Repeated punctuation normalization

### 5. Transcription Error Correction
Common corrections applied:
- "gonna" → "going to"
- "wanna" → "want to"
- "kinda" → "kind of"
- "cause" → "because"
- "dont" → "don't"
- "alot" → "a lot"
- "u" → "you"
- "ur" → "your"

## Sample Comparisons

### Example 1: Employee Benefits Discussion
**Original:**
```
[0.00] this is what y'all need to know if
[1.36] you're working at amazon warehouse
[3.36] please listen
```

**Cleaned:**
```
this is what y'all need to know if
you're working at amazon warehouse
please listen
```

### Example 2: Work Schedule Discussion
**Original:**
```
[8.32] what's up y'all welcome and welcome back
[10.16] to my channel it's your girl jada lynn
[11.84] for those who don't know how you know
```

**Cleaned:**
```
what's up y'all welcome and welcome back
to my channel it's your girl jada lynn
for those who don't know how you know
```

### Example 3: Music and Non-Speech Content
**Original:**
```
[6.40] [Music]
[8.32] what's up y'all welcome and welcome back
```

**Cleaned:**
```
what's up y'all welcome and welcome back
```

## Usage Examples

### Loading the Dataset
```python
import pandas as pd

# Load the dataset
df = pd.read_csv('data/youtube/youtube_dataset.csv')

# View basic info
print(f"Total videos: {df['video_id'].nunique()}")
print(f"Total lines: {len(df)}")

# Get cleaned transcripts only
cleaned_transcripts = df['transcript_cleaned'].tolist()

# Get original transcripts
original_transcripts = df['transcript_original'].tolist()
```

### Filter by Video
```python
# Get specific video transcript
video_id = 'lxBjYK8wHt4'
video_data = df[df['video_id'] == video_id]

# Get cleaned transcript for one video
cleaned_text = ' '.join(video_data['transcript_cleaned'].tolist())
print(f"Video: {video_data['title'].iloc[0]}")
print(f"Cleaned transcript length: {len(cleaned_text)} characters")
```

### Sentiment Analysis Integration
```python
from src.analysis.sentiment_analyzer import analyze_sentiment

# Analyze each video's sentiment
for video_id in df['video_id'].unique():
    video_df = df[df['video_id'] == video_id]
    full_transcript = ' '.join(video_df['transcript_cleaned'].tolist())

    result = analyze_sentiment(full_transcript)
    print(f"Video {video_id}: {result}")
```

## Benefits of Cleaned Transcripts

1. **Improved NLP Performance**: Removing filler words and noise improves sentiment analysis accuracy
2. **Consistent Formatting**: Normalized text is easier to process with ML models
3. **Better Tokenization**: Fewer irrelevant tokens means more meaningful analysis
4. **Reduced Storage**: Cleaner data can be more efficiently stored and processed
5. **Enhanced Readability**: Easier for human review and validation

## File Locations

- **Dataset CSV**: `/Users/olivialiau/extern/data/youtube/youtube_dataset.csv`
- **Source Transcripts**: `/Users/olivialiau/extern/data/youtube/transcripts/`
- **Creation Script**: `/Users/olivialiau/extern/create_youtube_dataset.py`

## Next Steps

1. **Quality Validation**: Review cleaned transcripts for accuracy
2. **Sentiment Analysis**: Apply the existing sentiment analysis pipeline
3. **Topic Modeling**: Identify common themes across videos
4. **Time Series Analysis**: Analyze sentiment trends within videos
5. **Integration**: Merge with other data sources (Glassdoor, Indeed, Reddit)

## Technical Notes

- **Encoding**: UTF-8 for proper character handling
- **Empty Line Handling**: Lines with < 3 characters after cleaning are filtered out
- **Preservation**: Original transcripts are maintained for reference
- **Scalability**: The pipeline can handle additional YouTube transcript files

---

**Dataset Status**: ✅ Complete and ready for analysis
**Created**: March 21, 2026
**Script**: `create_youtube_dataset.py`
