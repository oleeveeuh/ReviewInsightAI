# YouTube Video Selection Guide

## Objective
Manually find 15-20 relevant YouTube videos about Amazon Fulfillment Center (FC) employee experiences for sentiment analysis.

## Search Queries
Use these queries on YouTube:
- "Amazon fulfillment center experience"
- "Day in the life Amazon warehouse"
- "Amazon FC employee"
- "Working at Amazon warehouse"
- "Amazon peak season"
- "Former Amazon employee"
- "Amazon warehouse truth"
- "Amazon fulfillment center tour"

## Selection Criteria
| Criteria | Requirement |
|----------|-------------|
| Captions | Must have CC button available |
| Duration | 5-20 minutes |
| Views | 5,000+ |
| Language | English |
| Age | Published within last 2 years |
| Content | Actual employee experiences (not news/commentary) |

## Target Mix
- **40%** Current employees (typically positive/neutral)
- **40%** Former employees (typically negative/critical)
- **20%** Mixed/balanced perspectives

## How to Check Captions
1. Open video
2. Look for CC button near bottom right
3. If available, click → check "English - Auto-generated" or "English"

## How to Extract Video ID
From URL: `https://www.youtube.com/watch?v=dQw4w9WgXcQ`
Video ID is: `dQw4w9WgXcQ`

## CSV Format
Add each video to `video_list.csv` with these columns:

| Column | Description | Example |
|--------|-------------|---------|
| video_id | ID from URL | dQw4w9WgXcQ |
| title | Video title | Day in Life Amazon FC |
| channel | Channel name | John Doe |
| url | Full URL | https://youtube.com/watch?v=dQw4w9WgXcQ |
| views | View count | 50000 |
| upload_date | YYYY-MM-DD | 2024-03-15 |
| duration_min | Length in minutes | 12 |
| employee_status | current/former/unknown | current |
| notes | Brief description | Positive perspective |

## Example CSV Row
```csv
video_id,title,channel,url,views,upload_date,duration_min,employee_status,notes
dQw4w9WgXcQ,Day in Life Amazon FC,John Doe,https://youtube.com/watch?v=dQw4w9WgXcQ,50000,2024-03-15,12,current,Positive perspective
```

## Tracking Progress
Target: 15-20 videos
- [ ] Current employees: 0/8
- [ ] Former employees: 0/8
- [ ] Mixed/unknown: 0/4

## Notes
- Skip promotional/recruitment videos from Amazon official channels
- Prioritize authentic "day in the life" content
- Look for videos with discussion in comments (engagement signal)
