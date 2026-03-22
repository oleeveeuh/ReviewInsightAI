# Reddit API Access Request

## Reddit Account Name
[Your Reddit username]

## What benefit/purpose will the bot/app have for Redditors?
This is a **non-commercial academic research project** called "ReviewInsight" focused on understanding employee sentiment and workplace experiences in the fulfillment/logistics industry. The research aims to:

- Identify common pain points and positive aspects of warehouse/fulfillment work
- Track how employee sentiment changes over time (seasonal patterns, policy changes)
- Provide insights that could improve workplace conditions in the industry
- Create publicly available aggregated findings to help job seekers make informed decisions

**This is not a commercial product.** No data will be sold. Findings will be aggregated and anonymized. The bot/app itself will not interact with users or post to Reddit.

## Detailed Description of What the Bot/App Will Be Doing

The application is a **read-only data collection and analysis tool** that will:

1. **Collect public posts and comments** from work-related subreddits (r/jobs, r/recruitinghell, r/AmazonFC, r/AmazonEmployees, r/walmart, r/Target, etc.)

2. **Extract metadata** including:
   - Post title, content, timestamp
   - Comment text and timestamps
   - Upvote/downvote ratios
   - Flair tags

3. **Perform sentiment analysis** using NLP techniques to identify:
   - Overall sentiment (positive/negative/neutral)
   - Key themes mentioned (pay, management, workload, safety, etc.)
   - Temporal patterns (how sentiment changes during peak seasons, after policy announcements)

4. **Aggregate and anonymize findings** - No individual users will be identified in any published results

## What Examples Can You Provide?

Example research questions:
- How does sentiment toward Amazon warehouse work vary by season?
- What are the most frequently mentioned complaints across fulfillment companies?
- Has sentiment in r/jobs regarding warehouse work changed over the past 2 years?

Example output format (aggregated, anonymized):
| Company | Avg Sentiment | Top Themes | Sample Size |
|---------|---------------|------------|-------------|
| Amazon  | -0.32         | Schedule, workload, management | 5,400 posts |
| Walmart | -0.18         | Pay, staffing, breaks | 2,100 posts |

## What is Missing from Devvit That Prevents Building on That Platform?

Devvit is designed for building **interactive apps that run within Reddit's ecosystem** and engage users directly. This research project requires capabilities that Devvit does not support:

1. **Historical data access** - Need to collect posts/comments from the past 2+ years for trend analysis
2. **Bulk data extraction** - Need to collect thousands of posts across multiple subreddits
3. **External processing** - Data must be exported and analyzed offline using NLP tools (spaCy, transformers, etc.)
4. **No user interaction needed** - The tool never posts, comments, or responds to users; it only reads
5. **Cross-subreddit aggregation** - Analysis spans multiple communities, not confined to a single subreddit's Devvit app context

Devvit apps are designed for real-time user engagement within Reddit. This is a **read-only batch processing tool** for research purposes.

## Link to Source Code

Source code will be hosted at: [Your GitHub URL]

The codebase will include:
- API client using Reddit's official API (via PRAW or direct HTTP requests)
- Data export functionality for NLP analysis pipelines
- No automated posting, commenting, or voting functionality

## What Subreddits Do You Intend to Use the Bot/App In?

The tool will **read from** these subreddits (no posting):

- r/jobs - General job discussion
- r/recruitinghell - Job search/hiring experiences
- r/AmazonFC - Amazon fulfillment center employees
- r/AmazonEmployees - Amazon employees (if accessible)
- r/walmart - Walmart employees
- r/Target - Target employees
- r/warehouse - Warehouse work discussion
- r/logistics - Logistics industry discussion

## Additional Notes

- **Rate limiting**: Will respect Reddit's rate limits (60 requests/minute for authenticated requests)
- **Read-only**: The tool will never post, comment, vote, or send messages
- **Compliance**: Will comply with Reddit's Data API Terms and User Agreement
- **Academic use**: This is for research and educational purposes only

---

**After submission**, monitor your Reddit email/modmail for approval. This may take 1-4 weeks.

In the meantime, proceed with alternative data sources (YouTube, Glassdoor, Indeed).
