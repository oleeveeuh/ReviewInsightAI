# Glassdoor Data Collection Guide

## Objective
Collect employee reviews for Amazon fulfillment centers and warehouse positions.

## Target Data
- **Company**: Amazon (and competitors: Walmart, Target, FedEx, UPS)
- **Job Type**: Warehouse, Fulfillment Center, Logistics, Delivery Driver
- **Locations**: Major fulfillment hubs (California, Texas, Arizona, New Jersey, etc.)
- **Time Range**: Past 2 years

## Selection Criteria
| Criteria | Requirement |
|----------|-------------|
| Review length | 50+ words (enough for sentiment analysis) |
| Rating | All stars (1-5) - need full range |
| Date | Within last 2 years |
| Position | Warehouse/Fulfillment/Delivery roles |
| Language | English |

## Data to Extract
| Field | Description |
|-------|-------------|
| company | Amazon, Walmart, etc. |
| location | City, State |
| position | Job title |
| rating | 1-5 stars |
| date | Review date |
| pros | Employee's listed pros |
| cons | Employee's listed cons |
| advice_to_mgmt | Any advice to management |
| status | Current vs Former employee |

## CSV Format
Create: `data/glassdoor/reviews.csv`

```csv
company,location,position,rating,date,pros,cons,advice_to_mgmt,status,source_url
```

## Collection Methods

### Option 1: Manual (Small Scale)
1. Visit glassdoor.com
2. Search "Amazon fulfillment center [city]"
3. Filter by: Warehouse & Fulfillment category
4. Copy relevant reviews to CSV
5. Target: 50-100 reviews

### Option 2: Glassdoor API (Requires Application)
- Apply at: https://www.glassdoor.com/developer/index.htm
- Official API for research purposes
- More structured data

### Option 3: Third-party Libraries
- `glassdoor-python` (unofficial, may break)
- Use at your own risk, respect ToS

## Target Mix
- 70% Amazon reviews
- 30% Competitor reviews (for comparison)
- Mix of 1-star and 5-star reviews
- Mix of current and former employees

## Notes
- Glassdoor content is user-submitted and may be biased
- Consider verification status of reviews
- Aggregate findings, don't quote identifiable individuals
