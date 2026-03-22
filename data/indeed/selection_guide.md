# Indeed Data Collection Guide

## Objective
Collect employee reviews for Amazon fulfillment centers and warehouse positions.

## Target Data
- **Company**: Amazon (and competitors: Walmart, Target, FedEx, UPS)
- **Job Type**: Warehouse, Fulfillment Center, Logistics, Delivery Driver
- **Locations**: Major fulfillment hubs nationwide
- **Time Range**: Past 2 years

## Selection Criteria
| Criteria | Requirement |
|----------|-------------|
| Review length | 30+ words |
| Rating | All stars (1-5) |
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
| title | Review headline |
| content | Full review text |
| status | Current vs Former employee |

## CSV Format
Create: `data/indeed/reviews.csv`

```csv
company,location,position,rating,date,title,content,status,source_url
```

## Collection Methods

### Option 1: Manual Collection
1. Visit indeed.com
2. Search "Amazon" in Companies
3. Go to Reviews tab
4. Filter by job type and location
5. Copy relevant reviews to CSV

### Option 2: Indeed API
- Indeed has an official API for employers
- Research access may require partnership
- Info: https://www.indeed.com/hire/api

### Option 3: Browser Extension
- Use tools like Web Scraper browser extension
- Export to CSV format
- Check Indeed's ToS before scraping

## Target Mix
- 50% Amazon reviews
- 25% Walmart reviews
- 25% Target/FedEx/UPS reviews
- Balance of positive and negative reviews

## Indeed vs Glassdoor
| Aspect | Indeed | Glassdoor |
|--------|--------|-----------|
| Volume | Higher | Lower |
| Detail | Less detailed | More structured (pros/cons) |
| Verification | Less strict | More strict |
| Access | Easier manual | More structured API |

## Notes
- Indeed has higher review volume but less detail
- Combine with Glassdoor for more complete picture
- Focus on fulfillment/warehouse specific roles
