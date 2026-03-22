# Labeling Instructions

## Objective
Create a gold standard dataset for training sentiment analysis models on Amazon warehouse employee reviews.

## Target Labels
- **Goal**: 100 labeled reviews
- **Current progress**: Track via `data/labeled/gold_standard.jsonl`

---

## Label Categories

### 1. Overall Sentiment (1-5)

Consider the **entire review** tone, not just specific complaints or praise.

| Score | Label | Description |
|-------|-------|-------------|
| 1 | Very Negative | Strong criticism, actively seeking to leave, warning others away |
| 2 | Negative | Mostly complaints, disappointed, would not recommend |
| 3 | Neutral | Balanced pros/cons, factual tone, mixed feelings |
| 4 | Positive | Mostly praise, satisfied, would recommend |
| 5 | Very Positive | Enthusiastic, loves the job, highly recommended |

**Important Notes:**
- A 3-star Glassdoor rating can still be **negative sentiment** if the tone is critical
- Look at language used: "toxic", "awful", "love it", "great place"
- Consider intensity: "hate it" vs "could be better"

### 2. Themes (Select ALL that apply)

Reviews often contain multiple themes. Select all that are present.

| Code | Theme | Keywords |
|------|-------|----------|
| a | Overtime/Scheduling | mandatory OT, forced overtime, scheduling issues, shift changes |
| b | Pay/Benefits | wage, salary, benefits, insurance, PTO, raise, bonus, hourly |
| c | Management | manager, supervisor, boss, leadership, communication |
| d | Safety | unsafe, injury, dangerous, heat, safety violations, hazard |
| e | Career Growth | promotion, advancement, opportunities, career path |
| f | Workload/Pace | rate, quota, packages, boxes, too fast, slow, scanner |
| g | Work-Life Balance | schedule, shift work, life outside work, family time |
| h | Training | onboarding, training, learning, thrown in, not trained |
| i | Coworkers/Culture | team, people, coworkers, friends, culture, atmosphere |
| j | Other | anything not covered above |

### 3. Retention Risk

Based on the review, how likely is this employee to leave (or have they left)?

| Level | Description |
|-------|-------------|
| Low (🟢) | Satisfied or neutral tone, likely to stay, balanced view |
| Medium (🟡) | Has complaints but seems manageable, some frustration |
| High (🔴) | Very dissatisfied, actively looking, has quit, warning others |

**Consider:**
- "I'm quitting" → High
- "Thinking about leaving" → Medium/High
- "Good for now" → Low
- "Don't work here" → High

---

## Labeling Workflow

1. Read the entire review
2. Assess overall tone and sentiment
3. Identify all themes present
4. Evaluate retention risk
5. Add optional notes if needed

---

## Tips for Consistency

- **Focus on tone** over factual content
- A review can have positive content but negative overall tone
- A review can list pros but still express dissatisfaction
- When in doubt, lean toward **neutral (3)**

---

## Time Budget

- **100 reviews × ~3 minutes each = ~5 hours total**
- **Recommended**: 20-25 reviews per session
- **Sessions spread over 4-5 days** for consistency

---

## Commands

| Command | Action |
|---------|--------|
| [Enter] | Label current review |
| [s] | Skip current review |
| [q] | Quit and save progress |
| [b] | Go back to previous review |
| [p] | Show progress summary |
| [?] | Show help |

Progress is saved automatically after each label.
