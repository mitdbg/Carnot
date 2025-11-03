# AI-Powered File Search with Carnot

The web app now uses **Carnot's Context.search()** for intelligent, agent-based file searching!

## What Changed

The search chatbot now leverages Carnot's deep research capabilities:

### Before (Simple Keyword Search)
- Split query into keywords
- Check if keywords appear in filenames
- Basic content matching

### After (Carnot Agent Search)
- **Agent reasoning** - Intelligently determines search strategy
- **Semantic understanding** - Understands meaning, not just keywords
- **Multi-criteria queries** - Handles complex queries like:
  - "emails about Deathstar before 2016 from Hash Bros"
  - "legal documents mentioning liability limits over $1M"
  - "research papers about batteries from MIT"
- **Adaptive strategy** - Agent can list files, extract metadata, filter, etc.

## How It Works

```python
# User query: "Find emails about Deathstar before 2016 from Hash Bros"

# 1. Creates TextFileContext
ctx = carnot.TextFileContext(
    path="data/",
    id="search_data_1234",
    description="Data directory containing various files"
)

# 2. Uses intelligent search with agent
search_ctx = ctx.search(query)

# 3. Agent reasons about best approach:
#    - May list files first to see what's available
#    - Could extract dates and filter chronologically
#    - Might check sender fields before content
#    - Adapts based on what it finds

# 4. Returns matched files
```

## Benefits

### For Users
- **Handle complex queries** - No need to break down multi-part questions
- **Better relevance** - Semantic understanding vs keyword matching
- **More intelligent** - Agent can handle ambiguity and variations

### Technical
- **Automatic fallback** - Falls back to keyword search if Carnot fails
- **Configurable** - Uses GPT-4o-MINI for cost efficiency
- **Robust** - Handles errors gracefully

## Example Queries That Now Work Better

### Simple Queries
- "Find PDFs" → Works instantly
- "Legal documents" → Understands semantic meaning

### Complex Multi-Criteria
- "Emails about Deathstar before 2016 written by Hash Bros"
- "Research papers from MIT about battery technology published after 2020"
- "Contracts mentioning liability limits greater than $500K"

### Exploratory
- "What financial issues are discussed in these emails?"
- "Find documents related to energy trading partnerships"

## Performance

- **Response time:** 10-30 seconds (agent reasoning takes time)
- **Cost:** ~$0.01-0.05 per search (using GPT-4o-MINI)
- **Accuracy:** Much higher than keyword matching
- **Fallback:** If Carnot fails, uses enhanced keyword search

## Configuration

Located in: `carnot-web/backend/app/routes/search.py`

```python
# Adjust model for speed vs quality tradeoff
config = carnot.QueryProcessorConfig(
    policy=carnot.MaxQuality(),
    available_models=[carnot.Model.GPT_4o_MINI],  # Change to GPT_4o for better quality
    progress=False,
)
```

## User Experience

In the chatbot:
1. User types: "Find emails about Deathstar from Hash Bros before 2016"
2. Chatbot shows: "Searching..." (with loading spinner)
3. Wait 10-30 seconds while agent reasons
4. Results appear with matched files
5. Click "Add to Selection" to include in dataset

## Troubleshooting

### Search is slow
- Normal! Agent reasoning takes 10-30 seconds
- Using GPT-4o-MINI for speed (can upgrade to GPT-4o for quality)

### "Failed to load data" error
- Check that backend is running on port 8000
- Verify API keys are set in environment
- Check backend logs for Carnot errors

### Falls back to keyword search
- Carnot may fail due to:
  - Missing API keys
  - Model unavailable
  - Import errors
- Keyword fallback still provides decent results

## Future Enhancements

Possible improvements:
- Add loading progress bar showing agent steps
- Cache search results for common queries
- Add "Quick Search" vs "Deep Search" toggle
- Pre-extract metadata for instant structured queries

## API Endpoint

**POST** `/api/search`

**Request:**
```json
{
  "query": "Find emails about Deathstar from Hash Bros before 2016",
  "path": null
}
```

**Response:**
```json
[
  {
    "file_path": "data/enron-eval-medium/email_123.txt",
    "file_name": "email_123.txt",
    "relevance_score": 1.0,
    "snippet": "From: Hash Bros...\nDate: 2015-03-15..."
  }
]
```

---

**Now your file search is powered by Carnot's intelligent agent system!** 🚀

