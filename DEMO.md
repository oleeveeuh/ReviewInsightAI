# 🎬 Demo Setup Guide

## Quick Start (5 minutes)

### 1. Deploy Live Dashboard (Recommended)

```bash
# Test locally first
streamlit run dashboard/app.py

# Deploy to Streamlit Cloud
# 1. Go to https://streamlit.io/cloud
# 2. Connect your GitHub repo
# 3. Select dashboard/app.py
# 4. Click "Deploy"

# Add badge to README
[![Live Demo](https://img.shields.io/badge/Demo-Live%20Demo-brightgreen.svg)](https://your-app.streamlit.app)
```

### 2. Take Screenshots

```bash
# Create screenshots directory
mkdir -p docs/screenshots

# Start dashboard
streamlit run dashboard/app.py

# Take screenshots (macOS: Cmd+Shift+3/4, Windows: Win+Shift+S)
# Recommended shots:
# - Main dashboard overview
# - Sentiment trend charts
# - Theme distribution plots
# - High-risk reviews table
# - Search interface
```

### 3. Record Video Demo (Optional)

```bash
# Install OBS Studio (free)
brew install --cask obsstudio  # macOS
# or download from https://obsproject.com

# Record 2-3 minute demo covering:
# - Installation (30s)
# - Dashboard tour (1min)
# - Key features (1min)
# - API usage (30s)

# Upload to YouTube and embed in README
```

## What to Demonstrate

### 🏆 Key Results
- **91.95% Theme F1 Score** with v3.0 Few-Shot
- **2,500+ reviews** across 4 platforms
- **Cross-validation framework** for unbiased evaluation

### 🎯 Features to Show
- Upload and analyze reviews instantly
- Sentiment trend visualization
- Theme distribution charts
- High-risk employee detection
- Full-text search with highlighting

## Deployment Files

The following files are ready for Streamlit Cloud deployment:
- `.streamlit/config.toml` - Dashboard configuration
- `requirements-streamlit.txt` - Dependencies
- `Procfile` - Cloud deployment config

## Tips

- Keep video demos under 3 minutes
- Use high-quality screenshots
- Add captions to videos
- Test demo on mobile devices
- Always test before deploying

## Resources

- [Streamlit Cloud](https://streamlit.io/cloud)
- [OBS Studio](https://obsproject.com)
- [Streamlit Documentation](https://docs.streamlit.io)
