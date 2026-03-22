# 🎬 Demo Creation Guide for ReviewInsight AI

## Quick Start Options (Choose One)

### 🥇 Option 1: Live Dashboard Demo (RECOMMENDED - Easiest)

**Deploy to Streamlit Cloud (Free)**
```bash
# 1. Create requirements.txt for Streamlit Cloud
cat > requirements.txt << EOF
streamlit==1.40.0
pandas==2.2.0
plotly==5.24.0
python-dotenv==1.0.0
openai==1.54.0
EOF

# 2. Create .streamlit/config.toml
mkdir -p .streamlit
cat > .streamlit/config.toml << EOF
[theme]
primaryColor = "#FF6B6B"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"
font = "sans serif"

[client]
showErrorDetails = false
maxUploadSize = 200
EOF

# 3. Push to GitHub
git add .
git commit -m "Add Streamlit dashboard"
git push

# 4. Deploy at https://streamlit.io/cloud
#    - Connect your GitHub repo
#    - Select main.py or dashboard/app.py
#    - Click "Deploy"

# 5. Add badge to README
# [![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-url.streamlit.app)
```

**Benefits**:
- ✅ Live, interactive demo
- ✅ Free hosting on Streamlit Cloud
- ✅ Always up-to-date
- ✅ No maintenance needed
- ✅ Works on mobile devices

---

### 🥈 Option 2: Screenshots (Good Alternative)

**Capture Screenshots with macOS Preview**

```bash
# 1. Start the dashboard
streamlit run dashboard/app.py

# 2. Take screenshots using macOS shortcuts:
#    - Full screen: Cmd + Shift + 3
#    - Selection: Cmd + Shift + 4
#    - Timed screen: Cmd + Shift + 5

# 3. Rename and organize screenshots
mkdir -p docs/screenshots
mv screenshot_1.png docs/screenshots/01-dashboard-overview.png
mv screenshot_2.png docs/screenshots/02-sentiment-trends.png
mv screenshot_3.png docs/screenshots/03-theme-distribution.png
mv screenshot_4.png docs/screenshots/04-high-risk-reviews.png
mv screenshot_5.png docs/screenshots/05-search-interface.png
```

**Add to README**:
```markdown
### 📸 Dashboard Screenshots

#### Main Dashboard
![Dashboard Overview](docs/screenshots/01-dashboard-overview.png)

#### Sentiment Analysis
![Sentiment Trends](docs/screenshots/02-sentiment-trends.png)

#### Theme Distribution
![Theme Distribution](docs/screenshots/03-theme-distribution.png)
```

**Tips for Great Screenshots**:
- Use sample data with realistic employee reviews
- Show variety in sentiment scores (1-5 stars)
- Include charts and visualizations
- Capture both light and dark themes
- Ensure text is readable at small sizes

---

### 🥉 Option 3: Video Demo (Most Impressive)

**Record with OBS Studio (Free, Cross-Platform)**

```bash
# 1. Install OBS Studio
brew install --cask obsstudio  # macOS
# or download from https://obsproject.com

# 2. Configure Settings
#    - Video: 1920x1080, 30fps
#    - Audio: Default microphone
#    - Output: MP4, medium quality

# 3. Create Scene Layout
#    - Display Capture: Full screen (1920x1080)
#    - Window Capture: Browser/Dashboard (1280x720)
#    - Text Overlay: "ReviewInsight AI Demo"

# 4. Script Your Demo (2-3 minutes max)

# 5. Record multiple takes, edit best parts

# 6. Add captions (recommended)
#    Use YouTube's auto-caption or Rev.com

# 7. Upload to YouTube (unlisted or public)
#    - Title: "ReviewInsight AI - Employee Sentiment Analysis Demo"
#    - Description: Add GitHub link and features
#    - Thumbnail: Use eye-catching screenshot

# 8. Embed in README
#    [![Demo Video](docs/screenshots/video-thumbnail.png)](https://www.youtube.com/watch?v=YOUR_VIDEO_ID)
```

**Demo Script Template**:
```
[0:00-0:15] INTRO
- "Hi, I'm [Your Name]"
- "This is ReviewInsight AI"
- "Let me show you what it can do"

[0:15-0:45] SETUP
- Show installation: pip install -r requirements.txt
- Show config: .env file setup
- Quick start command

[0:45-1:30] DASHBOARD TOUR
- Main KPI overview
- Upload a sample review
- Show real-time analysis

[1:30-2:15] KEY FEATURES
- Sentiment trends over time
- Theme distribution
- High-risk review detection
- Search functionality

[2:15-2:45] API & INTEGRATION
- Show API endpoint
- Demonstrate curl command
- Show response JSON

[2:45-3:00] OUTRO
- "Thanks for watching"
- "Link in description below"
- "Star us on GitHub!"
```

---

### 🎁 Option 4: GIFs (Great for Specific Features)

**Create GIFs with Gifski (macOS) or LICEcap (Cross-Platform)**

```bash
# macOS: Install Gifski
brew install gifski

# Record GIF (keep under 15 seconds, under 5MB)
gifski --input screen-recording.mov --output feature-demo.gif

# Alternative: LICEcap (cross-platform)
# Download from https://www.cockos.com/licecap/

# Recommended GIFs to create:
1. Review processing animation (5-10s)
2. Filter interaction (5-10s)
3. Search functionality (5-10s)
4. Export process (5-10s)
```

**Best Practices for GIFs**:
- Keep under 15 seconds
- File size under 5MB
- Focus on one feature at a time
- Use smooth, deliberate mouse movements
- Optimize colors for web

**Add to README**:
```markdown
### 🎞️ Feature Demos

#### Review Analysis
![Review Analysis](docs/gifs/review-analysis.gif)

#### Search & Filter
![Search Demo](docs/gifs/search-demo.gif)
```

---

## 🎯 My Recommendation: Start with Live Demo

**Why Live Demo is Best**:
1. **Easiest to set up** - Just deploy to Streamlit Cloud
2. **Most impressive** - Users can interact themselves
3. **Always current** - Updates automatically with your code
4. **Free hosting** - No cost to you
5. **Mobile friendly** - Works on all devices

**Quick Start (5 minutes)**:
```bash
# 1. Ensure your dashboard works locally
streamlit run dashboard/app.py

# 2. Create simple Procfile for deployment
echo "web: streamlit run dashboard/app.py --server.port $PORT" > Procfile

# 3. Push to GitHub (if not already)
git add . && git commit -m "Add dashboard" && git push

# 4. Go to https://streamlit.io/cloud
#    - Click "New app"
#    - Connect your GitHub repo
#    - Select dashboard/app.py
#    - Click "Deploy"

# 5. Get your URL and add to README!
```

---

## 📋 Complete Demo Checklist

### Phase 1: Live Demo (Today)
- [ ] Deploy dashboard to Streamlit Cloud
- [ ] Test all features on live site
- [ ] Add live demo badge to README
- [ ] Test on mobile device

### Phase 2: Screenshots (This Week)
- [ ] Take 5-7 high-quality screenshots
- [ ] Organize in `docs/screenshots/`
- [ ] Add captions and descriptions
- [ ] Embed in README with proper sizing

### Phase 3: Video Demo (Next Week)
- [ ] Write demo script (2-3 minutes)
- [ ] Record with OBS Studio
- [ ] Edit and add captions
- [ ] Upload to YouTube
- [ ] Embed in README

### Phase 4: GIFs (Optional)
- [ ] Create 3-4 feature-specific GIFs
- [ ] Optimize file sizes
- [ ] Add to README
- [ ] Consider for social media

---

## 🚀 Quick Win: Add Demo Badge Today

**Add this to your README right now**:

```markdown
# ReviewInsight AI

[![Live Demo](https://img.shields.io/badge/Demo-Live%20Demo-brightgreen.svg)](https://your-demo-url.streamlit.app)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-demo-url.streamlit.app)

**Live Demo**: [Try it now!](https://your-demo-url.streamlit.app) 🚀
```

**After deploying, replace the placeholder URL with your actual Streamlit Cloud URL.**

---

## 📞 Help & Resources

**Streamlit Documentation**: https://docs.streamlit.io
**Streamlit Community**: https://discuss.streamlit.io
**OBS Studio Guide**: https://www.youtube.com/watch?v=U6OGPiD1LyU
**Screenshot Tools**: https://support.apple.com/guide/mac-help/use-screen-shots-on-mac-mh26698/mac

---

**Start with the live demo - it's the fastest way to showcase your work!** 🎉
