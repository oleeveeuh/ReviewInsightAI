#!/bin/bash

echo "🚀 ReviewInsight AI - Quick Demo Deployment"
echo "=========================================="
echo ""

# Check if we're in the right directory
if [ ! -f "dashboard/app.py" ]; then
    echo "❌ Error: Please run this script from the project root directory"
    exit 1
fi

echo "✅ Found dashboard at dashboard/app.py"
echo ""

# Create deployment files
echo "📁 Creating deployment files..."

# Create .streamlit directory
mkdir -p .streamlit

# Create Streamlit config
cat > .streamlit/config.toml << 'EOF'
[theme]
primaryColor = "#FF6B6B"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"
font = "sans serif"

[client]
showErrorDetails = false
maxUploadSize = 200

[browser]
gatherUsageStats = false
serverAddress = "localhost"
serverPort = 8501
EOF

echo "✅ Created .streamlit/config.toml"

# Create requirements.txt for Streamlit Cloud
cat > requirements-streamlit.txt << 'EOF'
streamlit==1.40.0
pandas==2.2.0
plotly==5.24.0
python-dotenv==1.0.0
openai==1.54.0
scipy==1.13.0
numpy==1.26.0
EOF

echo "✅ Created requirements-streamlit.txt"

# Create Procfile
cat > Procfile << 'EOF'
web: streamlit run dashboard/app.py --server.port $PORT --server.headless true
EOF

echo "✅ Created Procfile"

echo ""
echo "🎯 Deployment files created!"
echo ""

# Test dashboard locally
echo "🧪 Testing dashboard locally..."
echo "Starting dashboard at http://localhost:8501"
echo "Press Ctrl+C to stop"
echo ""

# Start dashboard in background
streamlit run dashboard/app.py --server.headless true &

# Get the process ID
DASHBOARD_PID=$!

echo "✅ Dashboard started (PID: $DASHBOARD_PID)"
echo ""
echo "📸 Next Steps:"
echo "1. Open http://localhost:8501 in your browser"
echo "2. Take screenshots of key features:"
echo "   - Main dashboard overview"
echo "   - Sentiment trend charts"
echo "   - Theme distribution plots"
echo "   - Search interface"
echo ""
echo "3. When ready to deploy:"
echo "   a. Add files to git:"
echo "      git add .streamlit/ Procfile requirements-streamlit.txt"
echo "   b. Commit changes:"
echo "      git commit -m 'Add Streamlit deployment files'"
echo "   c. Push to GitHub:"
echo "      git push"
echo "   d. Deploy to Streamlit Cloud:"
echo "      - Go to https://streamlit.io/cloud"
echo "      - Click 'New app'"
echo "      - Connect your GitHub repo"
echo "      - Select dashboard/app.py"
echo "      - Click 'Deploy'"
echo ""
echo "4. Add badge to README:"
echo "   [![Live Demo](https://img.shields.io/badge/Demo-Live%20Demo-brightgreen.svg)](https://your-app-url.streamlit.app)"
echo ""

# Wait for user input
read -p "Press Enter to stop the dashboard and continue..."

# Stop the dashboard
kill $DASHBOARD_PID

echo ""
echo "✅ Dashboard stopped"
echo ""
echo "📚 Documentation created:"
echo "   - DEMO_CREATION_GUIDE.md (comprehensive guide)"
echo "   - DEMO_CHECKLIST.md (step-by-step checklist)"
echo "   - EXPERIMENTS_AND_DEMO_SUMMARY.md (quick reference)"
echo ""
echo "🎉 Good luck with your demo!"
echo ""
