# FutureLens Quick Start Guide

## 🚀 Get Started in 5 Minutes

### Step 1: Set Up Environment Variables
```bash
# Create or edit .env file in the project root
GROQ_API_KEY=sk-xxxxxxxx...
GEMINI_API_KEY=xxxxxxxxxxxvx...
```

### Step 2: Install Dependencies (if needed)
```bash
cd e:\FutureLens_final\FutureLens
pip install -r requirements.txt
```

### Step 3: Start Backend
```bash
# In terminal/PowerShell
uvicorn main:app --reload
```
- Backend starts at: `http://localhost:8000`
- Docs available at: `http://localhost:8000/docs`

### Step 4: Start Frontend (new terminal)
```bash
# In new terminal/PowerShell
streamlit run app.py
```
- Frontend opens at: `http://localhost:8501`

## 🧪 Testing the 4 Main Fixes

### Test 1: Chart Rendering & Data Labels
**Goal:** Verify charts show all data with values

```
1. In Streamlit app, click "📊 Single/Cross-Sectional Analysis" (Tab 1)
2. Click "Demo Mode" or upload sample_data.csv
3. Select different chart types:
   - Bar Chart
   - Scatter Plot
   - Pie Chart
   - Line Chart
   - Histogram
4. Verify for each chart:
   ✓ ALL data points are shown (not just one)
   ✓ Values appear as labels on each point
   ✓ Hover shows detailed information
```

**What to look for:**
- Bar chart: Every bar has a value label on top
- Scatter: Every point has a value label nearby
- Pie: Every slice has a percentage label
- Line: Every point has a value label
- Histogram: Every bar shows count

### Test 2: Forecast Values on Graph
**Goal:** Verify forecast chart displays all data values

```
1. In "📊 Single/Cross-Sectional Analysis" tab
2. Upload sample_data.csv or use Demo Mode
3. Scroll down to "Forecast Chart"
4. Verify:
   ✓ Historical data points have value labels
   ✓ Forecast points have value labels
   ✓ Red anomaly markers show values
   ✓ Confidence bands (blue shading) visible
   ✓ Hover shows detailed information
```

**Expected output:**
- Historical line: Blue with dots, each labeled
- Forecast line: Orange with dots, each labeled
- Confidence bands: Light blue area around forecast
- Anomalies: Red dots/triangles with markers

### Test 3: ~10 Day Forecasting Horizon
**Goal:** Verify forecasting extends ~10 periods

```
1. Check the forecast chart
2. Count periods from now into the future:
   - Should be ~10 periods
   - For weekly data: ~10 weeks = ~70 days
   - For daily data: ~10 days
3. Hover over forecast points to see dates:
   ✓ Extends 10 periods ahead
   ✓ Dates are correctly formatted
   ✓ Confidence ranges are reasonable
```

**What to look for:**
- Last historical point: Today or most recent date
- Forecast extends: ~10 periods into future
- Confidence bands: Widen as you go further (normal)
- Values are reasonable: Similar magnitude to historical

### Test 4: Chat with Groq Agent
**Goal:** Verify chat works with new Groq architecture

```
1. Click "💬 Chat" (Tab 4)
2. Upload data if prompted
3. Try these questions:
   - "What will sales look like next month?"
   - "Are there any concerning trends?"
   - "Which product line is performing best?"
4. Verify:
   ✓ Response appears within 2-3 seconds
   ✓ Response is 2-4 sentences max
   ✓ Includes specific numbers/metrics
   ✓ Makes actionable recommendation
   ✓ Suggests follow-up question
```

**Check backend logs:**
```
Chat query: What will sales look like next month?
Groq result: groq_agent - answer: Next 4 weeks...
```

**Look for these in response:**
- Direct answer to your question
- Specific numbers (e.g., "6.2% growth")
- Key insight (e.g., "seasonal spike in week 3")
- Next step (e.g., "monitor ad spend")
- Follow-up question (e.g., "Would you like...")

## 📊 Testing with Sample Data

### Demo Mode (Easiest)
```
1. Tab 1: Click "Demo Mode"
2. Automatically loads sample_data.csv
3. Scroll through all sections
4. Try each chart type
5. Go to Chat tab and ask questions
```

### Using sample_data.csv
```
1. Go to Tab 1
2. Click "Upload a CSV file"
3. Select: data/sample_data.csv
4. Wait for analysis to complete (30-60 seconds)
5. Charts and forecasts appear below
```

## 🐛 Troubleshooting

### Chat not showing responses
**Problem:** Empty response or "Please ask a question"
- Solution: Upload CSV first in Tab 1
- Solution: Check GROQ_API_KEY is set
- Solution: Check internet connection

### Charts showing error
**Problem:** "Column must be numeric" or chart won't render
- Solution: Select numeric column for Y-axis
- Solution: Check CSV has at least one numeric column
- Solution: Reload page and try again

### Forecasting shows only historical data
**Problem:** No forecast line on chart
- Solution: Check data has at least 20 data points
- Solution: Check date column is recognized
- Solution: Try demo mode first

### Backend not starting
**Problem:** Port 8000 already in use
```bash
# Kill existing process and restart
netstat -ano | findstr :8000
taskkill /PID <PID> /F
uvicorn main:app --reload
```

### Streamlit connection refused
**Problem:** Cannot connect to backend
- Solution: Make sure backend is running (`uvicorn main:app --reload`)
- Solution: Check backend is at http://localhost:8000
- Solution: Refresh browser page

## 📈 Expected Results

### Good Signs ✅
- Charts fill entire space with all data points
- Every data point has a value label
- Forecast extends ~10 periods into future
- Chart has confidence bands (blue shading)
- Chat responds with specific metrics and recommendations
- Response appears in 2-3 seconds
- Suggested questions appear below response

### Warning Signs ⚠️
- Only one data point appears on chart
- No value labels on data points
- Forecast is only 3-4 periods out
- Chat takes 10+ seconds to respond
- Response is very long (more than 5 sentences)
- No suggested questions

## 💡 Pro Tips

1. **Demo Mode First:** Start with demo mode to see everything working
2. **Check Logs:** Backend logs show what Groq is doing
3. **Try Different Questions:** Chat gets better with variety
4. **Watch Hover:** Hover over chart points to see full details
5. **Zoom Chart:** Click and drag on chart to zoom
6. **Save Screenshots:** Use Print Screen to save good results

## 🎯 Next Steps

After testing:
1. Try your own CSV data
2. Experiment with different questions in chat
3. Check sensitivity to chart type changes
4. Verify forecasts match your domain knowledge
5. Review suggested questions for usefulness

## 📞 Still Having Issues?

1. Check FIXES_APPLIED.md for detailed changes
2. See ARCHITECTURE.md for system design
3. Review error logs in terminal
4. Check that all dependencies are installed
5. Try demo mode first to verify basics work

---

**Ready to test?** Start with backend + frontend, then test each of the 4 features above!

**Questions?** See ARCHITECTURE.md for deep technical details or FIXES_APPLIED.md for what changed.
