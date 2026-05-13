# 🚀 Setup Guide - AI Analytics Chatbot System

This guide walks you through setting up the chatbot system step by step.

## ⚡ Quick Setup (5 minutes)

### Step 1: Verify Installation
All Python packages are already installed. Verify by running:
```bash
cd d:\Dashboard Code\NO_WH\DS\chatbot_system
python -c "import fastapi, langchain, streamlit; print('✅ All packages installed')"
```

### Step 2: Configuration (Automatic!)

✅ **Your API keys are already configured!**

The system automatically loads from:
```
D:\Dashboard Code\NO_WH\DS\testing_openai\.streamlit\secrets.toml
```

**No .env file needed** - it will use your existing Century dashboard configuration.

**Optional:** If you want different API keys just for the chatbot:
```bash
copy .env.example .env
notepad .env
```

Then add your override credentials:
```env
# Optional overrides (if different from secrets.toml)
OPENAI_API_KEY=sk-proj-XXXXXXXXXX...
GEMINI_API_KEY=AIzaSyDXXXXXXXXX...

# Database (pre-configured for your system)
DB_HOST=localhost
DB_PORT=3307
DB_USER=postgres
DB_PASSWORD=hello
DB_NAME=salesdata

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
UI_PORT=8501

# Logging
LOG_LEVEL=INFO
```

### Step 3: Start the System

**Windows - One Click Start:**
```bash
start_all.bat
```

This will:
1. Check for `.env` configuration
2. Start FastAPI backend on port 8000
3. Start Streamlit frontend on port 8501
4. Open browser to http://localhost:8501

## 📋 Manual Setup (Detailed)

### Install Dependencies

```bash
# Navigate to project
cd "d:\Dashboard Code\NO_WH\DS"

# Install if not already installed
pip install fastapi uvicorn python-dotenv langchain langchain-openai langchain-google-genai psycopg2-binary pandas numpy scikit-learn streamlit requests
```

### Configure API Keys

1. **Get OpenAI API Key:**
   - Visit https://platform.openai.com/api-keys
   - Create new secret key
   - Copy to `.env`: `OPENAI_API_KEY=sk-proj-...`

2. **Get Google Gemini API Key (Optional):**
   - Visit https://aistudio.google.com/apikey
   - Create new API key
   - Copy to `.env`: `GEMINI_API_KEY=AIzaSy...`

3. **Database Already Configured:**
   - PostgreSQL running on localhost:3307
   - Database: `salesdata`
   - User: `postgres`
   - Password: `hello`

### Start Backend API

```bash
cd chatbot_system
python api.py
```

Expected output:
```
============================================================
🤖 AI Analytics Chatbot API
============================================================
🌐 Starting server on 0.0.0.0:8000
📊 API Docs: http://0.0.0.0:8000/docs
============================================================
INFO:     Uvicorn running on http://0.0.0.0:8000
✅ Database initialized successfully
✅ LLM Provider: OpenAI
```

### Start Frontend UI (New Terminal)

```bash
cd chatbot_system
streamlit run streamlit_ui.py
```

Expected output:
```
  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://192.168.1.X:8501
```

## 🧪 Testing

### 1. Check API Health
```bash
curl http://localhost:8000/health
```

Expected response:
```json
{
  "status": "✅ Healthy",
  "database": "✅ Connected",
  "timestamp": "2026-01-27T10:30:45Z"
}
```

### 2. Test Chat Endpoint
```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d {
    "question": "Show me top 5 shops by sales",
    "chat_history": []
  }
```

### 3. View API Documentation
Open in browser: http://localhost:8000/docs

This shows all available endpoints with testing interface.

### 4. Test UI
1. Open http://localhost:8501
2. Click the blue 💬 chat bubble
3. Type: "Show top 5 shops"
4. Press Enter

Expected: SQL is generated, executed, and insights are displayed.

## 🔧 Troubleshooting

### Problem: ".env file not found"

**Solution:**
```bash
cd chatbot_system
copy .env.example .env
# Edit .env with your API keys
```

### Problem: "OpenAI API key invalid"

**Solution:**
1. Check your API key at https://platform.openai.com/api-keys
2. Ensure it starts with `sk-proj-`
3. Update `.env`: `OPENAI_API_KEY=your-actual-key`
4. Restart API: `python api.py`

### Problem: "Database connection failed"

**Solution:**
```bash
# Check PostgreSQL is running
pg_isready -h localhost -p 3307

# Test connection
psql -h localhost -p 3307 -U postgres -d salesdata

# If fails, start PostgreSQL service
```

### Problem: "Port 8000 already in use"

**Solution:**
```bash
# Find process using port 8000
netstat -ano | findstr :8000

# Kill the process
taskkill /PID <process_id> /F

# Or use different port in .env
API_PORT=8001
```

### Problem: "Streamlit won't connect to API"

**Solution:**
```bash
# Ensure API is running first
python chatbot_system/api.py

# Check API is responding
curl http://localhost:8000/health

# If needed, restart API before UI
```

### Problem: "Slow responses"

**Solutions:**
1. Check database size: `SELECT pg_size_pretty(pg_database.datsize) FROM pg_database;`
2. Check database indexes: Run analysis in `kpi_app/INDEX_ANALYSIS.md`
3. Reduce query rows: Edit `database.py` MAX_ROWS = 100
4. Check internet for LLM latency: `curl -I https://api.openai.com`

## 📂 File Locations

```
chatbot_system/
├── config.py              ← Configuration management
├── database.py            ← PostgreSQL connection
├── llm_orchestrator.py    ← LangChain setup
├── data_analyzer.py       ← Analysis engine
├── api.py                 ← FastAPI server
├── streamlit_ui.py        ← Streamlit UI
├── .env                   ← Your API keys (create from .env.example)
├── .env.example           ← Template
├── start_all.bat          ← Start both services
├── start_api.bat          ← Start API only
├── start_ui.bat           ← Start UI only
└── README.md              ← Full documentation
```

## 🎯 Next Steps

### Basic Usage
1. Ask a data question: "Show top 5 overstock shops"
2. Get SQL generated and executed automatically
3. View insights and analysis
4. Download results as CSV if needed

### Advanced Usage
1. **Direct SQL**: Use `/api/query/execute` endpoint
2. **Validate Query**: Use `/api/query/validate` before execution
3. **Custom Analysis**: Create reusable query templates

### Production Deployment
1. Set `LOG_LEVEL=WARNING` in `.env`
2. Change `API_HOST` from `0.0.0.0` to specific IP
3. Add authentication (see `api.py` comments)
4. Use Docker for containerization
5. Set up monitoring and alerting

## 📊 Architecture Overview

```
┌──────────────────┐
│  Streamlit UI    │
│  (Port 8501)     │
└────────┬─────────┘
         │ HTTP/REST
         ▼
┌──────────────────────┐
│  FastAPI Backend     │
│  (Port 8000)         │
├─────────────────────┤
│ • Request handling   │
│ • Response shaping   │
│ • Error management   │
└────────┬─────────────┘
         │
    ┌────┼────┐
    ▼    ▼    ▼
   ┌──┐ ┌──┐ ┌──┐
   │LM│ │DB│ │AN│  LangChain, Database, Analysis
   └──┘ └──┘ └──┘
```

## 🔐 Security Notes

### Already Protected
- ✅ SQL injection prevention
- ✅ Only SELECT queries allowed
- ✅ Database connection pooling
- ✅ Error handling without exposing internals

### Recommended for Production
- Add API key authentication
- Use HTTPS/SSL
- Implement rate limiting
- Log all queries to audit trail
- Set up database backups
- Use environment-specific configs

## 📞 Support

If you encounter issues:

1. **Check Logs:**
   ```bash
   # API logs show in terminal running 'python api.py'
   # UI logs show in terminal running 'streamlit run streamlit_ui.py'
   ```

2. **Test Connection:**
   ```bash
   curl http://localhost:8000/health
   curl http://localhost:8501/
   ```

3. **Verify Configuration:**
   ```bash
   # Check .env file exists and has values
   # Not empty strings for API keys
   ```

4. **Database Check:**
   ```bash
   psql -h localhost -p 3307 -U postgres -c "SELECT 1"
   ```

---

**System Ready for Production Use!** 🚀

Once configured, the chatbot system will:
- ✅ Convert natural language to safe SQL
- ✅ Execute queries on your database
- ✅ Generate AI-powered insights
- ✅ Display results in beautiful chat interface
- ✅ Handle errors gracefully
- ✅ Log all operations for auditing
