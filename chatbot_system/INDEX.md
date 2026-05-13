# 📚 AI Chatbot System - Complete Documentation Index

## 🎯 Start Here

**New to the system?** Start with this order:

1. **[SYSTEM_SUMMARY.md](SYSTEM_SUMMARY.md)** (5 min read)
   - Overview of what was built
   - Architecture diagram
   - Quick start overview

2. **[QUICK_REFERENCE.md](QUICK_REFERENCE.md)** (2 min read)
   - One-page cheat sheet
   - Common commands
   - Health checks

3. **[SETUP_GUIDE.md](SETUP_GUIDE.md)** (10 min read)
   - Step-by-step configuration
   - Testing procedures
   - Manual setup if needed

4. **[TROUBLESHOOTING.md](TROUBLESHOOTING.md)** (as needed)
   - Solutions to common problems
   - Diagnostic commands
   - Emergency reset option

---

## 📖 Documentation Map

### Quick References
| Document | Purpose | Read Time |
|----------|---------|-----------|
| [QUICK_REFERENCE.md](QUICK_REFERENCE.md) | One-page cheat sheet | 2 min |
| [SYSTEM_SUMMARY.md](SYSTEM_SUMMARY.md) | Complete system overview | 5 min |
| [README.md](README.md) | Full documentation | 15 min |

### Setup & Deployment
| Document | Purpose | Read Time |
|----------|---------|-----------|
| [SETUP_GUIDE.md](SETUP_GUIDE.md) | Installation & configuration | 10 min |
| [TROUBLESHOOTING.md](TROUBLESHOOTING.md) | Problem solving | As needed |

### Source Code
| File | Purpose | Lines |
|------|---------|-------|
| [config.py](config.py) | Configuration management | 181 |
| [database.py](database.py) | PostgreSQL layer | 187 |
| [llm_orchestrator.py](llm_orchestrator.py) | LangChain orchestration | 412 |
| [data_analyzer.py](data_analyzer.py) | Analysis engine | 198 |
| [api.py](api.py) | FastAPI REST server | 289 |
| [streamlit_ui.py](streamlit_ui.py) | Streamlit frontend | 356 |

### Configuration
| File | Purpose |
|------|---------|
| [.env.example](.env.example) | Configuration template |
| [.env](SETUP_GUIDE.md) | Your actual config (create from .env.example) |

### Scripts
| Script | Purpose |
|--------|---------|
| [start_all.bat](start_all.bat) | Start everything (RECOMMENDED) |
| [start_api.bat](start_api.bat) | Start API only |
| [start_ui.bat](start_ui.bat) | Start UI only |

---

## 🚀 Getting Started (Choose Your Path)

### Path 1: 1-Click Start (RECOMMENDED)
```bash
cd d:\Dashboard Code\NO_WH\DS\chatbot_system
start_all.bat
```

**Done!** Browser opens to http://localhost:8501

**Requires:** `.env` file with API keys (see SETUP_GUIDE.md)

### Path 2: Manual Start
```bash
# Terminal 1: Start API
cd chatbot_system
python api.py

# Terminal 2: Start UI
cd chatbot_system
streamlit run streamlit_ui.py
```

### Path 3: Step-by-Step Setup
Follow [SETUP_GUIDE.md](SETUP_GUIDE.md) for detailed instructions

---

## 🎨 System Architecture

```
┌──────────────────────────────────────────────┐
│         Streamlit UI (Port 8501)             │
│  ┌──────────────────────────────────────┐    │
│  │  Chat Interface with Floating Bubble │    │
│  │  • Message display                   │    │
│  │  • Data visualization                │    │
│  │  • CSV download                      │    │
│  └──────────────────────────────────────┘    │
└─────────────────┬──────────────────────────┘
                  │ HTTP REST
                  ▼
┌──────────────────────────────────────────────┐
│       FastAPI Backend (Port 8000)            │
│  ┌──────────────────────────────────────┐    │
│  │  REST Endpoints                      │    │
│  │  • /api/chat                         │    │
│  │  • /api/query/validate               │    │
│  │  • /api/query/execute                │    │
│  │  • /health                           │    │
│  └──────────────────────────────────────┘    │
└─────────────────┬──────────────────────────┘
                  │
        ┌─────────┼──────────┐
        ▼         ▼          ▼
    ┌────────┐ ┌────────┐ ┌──────────┐
    │LLM Orch│ │Database│ │ Analysis │
    └────────┘ └────────┘ └──────────┘
```

---

## 💻 Key Components

### Frontend (Streamlit)
- **Location:** [streamlit_ui.py](streamlit_ui.py)
- **Port:** 8501
- **Purpose:** Modern chat interface
- **Features:** Floating bubble, avatars, data display

### Backend (FastAPI)
- **Location:** [api.py](api.py)
- **Port:** 8000
- **Purpose:** REST API server
- **Features:** Request validation, error handling, logging

### LLM Orchestration (LangChain)
- **Location:** [llm_orchestrator.py](llm_orchestrator.py)
- **Purpose:** SQL generation, insight generation
- **Features:** Multi-LLM support, fallback handling

### Database Layer
- **Location:** [database.py](database.py)
- **Purpose:** PostgreSQL connection management
- **Features:** Connection pooling, SQL safety, error handling

### Configuration Management
- **Location:** [config.py](config.py)
- **Purpose:** Centralized settings
- **Features:** Environment loading, defaults, validation

### Data Analysis
- **Location:** [data_analyzer.py](data_analyzer.py)
- **Purpose:** SQL result analysis
- **Features:** Statistics, trends, insights

---

## 🔧 Configuration

### Required API Keys

1. **OpenAI API Key** (https://platform.openai.com/api-keys)
   ```env
   OPENAI_API_KEY=sk-proj-...
   ```

2. **Google Gemini API Key** (https://aistudio.google.com/apikey) [Optional]
   ```env
   GEMINI_API_KEY=AIzaSy...
   ```

3. **Database Credentials** (Pre-configured)
   ```env
   DB_HOST=localhost
   DB_PORT=3307
   DB_USER=postgres
   DB_PASSWORD=hello
   DB_NAME=salesdata
   ```

**Setup:** See [SETUP_GUIDE.md](SETUP_GUIDE.md)

---

## 📊 API Documentation

### Interactive Docs
Open in browser: http://localhost:8000/docs

Shows all endpoints with:
- Parameter documentation
- Request/response examples
- Try-it-out functionality

### Main Endpoints

**POST /api/chat** - Chat with natural language
```json
Request: {"question": "Show top 5 shops", "chat_history": []}
Response: {"response": "...", "sql": "...", "insights": "..."}
```

**POST /api/query/validate** - Validate SQL before execution
```json
Request: {"sql": "SELECT * FROM table LIMIT 10"}
Response: {"valid": true, "message": "..."}
```

**POST /api/query/execute** - Execute SQL directly
```json
Request: {"sql": "SELECT * FROM table LIMIT 10"}
Response: {"data": [...], "row_count": 10}
```

**GET /health** - Check system health
```json
Response: {"status": "✅ Healthy", "database": "✅ Connected"}
```

---

## 🧪 Testing

### Quick Health Check
```bash
curl http://localhost:8000/health
```

### Test Chat
```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"question":"Show top 5 shops","chat_history":[]}'
```

### Test Database
```bash
psql -h localhost -p 3307 -U postgres -d salesdata -c "SELECT 1"
```

---

## 🐛 Troubleshooting

### Common Issues Quick Links

| Problem | Solution |
|---------|----------|
| .env not found | [SETUP_GUIDE.md](SETUP_GUIDE.md#step-2-create-configuration-file) |
| Port in use | [TROUBLESHOOTING.md](TROUBLESHOOTING.md#issue-2-port-8000-already-in-use) |
| API key invalid | [TROUBLESHOOTING.md](TROUBLESHOOTING.md#issue-4-openai-api-key-invalid) |
| Database error | [TROUBLESHOOTING.md](TROUBLESHOOTING.md#issue-5-database-connection-failed) |
| UI won't connect | [TROUBLESHOOTING.md](TROUBLESHOOTING.md#issue-6-api-running-but-ui-wont-connect) |
| Slow responses | [TROUBLESHOOTING.md](TROUBLESHOOTING.md#issue-8-slow-responses-or-query-times-out) |

Full guide: [TROUBLESHOOTING.md](TROUBLESHOOTING.md)

---

## 📈 Performance

- **Connection Pool:** 2-10 connections (reused)
- **Query Timeout:** 30 seconds max
- **Response Time:** <2 seconds typical
- **Data Limit:** 200 rows max (auto-limit)
- **Chat History:** Session-based, auto-cleared on restart

---

## 🔒 Security Features

✅ SQL injection prevention
✅ SELECT-only queries
✅ Connection pooling
✅ Error handling (no internal details exposed)
✅ Query validation
✅ Automatic LIMIT enforcement

---

## 📝 File Structure

```
chatbot_system/
├── 📄 Documentation
│   ├── README.md                 ← Full documentation
│   ├── SETUP_GUIDE.md            ← Installation guide
│   ├── QUICK_REFERENCE.md        ← One-page reference
│   ├── SYSTEM_SUMMARY.md         ← System overview
│   ├── TROUBLESHOOTING.md        ← Problem solutions
│   └── INDEX.md                  ← This file
│
├── 🐍 Python Code
│   ├── config.py                 ← Configuration
│   ├── database.py               ← PostgreSQL layer
│   ├── llm_orchestrator.py       ← LangChain setup
│   ├── data_analyzer.py          ← Analysis engine
│   ├── api.py                    ← FastAPI server
│   └── streamlit_ui.py           ← Frontend UI
│
├── ⚙️ Configuration
│   ├── .env.example              ← Template
│   └── .env                      ← Your config (create this)
│
└── 🚀 Startup Scripts
    ├── start_all.bat             ← Start everything
    ├── start_api.bat             ← Start API only
    └── start_ui.bat              ← Start UI only
```

---

## 🎯 Quick Tasks

### "I want to start using the chatbot"
1. Read: [SYSTEM_SUMMARY.md](SYSTEM_SUMMARY.md) (5 min)
2. Do: [SETUP_GUIDE.md](SETUP_GUIDE.md) (10 min)
3. Run: `start_all.bat`
4. Use: http://localhost:8501

### "I have an error/problem"
1. Read: [TROUBLESHOOTING.md](TROUBLESHOOTING.md)
2. Find your issue
3. Follow solution

### "I want to understand the code"
1. Read: [SYSTEM_SUMMARY.md](SYSTEM_SUMMARY.md) (architecture)
2. Review: [api.py](api.py) (REST endpoints)
3. Study: [llm_orchestrator.py](llm_orchestrator.py) (LLM chains)
4. Examine: [database.py](database.py) (DB layer)

### "I want to deploy to production"
1. Complete: [SETUP_GUIDE.md](SETUP_GUIDE.md)
2. Review: Security section in [README.md](README.md)
3. Setup: Monitoring & logging
4. Test: All endpoints in `/docs`

---

## 🆘 Support Resources

### Documentation
- **Complete Guide:** [README.md](README.md)
- **Setup Instructions:** [SETUP_GUIDE.md](SETUP_GUIDE.md)
- **Quick Reference:** [QUICK_REFERENCE.md](QUICK_REFERENCE.md)
- **Troubleshooting:** [TROUBLESHOOTING.md](TROUBLESHOOTING.md)
- **System Overview:** [SYSTEM_SUMMARY.md](SYSTEM_SUMMARY.md)

### Tools
- **API Documentation:** http://localhost:8000/docs (after starting)
- **Health Check:** http://localhost:8000/health
- **Frontend:** http://localhost:8501

### Debugging
- Check terminal logs (both API and UI)
- Review API docs at `/docs` endpoint
- Run diagnostic commands in [TROUBLESHOOTING.md](TROUBLESHOOTING.md)

---

## ✅ Verification Checklist

After setup, verify these work:

- [ ] `.env` file exists with API keys
- [ ] `python api.py` starts without errors
- [ ] `streamlit run streamlit_ui.py` opens UI
- [ ] http://localhost:8000/health returns JSON
- [ ] http://localhost:8000/docs shows API docs
- [ ] http://localhost:8501 shows chat interface
- [ ] Chat bubble is clickable and expands
- [ ] Message input works
- [ ] Example query "Show top 5 shops" works

---

## 🎉 You're All Set!

Your AI Analytics Chatbot System is ready to use.

### Next Steps:
1. Ensure `.env` is configured with your API keys
2. Run: `start_all.bat`
3. Open: http://localhost:8501
4. Start asking questions!

### For Help:
- Check [TROUBLESHOOTING.md](TROUBLESHOOTING.md)
- Review [SETUP_GUIDE.md](SETUP_GUIDE.md)
- Read [README.md](README.md)

---

**Built with ❤️ for Melcom Retail Analytics**

*Professional AI-Powered Analytics System*
