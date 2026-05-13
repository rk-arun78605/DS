# 🎯 Quick Reference Card

## ⚡ Start Everything (1 Click)

**✅ API keys automatically loaded from your existing secrets.toml**

```bash
cd d:\Dashboard Code\NO_WH\DS\chatbot_system
start_all.bat
```

Opens:
- 🌐 Frontend: http://localhost:8501
- 📊 API Docs: http://localhost:8000/docs

## 🛠️ Manual Start (2 Terminals)

### Terminal 1: API Backend
```bash
cd chatbot_system
python api.py
```

### Terminal 2: Frontend UI
```bash
cd chatbot_system
streamlit run streamlit_ui.py
```

## 🔑 Configuration

### File: `chatbot_system/.env`

Required fields:
```env
OPENAI_API_KEY=sk-proj-...        # Get from https://platform.openai.com
GEMINI_API_KEY=AIzaSy...          # Get from https://aistudio.google.com
DB_HOST=localhost
DB_PORT=3307
DB_USER=postgres
DB_PASSWORD=hello
DB_NAME=salesdata
```

## 🧪 Health Checks

```bash
# API Health
curl http://localhost:8000/health

# API Docs
curl http://localhost:8000/docs

# Frontend Running
curl http://localhost:8501
```

## 💬 Example Queries

```
"Show top 5 overstock shops"
"Which items need replenishment?"
"What are sales by department?"
"List shops with low inventory"
"Give me inventory recommendations"
```

## 🔍 API Endpoints

### Chat
```bash
POST /api/chat
{
  "question": "Show top 5 shops",
  "chat_history": []
}
```

### Query Validate
```bash
POST /api/query/validate
{
  "sql": "SELECT * FROM mv_recommendations_complete LIMIT 10"
}
```

### Query Execute
```bash
POST /api/query/execute
{
  "sql": "SELECT * FROM mv_recommendations_complete LIMIT 10"
}
```

### Health
```bash
GET /health
```

## 📁 Project Structure

```
chatbot_system/
├── config.py              ← Settings
├── database.py            ← DB layer
├── llm_orchestrator.py    ← LLM chains
├── data_analyzer.py       ← Analysis
├── api.py                 ← FastAPI
├── streamlit_ui.py        ← Frontend
├── .env                   ← Your keys ⚙️
├── .env.example           ← Template
├── start_all.bat          ← Run everything
└── README.md              ← Full docs
```

## 🐛 Quick Fixes

| Problem | Fix |
|---------|-----|
| Port 8000 in use | `taskkill /PID <id> /F` |
| .env not found | `copy .env.example .env` |
| API key invalid | Check https://platform.openai.com |
| No database | `pg_isready -h localhost -p 3307` |
| Slow response | Check query in `/docs`, reduce rows |

## 🚀 First Run

1. Create `.env` from `.env.example`
2. Add your API keys
3. Run `start_all.bat`
4. Open http://localhost:8501
5. Click 💬 button
6. Ask: "Show top 5 shops"
7. Enjoy! 🎉

## 📊 Architecture

```
User Question
    ↓
Streamlit UI (Port 8501)
    ↓ HTTP POST /api/chat
FastAPI Backend (Port 8000)
    ↓
LangChain (SQL Generation)
    ↓
PostgreSQL Database
    ↓
Data Analyzer (Insights)
    ↓
Response to UI
```

## 🔒 Security

- ✅ SQL injection prevention
- ✅ Only SELECT allowed
- ✅ Connection pooling
- ✅ Error handling

## 📈 Performance

- Query timeout: 30 seconds
- Max rows: 200 per query
- Connection pool: 2-10 connections
- Response time: <2 seconds typical

## 💡 Tips

- Use `/docs` for API testing
- Check `curl http://localhost:8000/health` if UI won't connect
- Review logs in terminal for detailed errors
- Start API first, then UI

## 🎨 UI Features

- 💬 Floating chat bubble
- 📊 Data tables with download
- ✨ Gradient backgrounds
- 🎯 Sample query suggestions
- 📈 Insight boxes
- ⚡ Real-time chat history

## 🆘 Troubleshooting

### API won't start
```bash
# Check port
netstat -ano | findstr :8000

# Try different port
edit .env → API_PORT=8001
```

### UI won't load
```bash
# Restart both:
# 1. Kill api.py (Ctrl+C)
# 2. Kill streamlit (Ctrl+C)
# 3. Run start_all.bat again
```

### Database error
```bash
# Test connection
psql -h localhost -p 3307 -U postgres -c "SELECT 1"

# Check tables exist
psql -h localhost -p 3307 -U postgres -d salesdata -c "\dt"
```

---

**Built with ❤️ for Melcom Retail Analytics**
