# 🤖 AI Analytics Chatbot System

A professional, production-ready chatbot system that converts natural language questions into SQL queries, executes them, and provides AI-powered insights.

## ✨ Features

### Core Capabilities
- **Natural Language to SQL**: Uses LLM to convert questions to safe SQL queries
- **Multi-LLM Support**: OpenAI, Google Gemini, Perplexity with automatic fallback
- **Data Analysis**: Comprehensive analysis of query results with statistics
- **Insights Generation**: AI-powered insights from data
- **Professional UI**: Modern Streamlit interface with gradient styling
- **REST API**: FastAPI backend for scalability

### Architecture

```
┌─────────────────────────────────────────────────────────┐
│         Streamlit UI (Port 8501)                        │
│  - Chat Interface with Avatar & Message Cards           │
│  - Data Visualization & Download                        │
│  - Query History & Sample Queries                       │
└────────────────────┬────────────────────────────────────┘
                     │ HTTP Requests
                     ▼
┌─────────────────────────────────────────────────────────┐
│         FastAPI Backend (Port 8000)                     │
│  - REST API Endpoints                                   │
│  - Request Validation                                   │
│  - Error Handling                                       │
└────────────────────┬────────────────────────────────────┘
                     │
        ┌────────────┼────────────┐
        ▼            ▼            ▼
    ┌────────┐  ┌─────────────┐  ┌──────────┐
    │  LLM   │  │  Data Layer │  │ Analysis │
    │ Orch.  │  │ (Database)  │  │ Engine   │
    └────────┘  └─────────────┘  └──────────┘
```

## 🚀 Quick Start

### 1. Install Dependencies

All required packages are already installed:
```bash
pip install fastapi uvicorn python-dotenv langchain langchain-openai langchain-google-genai
```

### 2. Configure Environment

Copy `.env.example` to `.env` and update with your API keys:

```bash
cp chatbot_system/.env.example chatbot_system/.env
```

Edit `.env`:
```env
# LLM Keys (at least one required)
OPENAI_API_KEY=sk-proj-your-key-here
GEMINI_API_KEY=your-gemini-key-here

# Database (already configured for your setup)
DB_HOST=localhost
DB_PORT=3307
DB_USER=postgres
DB_PASSWORD=hello
DB_NAME=salesdata
### 2. API Keys (Automatic!)

✅ **API keys are automatically loaded from:**
```
D:\Dashboard Code\NO_WH\DS\testing_openai\.streamlit\secrets.toml
```

Your existing OpenAI, Gemini, and Perplexity keys (from the Century dashboard) are automatically used by the chatbot!

**Optional:** To use different keys just for the chatbot, create a `.env` file:
```bash
copy .env.example .env
```

And set your override keys:
```env
# Optional - only if different from secrets.toml
OPENAI_API_KEY=sk-proj-your-key-here
GEMINI_API_KEY=your-gemini-key-here

# Database (already configured)
DB_HOST=localhost
DB_PORT=3307
DB_USER=postgres
DB_PASSWORD=hello
DB_NAME=salesdata
```
```

### 3. Start Backend API

```bash
cd chatbot_system
python api.py
```

Expected output:
```
✅ Database connection pool created
✅ OpenAI LLM initialized
🌐 Starting server on 0.0.0.0:8000
```

### 4. Start Frontend (in new terminal)

```bash
cd chatbot_system
streamlit run streamlit_ui.py
```

Browser opens at: `http://localhost:8501`

## 📁 Project Structure

```
chatbot_system/
├── config.py              # Configuration management
├── database.py            # PostgreSQL connection pool
├── llm_orchestrator.py    # LangChain LLM coordination
├── data_analyzer.py       # Data analysis & insights
├── api.py                 # FastAPI application
├── streamlit_ui.py        # Streamlit frontend
├── start_api.bat          # Start backend script
├── start_ui.bat           # Start frontend script
├── .env.example           # Environment template
└── README.md              # This file
```

## 🔧 Components

### 1. **Config Module** (`config.py`)
- Centralized configuration management
- Environment variable loading with defaults
- Feature flags and safety settings
- Database and LLM credentials

### 2. **Database Module** (`database.py`)
- PostgreSQL connection pooling
- Query execution with pandas integration
- Automatic resource cleanup
- Error handling and logging

### 3. **LLM Orchestrator** (`llm_orchestrator.py`)
- Multi-LLM support (OpenAI, Gemini, Perplexity)
- Automatic fallback between providers
- SQL generation from natural language
- SQL sanitization and validation
- Insight generation from data

### 4. **Data Analyzer** (`data_analyzer.py`)
- Statistical analysis of results
- Query type detection
- Business metrics calculation
- Report generation
- Data formatting for display

### 5. **API Layer** (`api.py`)
- FastAPI REST endpoints
- Chat endpoint: `/api/chat` (POST)
- Query validation: `/api/query/validate` (POST)
- Direct query execution: `/api/query/execute` (POST)
- Insight generation: `/api/insights/generate` (POST)
- Health check: `/health` (GET)
- CORS enabled for cross-origin requests

### 6. **Streamlit UI** (`streamlit_ui.py`)
- Modern chat interface with gradients
- Real-time chat history
- Data visualization with pandas
- CSV download functionality
- Sample query suggestions
- API status monitoring

## 💬 Chat Examples

Try these questions:

### Data Queries (SQL Generated)
```
"Show top 5 overstock shops"
"Which items need replenishment?"
"What are sales by department?"
"List shops with low inventory"
"Find items expiring soon"
```

### Insight Queries (LLM Only)
```
"What should we prioritize?"
"How is our business doing?"
"Give me recommendations"
```

## 🔐 Security Features

### SQL Safety
- ✅ Blocks: INSERT, UPDATE, DELETE, DROP, ALTER, TRUNCATE
- ✅ SELECT queries only
- ✅ Automatic LIMIT enforcement (max 200 rows)
- ✅ Whitelist of safe tables
- ✅ SQL injection prevention

### Data Protection
- ✅ Connection pooling (reuse, no resource leaks)
- ✅ Automatic transaction management
- ✅ Error handling without exposing internals
- ✅ Comprehensive logging

## 📊 API Endpoints

### POST `/api/chat`
Chat with natural language questions

**Request:**
```json
{
  "question": "Show top 5 overstock shops",
  "chat_history": []
}
```

**Response:**
```json
{
  "response": "✅ Found 5 results",
  "insights": "• Shop A has 150% overstocking...",
  "data": [...],
  "row_count": 5,
  "sql_used": "SELECT * FROM..."
}
```

### GET `/health`
Check API and database health

**Response:**
```json
{
  "status": "✅ Healthy",
  "database": "✅ Connected",
  "llm_available": true
}
```

### POST `/api/query/validate`
Validate SQL without executing

### POST `/api/query/execute`
Execute SQL query directly

## 🎨 UI Styling

The interface features:
- 🎨 Gradient backgrounds (purple to pink)
- 💬 Chat bubbles with avatars
- 📊 Data tables with shadows
- 🎯 Insight boxes with highlights
- 📥 Download buttons
- ⚡ Smooth animations

## 🔄 Processing Flow

```
User Question
    ↓
[Detect Query Type]
    ├─→ Data Query (SQL) ──→ [Generate SQL] ──→ [Execute]
    │                              ↓               ↓
    │                        [Validate]      [Analyze Data]
    │                                             ↓
    └─→ Insight Query (LLM) ────────────→ [Generate Insights]
                                                  ↓
                                          [Format Response]
                                                  ↓
                                          Return to User
```

## 📈 Performance

- **Connection Pool**: Reuses 2-10 connections
- **Query Timeout**: 30 seconds max
- **Response Cache**: Latest 10 queries
- **Data Limit**: Max 200 rows per query (configurable)
- **API Response**: <2s for typical queries

## 🐛 Troubleshooting

### API won't start
```bash
# Check if port 8000 is in use
netstat -ano | findstr :8000

# Kill process using port
taskkill /PID <PID> /F

# Restart API
python api.py
```

### Database connection failed
```bash
# Check PostgreSQL running
pg_isready -h localhost -p 3307

# Test connection
psql -h localhost -p 3307 -U postgres -d salesdata
```

### LLM not working
- Check `.env` file has API keys
- Test API keys manually
- Check internet connection
- Review logs: `python api.py` shows LLM initialization

### Streamlit won't connect to API
```bash
# Ensure API is running first
python chatbot_system/api.py

# Check API health
curl http://localhost:8000/health
```

## 📝 Logging

Both API and frontend log to console with timestamps:

```
2026-01-27 10:30:45 - llm_orchestrator - INFO - ✅ SQL generated by OpenAI
2026-01-27 10:30:46 - database - INFO - ✅ Query executed: 5 rows returned
2026-01-27 10:30:47 - data_analyzer - INFO - ✅ Analysis complete: 5 rows, 8 columns
```

## 🚀 Deployment

### Local Development
```bash
# Terminal 1: Start API
python chatbot_system/api.py

# Terminal 2: Start UI
streamlit run chatbot_system/streamlit_ui.py
```

### Docker (Optional)
Create `Dockerfile` in `chatbot_system/`:
```dockerfile
FROM python:3.13
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
EXPOSE 8000 8501
CMD ["sh", "-c", "python api.py & streamlit run streamlit_ui.py"]
```

## 📚 Dependencies

- **FastAPI**: Web framework for API
- **Uvicorn**: ASGI server
- **LangChain**: LLM orchestration
- **langchain-openai**: OpenAI integration
- **langchain-google-genai**: Google Gemini integration
- **psycopg2-binary**: PostgreSQL driver
- **pandas**: Data manipulation
- **Streamlit**: Frontend UI
- **python-dotenv**: Environment management

## 📄 License

Proprietary - Melcom Retail Analytics

## 👨‍💻 Support

For issues or questions, check:
1. Logs: Console output shows detailed errors
2. API Health: `curl http://localhost:8000/health`
3. Database: Verify PostgreSQL is running
4. LLM Keys: Check `.env` file configuration

---

**Built with ❤️ for professional analytics**
