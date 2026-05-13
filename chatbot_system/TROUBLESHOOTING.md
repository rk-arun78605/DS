# 🔧 Troubleshooting Guide (Windows)

## 🚨 Common Issues & Solutions

### Issue 1: ".env file not found" when running start_all.bat

**Symptoms:**
```
❌ ERROR: .env file not found!
Please create chatbot_system\.env from chatbot_system\.env.example
```

**Solution:**
```bash
# Navigate to chatbot_system directory
cd d:\Dashboard Code\NO_WH\DS\chatbot_system

# Copy example file
copy .env.example .env

# Edit with your API keys
notepad .env
```

Required keys:
```env
OPENAI_API_KEY=sk-proj-your-key-here
GEMINI_API_KEY=your-gemini-key-here
```

---

### Issue 2: "Port 8000 already in use"

**Symptoms:**
```
OSError: [WinError 10048] Only one usage of each socket address permitted
```

**Solution - Option A: Kill existing process**
```bash
# Find process using port 8000
netstat -ano | findstr :8000

# Output example:
# TCP    127.0.0.1:8000    LISTENING    12345

# Kill process
taskkill /PID 12345 /F

# Restart API
python api.py
```

**Solution - Option B: Use different port**
```bash
# Edit .env
notepad .env

# Change:
API_PORT=8000
# To:
API_PORT=8001

# Restart: python api.py
```

---

### Issue 3: "Port 8501 already in use"

**Symptoms:**
```
Address already in use
port 8501
```

**Solution:**
```bash
# Find process
netstat -ano | findstr :8501

# Kill process
taskkill /PID <PID> /F

# Or edit .env
notepad .env
UI_PORT=8502

# Restart Streamlit
streamlit run streamlit_ui.py --server.port 8502
```

---

### Issue 4: "OpenAI API key invalid"

**Symptoms:**
```
AuthenticationError: Incorrect API key provided
```

**Solution:**

1. **Verify API key format:**
   - Must start with `sk-proj-`
   - Must be at least 40 characters
   - No spaces or extra characters

2. **Get new key:**
   - Go to https://platform.openai.com/api-keys
   - Click "Create new secret key"
   - Copy immediately (won't show again)

3. **Update .env:**
   ```bash
   notepad .env
   # Paste: OPENAI_API_KEY=sk-proj-...
   ```

4. **Restart API:**
   ```bash
   python api.py
   ```

---

### Issue 5: "Database connection failed"

**Symptoms:**
```
psycopg2.OperationalError: could not connect to server
```

**Solution - Step 1: Check PostgreSQL running**
```bash
# Test connection
pg_isready -h localhost -p 3307

# Expected output:
# accepting connections

# If failed, start PostgreSQL
```

**Solution - Step 2: Verify database exists**
```bash
# List databases
psql -h localhost -p 3307 -U postgres -l

# Should show: salesdata
```

**Solution - Step 3: Check .env credentials**
```bash
notepad .env

# Verify:
DB_HOST=localhost
DB_PORT=3307
DB_USER=postgres
DB_PASSWORD=hello
DB_NAME=salesdata
```

**Solution - Step 4: Test connection manually**
```bash
psql -h localhost -p 3307 -U postgres -d salesdata -c "SELECT 1"

# Should output:
#  ?column?
# ----------
#        1
```

---

### Issue 6: "API running but UI won't connect"

**Symptoms:**
- UI shows "API is not connected"
- Buttons are greyed out
- Error messages about connection refused

**Solution:**

1. **Verify API is actually running:**
   ```bash
   curl http://localhost:8000/health
   
   # Should return JSON, not error
   ```

2. **Check API logs:**
   - Look at terminal running `python api.py`
   - Should show: `Uvicorn running on http://0.0.0.0:8000`
   - Should show: `✅ Database initialized successfully`

3. **Restart both services:**
   ```bash
   # Kill both terminals (Ctrl+C)
   
   # Terminal 1: Restart API
   cd chatbot_system
   python api.py
   
   # Wait for it to show "running on 0.0.0.0:8000"
   
   # Terminal 2: Restart UI
   cd chatbot_system
   streamlit run streamlit_ui.py
   ```

4. **Clear browser cache:**
   - Press F12 → Application tab
   - Clear Local Storage
   - Refresh page

---

### Issue 7: "ModuleNotFoundError: No module named 'fastapi'"

**Symptoms:**
```
ModuleNotFoundError: No module named 'fastapi'
ModuleNotFoundError: No module named 'langchain'
```

**Solution:**

```bash
# Install missing packages
pip install fastapi uvicorn python-dotenv langchain langchain-openai langchain-google-genai psycopg2-binary pandas streamlit requests

# Verify installation
python -c "import fastapi; print('✅ FastAPI installed')"
```

---

### Issue 8: "Slow responses" or "Query times out"

**Symptoms:**
- Queries take >10 seconds
- Some queries hit "Query execution timeout"
- UI shows spinner for a long time

**Solution - Option 1: Reduce data size**
```bash
# Edit database.py
notepad database.py

# Find: MAX_ROWS = 200
# Change to: MAX_ROWS = 100
```

**Solution - Option 2: Optimize query**
- Use filters instead of loading all data
- Ask: "Show top 10 shops" (instead of "Show all shops")
- Add date range: "Sales in January"

**Solution - Option 3: Check database health**
```bash
# Check table size
psql -h localhost -p 3307 -U postgres -d salesdata -c "
SELECT 
  schemaname,
  tablename,
  pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) AS size
FROM pg_tables
ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC
LIMIT 10;"
```

---

### Issue 9: "Streamlit app keeps crashing"

**Symptoms:**
```
streamlit.errors.StreamlitAPIException
Something went wrong
```

**Solution:**

1. **Restart from scratch:**
   ```bash
   # Kill Streamlit (Ctrl+C)
   # Kill API (Ctrl+C)
   
   # Clear Streamlit cache
   rmdir /s %USERPROFILE%\.streamlit\cache\
   
   # Start API again
   python api.py
   
   # Start Streamlit in new terminal
   streamlit run streamlit_ui.py --client.showErrorDetails=true
   ```

2. **Check terminal for errors:**
   - Look for red error messages
   - Copy full error text
   - Check against this troubleshooting guide

---

### Issue 10: "API returns 500 Internal Server Error"

**Symptoms:**
```
500 Internal Server Error
```

**Solution:**

1. **Check API logs** in the terminal where `python api.py` is running
   - Look for red error messages
   - Note the exact error

2. **Common causes:**

   **LLM API Error:**
   ```
   AuthenticationError: Invalid API key
   ```
   → Check OPENAI_API_KEY in .env

   **Database Error:**
   ```
   psycopg2.OperationalError
   ```
   → Check database is running

   **Query Error:**
   ```
   psycopg2.ProgrammingError: column does not exist
   ```
   → Check table schema, verify SQL is correct

3. **Test with curl:**
   ```bash
   curl -X POST http://localhost:8000/api/chat ^
     -H "Content-Type: application/json" ^
     -d "{\"question\":\"SELECT 1\",\"chat_history\":[]}"
   ```

---

### Issue 11: "LLM not generating SQL"

**Symptoms:**
- API responds but with generic text
- No SQL is shown in chat
- "I can help with that..." messages

**Solution:**

1. **Verify LLM is initialized:**
   - Check API startup logs for:
   ```
   ✅ OpenAI LLM initialized
   ```
   or
   ```
   ✅ Gemini LLM initialized
   ```

2. **If no LLM shows:**
   - Edit `.env`
   - Add: `OPENAI_API_KEY=sk-proj-...`
   - Restart API

3. **Test LLM directly:**
   ```bash
   curl -X POST http://localhost:8000/api/chat ^
     -H "Content-Type: application/json" ^
     -d "{\"question\":\"What is 2+2?\",\"chat_history\":[]}"
   ```

---

### Issue 12: "Certificate verification failed"

**Symptoms:**
```
SSLError: [SSL: CERTIFICATE_VERIFY_FAILED]
```

**Solution:**

1. **For OpenAI/Gemini (not recommended for production):**
   ```python
   # Edit api.py or llm_orchestrator.py
   # Add after imports:
   import urllib3
   urllib3.disable_warnings()
   
   # This disables SSL verification
   ```

2. **Better solution: Update certificates**
   ```bash
   # Install/update certificates
   pip install --upgrade certifi
   
   # Then restart API
   ```

3. **For development only:**
   - Use the SSL disable approach above
   - Switch to production certificates before deployment

---

## 🧪 Diagnostic Commands

### Check Everything
```bash
# Run all checks
echo === Checking Python ===
python --version

echo === Checking PostgreSQL ===
pg_isready -h localhost -p 3307

echo === Checking API Health ===
curl http://localhost:8000/health

echo === Checking Database Connection ===
psql -h localhost -p 3307 -U postgres -c "SELECT 1"

echo === Checking Packages ===
python -c "import fastapi, langchain, streamlit; print('✅ All packages OK')"
```

### Check Port Usage
```bash
# Windows
netstat -ano | findstr :8000
netstat -ano | findstr :8501
netstat -ano | findstr :3307

# Kill if needed
taskkill /PID <PID> /F
```

### Check .env File
```bash
# View .env content
type .env

# Check for required keys
findstr "OPENAI_API_KEY" .env
findstr "DB_HOST" .env
```

---

## 📋 Before Asking for Help

When reporting an issue, gather:

1. **Error message** (full text from terminal)
2. **Steps to reproduce** (exactly what you did)
3. **Environment info:**
   ```bash
   python --version
   pip list | findstr fastapi
   pg_isready -h localhost -p 3307
   ```
4. **Check logs:**
   - API terminal output
   - UI terminal output
   - Browser console (F12)

---

## 🚀 Reset Everything (Nuclear Option)

If nothing works:

```bash
# Kill all Python processes
taskkill /F /IM python.exe

# Kill all node processes (if Streamlit cache)
taskkill /F /IM node.exe

# Clear caches
rmdir /s /q %USERPROFILE%\.streamlit
rmdir /s /q %USERPROFILE%\.cache

# Delete and recreate .env
del .env
copy .env.example .env
notepad .env  # Add your API keys

# Restart fresh
python api.py  # In terminal 1
# Wait 5 seconds
streamlit run streamlit_ui.py  # In terminal 2
```

---

## ✅ Verify Working State

Once running, test these:

1. **API Health:**
   ```bash
   curl http://localhost:8000/health
   # Should return: {"status": "✅ Healthy", ...}
   ```

2. **API Docs:**
   - Open http://localhost:8000/docs
   - Should show Swagger UI with all endpoints

3. **Frontend:**
   - Open http://localhost:8501
   - Should see chat interface
   - Blue 💬 button should be clickable

4. **Chat:**
   - Click chat bubble
   - Type: "Show top 5 shops"
   - Should get response with SQL and data

---

## 📞 Still Having Issues?

1. **Check this guide** for your specific error
2. **Review logs** in both terminals
3. **Run diagnostic commands** above
4. **Try reset option** if stuck
5. **Check dependencies** are installed

---

**Most issues are solved by:**
1. Creating `.env` with API keys
2. Restarting both services
3. Clearing cache and browser data
4. Verifying PostgreSQL is running

Good luck! 🚀
