# 🚀 GET STARTED IN 3 STEPS

## Step 1️⃣ : Configuration (AUTOMATIC - No Setup Needed!)

✅ **Good news!** Your API keys are automatically loaded from your existing Streamlit config:

```
D:\Dashboard Code\NO_WH\DS\testing_openai\.streamlit\secrets.toml
```

The chatbot system will automatically use the same API keys as your Century dashboard!

**No action needed** - Just proceed to Step 2 to start the chatbot.

---

### Optional: Override API Keys

If you want to use different API keys just for the chatbot:

```bash
cd d:\Dashboard Code\NO_WH\DS\chatbot_system
copy .env.example .env
notepad .env
```

Then add your override keys:
```env
OPENAI_API_KEY=sk-proj-your-key-here
GEMINI_API_KEY=your-gemini-key-here
```

### Where to get API keys:
- **OpenAI:** https://platform.openai.com/api-keys
- **Gemini:** https://aistudio.google.com/apikey

**Save and close** (Ctrl+S, then close notepad)

---

## Step 2️⃣ : Start Everything (1 click)

```bash
# Make sure you're still in the chatbot_system folder
cd d:\Dashboard Code\NO_WH\DS\chatbot_system

# Run the startup script
start_all.bat
```

**What happens:**
1. ✅ Checks your `.env` file
2. ✅ Starts FastAPI backend (port 8000)
3. ✅ Starts Streamlit frontend (port 8501)
4. ✅ Opens browser automatically

**Expected output in terminal:**
```
✅ Configuration found (.env)
[1/2] Starting FastAPI Backend on port 8000...
[2/2] Starting Streamlit Frontend on port 8501...
✅ Both services started successfully!
```

---

## Step 3️⃣ : Start Using It! (30 seconds)

Browser should auto-open to: **http://localhost:8501**

### Using the Chat Interface:

1. **See the blue 💬 button** (bottom right corner)
2. **Click it to expand** the chat
3. **Type a question** in the input field:
   ```
   "Show top 5 overstock shops"
   ```
4. **Press Enter** to send
5. **Magic happens!** ✨
   - SQL is generated automatically
   - Query executes
   - Results appear in chat
   - AI insights are shown

---

## 🎯 Example Queries to Try

### Data Queries
```
"Show top 5 overstock shops"
"Which items need replenishment?"
"What are sales by department?"
"List shops with low inventory"
"Find items expiring soon"
```

### Insight Queries
```
"What should we prioritize?"
"How is our business doing?"
"Give me recommendations"
"What are the trends?"
```

---

## 📊 What You'll See

### Chat Interface
```
┌─────────────────────────────────────┐
│         Your Chat Interface         │
│                                     │
│  [AI Response with insights]        │
│                                     │
│  📊 [Data table shown here]         │
│                                     │
│  📥 [Download CSV button]           │
│                                     │
│  [Your message input field]         │
│  [Send Button]                      │
└─────────────────────────────────────┘
   💬 (Click to collapse)
```

### Response Example
```
User: "Show top 5 overstock shops"

AI Response:
✅ Found 5 results

SQL Used:
SELECT shop_code, total_stock, sales_30d 
FROM mv_recommendations_complete 
ORDER BY total_stock DESC LIMIT 5

📊 Data Table:
| Shop | Stock | Sales |
|------|-------|-------|
| SPN  | 1500  | 450   |
| MSS  | 1200  | 380   |
| ...  | ...   | ...   |

💡 Insights:
• SPN has highest stock level (1500 units)
• Stock levels are 3x 30-day sales (overstocked)
• Recommend transfer to lower-stock shops
```

---

## ✅ Verify Everything Works

### Check 1: API Health
Open in browser:
```
http://localhost:8000/health
```
Should show: `{"status": "✅ Healthy", ...}`

### Check 2: API Documentation
Open in browser:
```
http://localhost:8000/docs
```
Should show interactive API documentation

### Check 3: Frontend
Open in browser:
```
http://localhost:8501
```
Should show your chat interface with blue 💬 button

### Check 4: Send a Test Message
1. Click the 💬 button
2. Type: "SELECT 1" 
3. Should get a response with SQL

---

## 🆘 If Something Goes Wrong

### Problem: .env file not found
```bash
cd d:\Dashboard Code\NO_WH\DS\chatbot_system
copy .env.example .env
notepad .env  # Add your API keys
```

### Problem: Port 8000 in use
```bash
# Kill the process
taskkill /F /IM python.exe

# Then restart
start_all.bat
```

### Problem: API key invalid
```bash
# Check your API key at:
# https://platform.openai.com/api-keys
# Must start with: sk-proj-

# Update .env and restart
notepad .env
start_all.bat
```

### Problem: UI won't connect
```bash
# Make sure API started first
# Check: http://localhost:8000/health

# If that works, restart UI:
# Kill Streamlit window (Ctrl+C)
# Then: streamlit run streamlit_ui.py
```

**Full troubleshooting:** See [TROUBLESHOOTING.md](TROUBLESHOOTING.md)

---

## 📚 Full Documentation

Once everything is working, read these to understand more:

1. **[QUICK_REFERENCE.md](QUICK_REFERENCE.md)** - One-page cheat sheet (2 min)
2. **[README.md](README.md)** - Complete documentation (15 min)
3. **[SETUP_GUIDE.md](SETUP_GUIDE.md)** - Detailed setup info (10 min)
4. **[TROUBLESHOOTING.md](TROUBLESHOOTING.md)** - Problem solutions

---

## 🎯 Next Steps After Setup

### Short Term (After 1st run)
- [ ] Try different questions
- [ ] Download CSV results
- [ ] Check API docs at `/docs`
- [ ] Review logs in terminal

### Medium Term (This week)
- [ ] Customize your queries
- [ ] Add more SQL examples
- [ ] Set up monitoring

### Long Term (Next month)
- [ ] Deploy to production
- [ ] Add authentication
- [ ] Scale to more users

---

## 💡 Pro Tips

### 1. API Documentation
Always available at: **http://localhost:8000/docs**
- Shows all endpoints
- Try requests directly
- See examples

### 2. Check Health
```bash
curl http://localhost:8000/health
```
Quick way to verify everything is working

### 3. View Logs
- API logs: In terminal running `python api.py`
- UI logs: In terminal running `streamlit run streamlit_ui.py`
- Shows detailed errors if something fails

### 4. Sample Queries
UI has built-in sample queries
- Click suggestions in sidebar
- Helps you learn the system

### 5. Download Results
Every response can be exported as CSV
- Click "Download CSV" button
- Use in Excel, etc.

---

## 🎉 You're Ready!

### Summary of what you just did:
1. ✅ Created `.env` with API keys
2. ✅ Started FastAPI backend
3. ✅ Started Streamlit frontend
4. ✅ Opened chat interface
5. ✅ Asked first question!

### Your system now:
- ✅ Converts natural language to SQL
- ✅ Executes queries safely
- ✅ Shows results with AI insights
- ✅ Allows CSV export
- ✅ Maintains chat history
- ✅ Provides professional interface

---

## 🚀 Launch Command (Anytime)

To run again later:
```bash
cd d:\Dashboard Code\NO_WH\DS\chatbot_system
start_all.bat
```

Then open: **http://localhost:8501**

---

## 📋 Quick Reference Card

| Task | Command |
|------|---------|
| Start everything | `start_all.bat` |
| Start API only | `python api.py` |
| Start UI only | `streamlit run streamlit_ui.py` |
| View API docs | http://localhost:8000/docs |
| View UI | http://localhost:8501 |
| Check health | http://localhost:8000/health |
| View config | `notepad .env` |
| Edit config | `notepad .env` |
| Stop services | Ctrl+C in both terminals |

---

## ❓ Common Questions

**Q: Why do I need API keys?**
A: The system uses OpenAI or Gemini to convert your questions to SQL. You need keys for those services.

**Q: Is my data safe?**
A: Yes! Only SELECT queries are allowed. No INSERT/UPDATE/DELETE. All queries are validated.

**Q: Can I use this without internet?**
A: No, because the LLM (language model) is in the cloud. You need internet for OpenAI/Gemini.

**Q: What if I don't want to use OpenAI?**
A: The system also supports Google Gemini. Just add your Gemini key in `.env`.

**Q: Can I modify the code?**
A: Yes! It's fully documented. See the source files and inline comments.

**Q: How do I update my questions?**
A: Just type in the chat box. No code changes needed.

---

**That's it! You're all set! 🎉**

Next: Open http://localhost:8501 and start asking questions!

For help: See [TROUBLESHOOTING.md](TROUBLESHOOTING.md)

---

**Built with ❤️ for Melcom Retail Analytics**
