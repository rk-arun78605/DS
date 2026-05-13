"""
Configuration management for the chatbot system
"""
import os
import toml
from pathlib import Path
from dotenv import load_dotenv

# Try to load from secrets.toml first (shared with Century dashboard)
SECRETS_PATH = Path("D:\\Dashboard Code\\NO_WH\\DS\\testing_openai\\.streamlit\\secrets.toml")
secrets = {}

if SECRETS_PATH.exists():
    try:
        with open(SECRETS_PATH, 'r') as f:
            secrets = toml.load(f)
    except Exception as e:
        print(f"⚠️ Warning: Could not load secrets.toml: {e}")

# Fall back to .env if no secrets.toml
load_dotenv()

class Config:
    """Central configuration for all components"""
    
    # API Configuration
    API_HOST = os.getenv("API_HOST", "localhost")
    API_PORT = int(os.getenv("API_PORT", 8000))
    API_TITLE = "AI Chatbot Analytics API"
    API_VERSION = "1.0.0"
    
    # Database Configuration
    DB_HOST = os.getenv("DB_HOST", "localhost")
    DB_PORT = int(os.getenv("DB_PORT", 3307))
    DB_USER = os.getenv("DB_USER", "postgres")
    DB_PASSWORD = os.getenv("DB_PASSWORD", "hello")
    DB_NAME = os.getenv("DB_NAME", "salesdata")
    
    # LLM Configuration (from secrets.toml > .env > defaults)
    OPENAI_API_KEY = secrets.get("llm", {}).get("openai_key") or os.getenv("OPENAI_API_KEY", "")
    OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    
    GEMINI_API_KEY = secrets.get("llm", {}).get("gemini_key") or os.getenv("GEMINI_API_KEY", "")
    GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
    
    PERPLEXITY_API_KEY = secrets.get("llm", {}).get("perplexity_key") or os.getenv("PERPLEXITY_API_KEY", "")
    PERPLEXITY_MODEL = os.getenv("PERPLEXITY_MODEL", "llama-3.1-sonar-small-128k-chat")
    
    # Feature Flags
    ENABLE_SQL_GENERATION = True
    ENABLE_DATA_ANALYSIS = True
    ENABLE_INSIGHTS = True
    MAX_QUERY_RESULTS = 200
    QUERY_TIMEOUT = 30
    
    # Chat Configuration
    MAX_CHAT_HISTORY = 10
    CHAT_CONTEXT_WINDOW = 5
    
    # Logging
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    
    # Safety Rules
    BLOCKED_KEYWORDS = [
        'insert', 'update', 'delete', 'drop', 'alter',
        'truncate', 'grant', 'revoke', 'create', 'copy'
    ]
    
    # SQL Generation Rules
    SAFE_TABLES = [
        'mv_century_penetration',
        'sales_2024', 'sales_2025', 'sales_2026',
        'inventory_master', 'itemdetails'
    ]

config = Config()
