#!/usr/bin/env python3
"""
Configuration Verification Script
Checks if API keys are properly loaded from secrets.toml or .env
"""
import sys
from pathlib import Path
from config import config

def check_config():
    """Verify all required configuration is present"""
    print("=" * 60)
    print("🔍 Chatbot Configuration Verification")
    print("=" * 60)
    print()
    
    # Check secrets.toml
    secrets_path = Path("D:\\Dashboard Code\\NO_WH\\DS\\testing_openai\\.streamlit\\secrets.toml")
    print(f"📍 Secrets file: {secrets_path}")
    print(f"   Status: {'✅ FOUND' if secrets_path.exists() else '❌ NOT FOUND'}")
    print()
    
    # Check API Keys
    print("🔑 API Keys Status:")
    print(f"   OpenAI:     {'✅ Configured' if config.OPENAI_API_KEY else '❌ Not configured'}")
    print(f"   Gemini:     {'✅ Configured' if config.GEMINI_API_KEY else '❌ Not configured'}")
    print(f"   Perplexity: {'✅ Configured' if config.PERPLEXITY_API_KEY else '❌ Not configured'}")
    print()
    
    # Check Database
    print("💾 Database Configuration:")
    print(f"   Host: {config.DB_HOST}:{config.DB_PORT}")
    print(f"   User: {config.DB_USER}")
    print(f"   Database: {config.DB_NAME}")
    print()
    
    # Check API Settings
    print("⚙️  API Configuration:")
    print(f"   Host: {config.API_HOST}")
    print(f"   Port: {config.API_PORT}")
    print()
    
    # Check if at least one LLM is configured
    has_llm = config.OPENAI_API_KEY or config.GEMINI_API_KEY or config.PERPLEXITY_API_KEY
    
    print("=" * 60)
    if has_llm:
        print("✅ Configuration is READY!")
        print()
        print("You can now run:")
        print("   python api.py")
        print("or")
        print("   start_all.bat")
        print("=" * 60)
        return True
    else:
        print("❌ Configuration is INCOMPLETE!")
        print()
        print("No LLM API keys found. Please:")
        print("1. Check that secrets.toml exists at:")
        print(f"   {secrets_path}")
        print("2. OR create a .env file with API keys")
        print()
        print("See SETUP_GUIDE.md for details.")
        print("=" * 60)
        return False

if __name__ == "__main__":
    success = check_config()
    sys.exit(0 if success else 1)
