"""
FastAPI application - REST API for chatbot
Built with ❤️ for Melcom Retail Analytics
"""
import logging
import sys
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, HTTPException, WebSocket, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from datetime import datetime
import json
import traceback

from config import config
from database import DatabaseManager
from llm_orchestrator import llm_orchestrator
from data_analyzer import DataAnalyzer, MetricsCalculator, ReportGenerator

# Configure logging with colors
logging.basicConfig(
    level=config.LOG_LEVEL,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger(__name__)

# Initialize FastAPI with enhanced documentation
app = FastAPI(
    title="🤖 AI Analytics Chatbot API",
    version=config.API_VERSION,
    description="Professional AI-powered analytics chatbot with SQL generation, multi-LLM support, and advanced insights",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# CORS middleware - Allow all origins for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    max_age=600,
)

# ============================================================
# Pydantic Models
# ============================================================

class ChatMessage(BaseModel):
    """Chat message model"""
    role: str  # 'user' or 'assistant'
    content: str
    timestamp: Optional[datetime] = None

class ChatRequest(BaseModel):
    """Chat query request"""
    question: str
    chat_history: Optional[List[ChatMessage]] = []
    user_id: Optional[str] = "guest"

class QueryResult(BaseModel):
    """Query execution result"""
    success: bool
    sql: Optional[str] = None
    row_count: int = 0
    data: List[Dict[str, Any]] = []
    error: Optional[str] = None

class InsightResponse(BaseModel):
    """Insight generation response"""
    question: str
    query_type: str
    row_count: int
    insights: str
    summary: str
    data_preview: List[Dict[str, Any]]

class ChatResponse(BaseModel):
    """Chat response model"""
    response: str
    insights: Optional[str] = None
    data: Optional[List[Dict[str, Any]]] = None
    row_count: int = 0
    sql_used: Optional[str] = None
    timestamp: datetime = datetime.now()

# ============================================================
# API Routes
# ============================================================

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        result = DatabaseManager.execute_fetch_one("SELECT 1")
        return {
            "status": "✅ Healthy",
            "database": "✅ Connected",
            "llm_available": llm_orchestrator.openai_client is not None or llm_orchestrator.gemini_client is not None,
            "timestamp": datetime.now()
        }
    except Exception as e:
        logger.error(f"❌ Health check failed: {str(e)}")
        return {
            "status": "❌ Unhealthy",
            "error": str(e),
            "timestamp": datetime.now()
        }

@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Process chat query with SQL generation and insights"""
    
    try:
        logger.info(f"🔍 Processing query: {request.question[:100]}")
        
        # Check if it's a data query
        sql_keywords = ['show', 'list', 'find', 'top', 'which', 'how many', 'count', 'shop', 'item', 'stock']
        is_data_query = any(kw in request.question.lower() for kw in sql_keywords)
        
        if not is_data_query:
            # For non-data queries, return insights only
            insights, llm_used = llm_orchestrator.generate_insights(
                request.question,
                "General analytics question",
                0
            )
            return ChatResponse(
                response=insights,
                sql_used=None
            )
        
        # Generate SQL from question
        sql, sql_provider = llm_orchestrator.generate_sql(request.question)
        logger.info(f"📝 Generated SQL from {sql_provider}: {sql[:100]}...")
        
        # Execute SQL
        df = DatabaseManager.execute_query(sql)
        
        if df.empty:
            return ChatResponse(
                response="❌ No data found matching your criteria. Try a different query.",
                sql_used=sql,
                row_count=0
            )
        
        # Analyze results
        analysis = DataAnalyzer.analyze_dataframe(df)
        
        # Generate insights
        data_summary = analysis['summary']
        insights, insight_provider = llm_orchestrator.generate_insights(
            request.question,
            data_summary,
            len(df)
        )
        logger.info(f"💡 Insights generated by {insight_provider}")
        
        # Format response
        data_preview = DataAnalyzer.format_for_display(df, max_rows=20)
        
        return ChatResponse(
            response=f"✅ Found {len(df)} results",
            insights=insights,
            data=data_preview,
            row_count=len(df),
            sql_used=sql
        )
        
    except ValueError as e:
        logger.error(f"❌ Validation error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"❌ Chat processing error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/query/validate")
async def validate_query(sql: str):
    """Validate SQL query without executing"""
    
    try:
        # Validate SQL
        validated_sql = llm_orchestrator._sanitize_sql(sql)
        return {
            "valid": True,
            "sql": validated_sql,
            "message": "✅ Query is valid"
        }
    except ValueError as e:
        return {
            "valid": False,
            "error": str(e)
        }

@app.get("/api/tables")
async def get_tables():
    """Get list of available tables"""
    return {
        "available_tables": config.SAFE_TABLES,
        "blocked_keywords": config.BLOCKED_KEYWORDS
    }

@app.post("/api/query/execute")
async def execute_query(sql: str, limit: int = 100):
    """Execute SQL query directly"""
    
    try:
        # Validate SQL
        validated_sql = llm_orchestrator._sanitize_sql(sql)
        
        # Execute
        df = DatabaseManager.execute_query(validated_sql)
        
        # Analyze
        analysis = DataAnalyzer.analyze_dataframe(df)
        data = DataAnalyzer.format_for_display(df, max_rows=limit)
        
        return QueryResult(
            success=True,
            sql=validated_sql,
            row_count=len(df),
            data=data
        )
        
    except ValueError as e:
        logger.error(f"❌ Query validation failed: {str(e)}")
        return QueryResult(
            success=False,
            error=str(e)
        )
    except Exception as e:
        logger.error(f"❌ Query execution failed: {str(e)}")
        return QueryResult(
            success=False,
            error=str(e)
        )

@app.post("/api/insights/generate")
async def generate_insights(question: str, data_summary: str, sample_size: int = 0):
    """Generate insights from data"""
    
    try:
        insights, provider = llm_orchestrator.generate_insights(
            question,
            data_summary,
            sample_size
        )
        return {
            "insights": insights,
            "provider": provider,
            "timestamp": datetime.now()
        }
    except Exception as e:
        logger.error(f"❌ Insight generation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.on_event("startup")
async def startup_event():
    """Startup event handler"""
    logger.info("🚀 Starting Chatbot API Server")
    logger.info(f"📊 Database: {config.DB_HOST}:{config.DB_PORT}/{config.DB_NAME}")
    logger.info(f"🤖 LLM: {'OpenAI' if llm_orchestrator.openai_client else 'Gemini' if llm_orchestrator.gemini_client else 'None'}")

@app.on_event("shutdown")
async def shutdown_event():
    """Shutdown event handler"""
    logger.info("🛑 Shutting down Chatbot API Server")
    DatabaseManager.close_pool()

if __name__ == "__main__":
    import uvicorn
    
    logger.info("=" * 60)
    logger.info("🤖 AI Analytics Chatbot API")
    logger.info("=" * 60)
    logger.info(f"🌐 Starting server on {config.API_HOST}:{config.API_PORT}")
    logger.info(f"📊 API Docs: http://{config.API_HOST}:{config.API_PORT}/docs")
    logger.info("=" * 60)
    
    uvicorn.run(
        app,
        host=config.API_HOST,
        port=config.API_PORT,
        log_level=config.LOG_LEVEL.lower(),
        access_log=True,
    )
