"""
Database layer - Connection pooling and query execution
"""
import logging
from contextlib import contextmanager
from typing import Optional, List, Dict, Any
import psycopg2
from psycopg2 import pool
from psycopg2.extras import RealDictCursor
import pandas as pd
from config import config

logger = logging.getLogger(__name__)

class DatabaseManager:
    """Manages PostgreSQL connections and queries"""
    
    _pool: Optional[pool.SimpleConnectionPool] = None
    
    @classmethod
    def get_pool(cls) -> pool.SimpleConnectionPool:
        """Get or create connection pool"""
        if cls._pool is None:
            cls._pool = pool.SimpleConnectionPool(
                minconn=2,
                maxconn=10,
                host=config.DB_HOST,
                port=config.DB_PORT,
                user=config.DB_USER,
                password=config.DB_PASSWORD,
                dbname=config.DB_NAME,
                connect_timeout=5
            )
            logger.info("✅ Database connection pool created")
        return cls._pool
    
    @classmethod
    @contextmanager
    def get_connection(cls):
        """Get connection from pool"""
        pool_obj = cls.get_pool()
        conn = pool_obj.getconn()
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.error(f"❌ Database error: {str(e)}")
            raise
        finally:
            pool_obj.putconn(conn)
    
    @staticmethod
    def execute_query(sql: str, params: tuple = ()) -> pd.DataFrame:
        """Execute query and return pandas DataFrame"""
        try:
            with DatabaseManager.get_connection() as conn:
                df = pd.read_sql_query(sql, conn, params=params)
                logger.info(f"✅ Query executed: {len(df)} rows returned")
                return df
        except psycopg2.Error as e:
            logger.error(f"❌ Query execution failed: {str(e)}")
            raise
    
    @staticmethod
    def execute_fetch_one(sql: str, params: tuple = ()) -> Optional[Dict[str, Any]]:
        """Execute query and fetch one row"""
        try:
            with DatabaseManager.get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute(sql, params)
                    result = cur.fetchone()
                    return dict(result) if result else None
        except psycopg2.Error as e:
            logger.error(f"❌ Fetch one failed: {str(e)}")
            raise
    
    @staticmethod
    def close_pool():
        """Close all connections in pool"""
        if DatabaseManager._pool:
            DatabaseManager._pool.closeall()
            DatabaseManager._pool = None
            logger.info("✅ Database connection pool closed")

# Initialize logger
logging.basicConfig(
    level=config.LOG_LEVEL,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
