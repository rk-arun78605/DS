"""
Data analysis and processing module
"""
import logging
from typing import Dict, List, Any
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

class DataAnalyzer:
    """Analyzes query results and generates summaries"""
    
    @staticmethod
    def analyze_dataframe(df: pd.DataFrame) -> Dict[str, Any]:
        """Generate comprehensive analysis of dataframe"""
        
        if df.empty:
            return {
                "row_count": 0,
                "summary": "Query returned no results",
                "columns": [],
                "stats": {}
            }
        
        analysis = {
            "row_count": len(df),
            "column_count": len(df.columns),
            "columns": df.columns.tolist(),
            "dtypes": df.dtypes.astype(str).to_dict(),
            "summary": DataAnalyzer._generate_summary(df)
        }
        
        # Numeric analysis
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            analysis["numeric_stats"] = {
                col: {
                    "mean": float(df[col].mean()),
                    "median": float(df[col].median()),
                    "min": float(df[col].min()),
                    "max": float(df[col].max()),
                    "std": float(df[col].std())
                }
                for col in numeric_cols
            }
        
        logger.info(f"✅ Analysis complete: {analysis['row_count']} rows, {analysis['column_count']} columns")
        return analysis
    
    @staticmethod
    def _generate_summary(df: pd.DataFrame) -> str:
        """Generate human-readable summary"""
        
        rows = len(df)
        cols = len(df.columns)
        
        # Find numeric and categorical columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        summary_parts = [f"{rows} records with {cols} fields"]
        
        if numeric_cols:
            summary_parts.append(f"Numeric: {', '.join(numeric_cols[:3])}")
        
        if categorical_cols:
            summary_parts.append(f"Categories: {', '.join(categorical_cols[:2])}")
        
        return " | ".join(summary_parts)
    
    @staticmethod
    def format_for_display(df: pd.DataFrame, max_rows: int = 50) -> List[Dict[str, Any]]:
        """Format dataframe for API response"""
        return df.head(max_rows).to_dict('records')
    
    @staticmethod
    def detect_query_type(df: pd.DataFrame) -> str:
        """Detect type of analysis (comparison, ranking, trend, etc.)"""
        
        numeric_count = len(df.select_dtypes(include=[np.number]).columns)
        text_count = len(df.select_dtypes(include=['object']).columns)
        
        if numeric_count > 3:
            return "detailed_analysis"
        elif numeric_count > 1 and text_count > 0:
            return "comparison"
        elif numeric_count == 1:
            return "simple_ranking"
        else:
            return "categorical"

class MetricsCalculator:
    """Calculate business metrics from data"""
    
    @staticmethod
    def calculate_growth(current: float, previous: float) -> float:
        """Calculate percentage growth"""
        if previous == 0:
            return 0.0
        return ((current - previous) / previous) * 100
    
    @staticmethod
    def calculate_variance(actual: float, expected: float) -> float:
        """Calculate variance percentage"""
        if expected == 0:
            return 0.0
        return ((actual - expected) / expected) * 100
    
    @staticmethod
    def aggregate_by_category(df: pd.DataFrame, category_col: str, metric_col: str) -> Dict[str, float]:
        """Aggregate metrics by category"""
        return df.groupby(category_col)[metric_col].sum().to_dict()

class ReportGenerator:
    """Generate formatted reports from analysis"""
    
    @staticmethod
    def generate_markdown_report(title: str, analysis: Dict[str, Any], data: List[Dict]) -> str:
        """Generate markdown formatted report"""
        
        report = f"# {title}\n\n"
        report += f"**Summary:** {analysis.get('summary', 'N/A')}\n\n"
        report += f"📊 **Records:** {analysis.get('row_count', 0)}\n\n"
        
        if 'numeric_stats' in analysis:
            report += "## Key Metrics\n\n"
            for col, stats in analysis['numeric_stats'].items():
                report += f"- **{col}:** Min={stats['min']:.2f}, Max={stats['max']:.2f}, Avg={stats['mean']:.2f}\n"
        
        report += "\n## Top Results\n\n"
        for i, row in enumerate(data[:5], 1):
            report += f"{i}. {row}\n"
        
        return report
