# Pipelines package for ReviewInsight AI
try:
    from src.pipelines.build_database import DatabaseBuilder
    __all__ = ['DatabaseBuilder']
except ImportError:
    # duckdb not installed
    DatabaseBuilder = None
    __all__ = []
