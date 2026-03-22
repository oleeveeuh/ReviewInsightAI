# Database package for ReviewInsight AI
try:
    from src.database.db_manager import ReviewDatabase
    __all__ = ['ReviewDatabase']
except ImportError:
    # duckdb not installed
    ReviewDatabase = None
    __all__ = []
