"""
Logging utility for the project.
Provides consistent logging across all modules.
"""

import logging
import sys
from pathlib import Path
from datetime import datetime


def setup_logger(
    name: str,
    log_file: str = None,
    level: int = logging.INFO,
    log_to_console: bool = True
) -> logging.Logger:
    """
    Set up a logger with file and console handlers.
    
    Args:
        name: Logger name
        log_file: Path to log file (optional)
        level: Logging level
        log_to_console: Whether to log to console
    
    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Remove existing handlers to avoid duplicates
    logger.handlers = []
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler with UTF-8 encoding support
    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        
        # Set UTF-8 encoding for console output
        try:
            console_handler.stream.reconfigure(encoding='utf-8')
        except AttributeError:
            # Python < 3.7 compatibility
            pass
        
        logger.addHandler(console_handler)
    
    # File handler with UTF-8 encoding
    if log_file:
        # Create logs directory if it doesn't exist
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get or create a logger with default settings.
    
    Args:
        name: Logger name (typically __name__ of the module)
    
    Returns:
        Logger instance
    """
    # Create log file with timestamp
    timestamp = datetime.now().strftime("%Y%m%d")
    log_file = f"logs/{name.replace('.', '_')}_{timestamp}.log"
    
    return setup_logger(
        name=name,
        log_file=log_file,
        level=logging.INFO,
        log_to_console=True
    )


if __name__ == "__main__":
    # Test the logger
    logger = get_logger("test")
    
    logger.info("[OK] Logger test: INFO level")
    logger.warning("[WARN] Logger test: WARNING level")
    logger.error("[ERROR] Logger test: ERROR level")
    
    print(f"\n[OK] Log file created at: logs/")