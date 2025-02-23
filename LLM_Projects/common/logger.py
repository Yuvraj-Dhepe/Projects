import sys
from loguru import logger

# Remove default handlers to avoid duplicate logs
logger.remove()

# Add a new handler with colorized output
logger.add(
    sys.stdout,
    colorize=True,  # Enables color output
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
    "<level>{message}</level>",
    level="DEBUG",
)
