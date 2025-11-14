"""
Logging configuration for the ML project.
Provides centralized logging setup with different log levels and handlers.
"""

import logging
import logging.handlers
import sys
from pathlib import Path


def setup_logging(log_level=logging.INFO, log_file='ml_project.log', console=True):
    """
    Set up logging configuration for the project.

    Parameters:
    log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    log_file: Path to log file
    console: Whether to also log to console
    """
    # Create logger
    logger = logging.getLogger('ml_energy_optimizer')
    logger.setLevel(log_level)

    # Create formatters
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
    )
    console_formatter = logging.Formatter(
        '%(levelname)s - %(name)s - %(message)s'
    )

    # File handler with rotation
    file_handler = logging.handlers.RotatingFileHandler(
        log_file, maxBytes=10*1024*1024, backupCount=5
    )
    file_handler.setLevel(log_level)
    file_handler.setFormatter(file_formatter)

    # Console handler
    console_handler = None
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_handler.setFormatter(console_formatter)

    # Add handlers to logger
    logger.addHandler(file_handler)
    if console_handler:
        logger.addHandler(console_handler)

    # Prevent duplicate logs
    logger.propagate = False

    return logger


def get_logger(name):
    """
    Get a logger instance with the specified name.

    Parameters:
    name: Name for the logger (usually __name__)

    Returns:
    Logger instance
    """
    return logging.getLogger(f'ml_energy_optimizer.{name}')


class LoggerMixin:
    """Mixin class to add logging capabilities to any class."""

    @property
    def logger(self):
        """Get logger for this class."""
        if not hasattr(self, '_logger'):
            self._logger = get_logger(self.__class__.__name__)
        return self._logger


def log_function_call(func):
    """
    Decorator to log function calls with parameters and execution time.

    Parameters:
    func: Function to decorate

    Returns:
    Decorated function
    """
    def wrapper(*args, **kwargs):
        logger = get_logger(func.__module__)
        func_name = func.__name__

        # Log function entry
        logger.debug(f"Entering {func_name}")

        try:
            # Log arguments (be careful with sensitive data)
            if logger.isEnabledFor(logging.DEBUG):
                arg_str = f"args: {len(args)} items" if args else "args: none"
                kwarg_str = f"kwargs: {list(kwargs.keys())}" if kwargs else "kwargs: none"
                logger.debug(f"{func_name} - {arg_str}, {kwarg_str}")

            # Execute function
            import time
            start_time = time.time()
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time

            # Log successful completion
            logger.info(".4f")

            return result

        except Exception as e:
            # Log exception
            logger.error(f"Exception in {func_name}: {str(e)}", exc_info=True)
            raise

    return wrapper


def log_exceptions(func):
    """
    Decorator to log exceptions in functions.

    Parameters:
    func: Function to decorate

    Returns:
    Decorated function
    """
    def wrapper(*args, **kwargs):
        logger = get_logger(func.__module__)
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Exception in {func.__name__}: {str(e)}", exc_info=True)
            raise

    return wrapper


# Global logger instance
logger = setup_logging()
