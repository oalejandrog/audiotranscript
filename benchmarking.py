import time
import functools
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Benchmark")

class Benchmark:
    def __init__(self, name="Operation"):
        self.name = name
        self.start_time = None
        self.end_time = None

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        duration = (self.end_time - self.start_time) * 1000
        logger.info(f"[{self.name}] took {duration:.2f}ms")

def measure_latency(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        logger.info(f"Function {func.__name__} latency: {(end - start) * 1000:.2f}ms")
        return result
    return wrapper
