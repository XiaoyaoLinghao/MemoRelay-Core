import os
import tempfile
import pytest

TEST_MEMORY_DIR = os.path.join(os.path.dirname(__file__), "..", "fixtures")

@pytest.fixture
def temp_db_path():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    yield path
    try:
        os.unlink(path)
    except OSError:
        pass

@pytest.fixture
def sample_memory_dir():
    return TEST_MEMORY_DIR
