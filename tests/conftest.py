import os
from pathlib import Path

TEST_DIR = Path(__file__).parent
os.environ["SENTINEL_DATABASE_URL"] = f"sqlite:///{TEST_DIR / 'test.db'}"
os.environ["SENTINEL_UPLOAD_DIR"] = str(TEST_DIR / "uploads")
(TEST_DIR / "test.db").unlink(missing_ok=True)
