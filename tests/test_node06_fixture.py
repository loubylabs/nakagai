import hashlib
from pathlib import Path


FIXTURES = Path(__file__).parent / "fixtures"


def test_node06_baseline_fixture_matches_its_transferred_digest():
    payload = (FIXTURES / "node06-before.json").read_bytes()
    recorded = (FIXTURES / "node06-before.sha256").read_text().strip()
    assert recorded == (
        f"{hashlib.sha256(payload).hexdigest()}  node06-before.json"
    )
