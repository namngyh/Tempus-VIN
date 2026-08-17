# tests/test_package_smoke.py
import raemf_mc


def test_package_importable_and_versioned():
    assert raemf_mc.__version__ == "0.1.0"
