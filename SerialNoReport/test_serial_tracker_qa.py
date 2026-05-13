from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_qa_module():
    qa_path = Path(__file__).with_name("qa_serial_tracker_dashboard.py")
    spec = importlib.util.spec_from_file_location("qa_serial_tracker_dashboard", qa_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load QA module from {qa_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> int:
    qa_module = _load_qa_module()
    source = Path(__file__).with_name("serial_tracker_dashboard_new.py")
    text = source.read_text(encoding="utf-8")
    findings = qa_module.run_all_checks(text)
    critical = [f for f in findings if f.severity == "CRITICAL"]

    if critical:
        print("Critical QA findings detected:")
        for item in critical:
            print(f"- {item.code} line {item.line}: {item.message}")
        return 1

    print("QA test passed: no CRITICAL findings.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
