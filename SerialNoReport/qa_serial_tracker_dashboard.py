from __future__ import annotations

import ast
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


SOURCE_FILE = Path(__file__).with_name("serial_tracker_dashboard_new.py")


@dataclass
class Finding:
    severity: str  # CRITICAL | WARNING | INFO
    code: str
    message: str
    line: int | None = None


def _line_for_offset(text: str, offset: int) -> int:
    return text.count("\n", 0, offset) + 1


def check_syntax(text: str) -> list[Finding]:
    findings: list[Finding] = []
    try:
        compile(text, str(SOURCE_FILE), "exec")
    except SyntaxError as exc:
        findings.append(
            Finding(
                severity="CRITICAL",
                code="PY_SYNTAX",
                message=f"Syntax error: {exc.msg}",
                line=exc.lineno,
            )
        )
    return findings


def check_for_unicode_sql_operators(text: str) -> list[Finding]:
    findings: list[Finding] = []
    for match in re.finditer(r"[≥≤]", text):
        findings.append(
            Finding(
                severity="CRITICAL",
                code="SQL_UNICODE_OPERATOR",
                message="Found unicode SQL comparison operator. Use >= or <= in SQL.",
                line=_line_for_offset(text, match.start()),
            )
        )
    return findings


def check_round_float_signature(text: str) -> list[Finding]:
    findings: list[Finding] = []
    patterns = [
        r"ROUND\s*\(\s*[^\)]*::\s*FLOAT[^\)]*,\s*\d+\s*\)",
        r"ROUND\s*\(\s*[^\)]*::\s*DOUBLE\s+PRECISION[^\)]*,\s*\d+\s*\)",
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE | re.DOTALL):
            findings.append(
                Finding(
                    severity="CRITICAL",
                    code="SQL_ROUND_FLOAT",
                    message=(
                        "Potential PostgreSQL error: ROUND(value, digits) with float/double. "
                        "Cast expression to NUMERIC before ROUND(value, digits)."
                    ),
                    line=_line_for_offset(text, match.start()),
                )
            )
    return findings


def _iter_try_nodes(tree: ast.AST) -> Iterable[ast.Try]:
    for node in ast.walk(tree):
        if isinstance(node, ast.Try):
            yield node


def _contains_bare_conn_close(nodes: list[ast.stmt]) -> bool:
    for node in nodes:
        for sub in ast.walk(node):
            if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Attribute):
                if isinstance(sub.func.value, ast.Name) and sub.func.value.id == "conn" and sub.func.attr == "close":
                    return True
    return False


def _contains_conn_guard(nodes: list[ast.stmt]) -> bool:
    for node in nodes:
        if isinstance(node, ast.If):
            test = node.test
            if isinstance(test, ast.Name) and test.id == "conn":
                return True
            if isinstance(test, ast.Compare) and isinstance(test.left, ast.Name) and test.left.id == "conn":
                return True
    return False


def check_connection_close_handling(text: str) -> list[Finding]:
    findings: list[Finding] = []
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return findings

    for try_node in _iter_try_nodes(tree):
        for handler in try_node.handlers:
            if _contains_bare_conn_close(handler.body) and not _contains_conn_guard(handler.body):
                findings.append(
                    Finding(
                        severity="WARNING",
                        code="CONN_CLOSE_GUARD",
                        message=(
                            "except block calls conn.close() without explicit guard. "
                            "If conn can be None/unassigned, this can mask the root error."
                        ),
                        line=handler.lineno,
                    )
                )
    return findings


def check_tab_placeholder_residue(text: str) -> list[Finding]:
    findings: list[Finding] = []
    for label in ["Discrepancies (Coming Soon)", "Analytics (Coming Soon)"]:
        idx = text.find(label)
        if idx != -1:
            findings.append(
                Finding(
                    severity="INFO",
                    code="PLACEHOLDER_TAB_TEXT",
                    message=f"Placeholder tab label still present: {label}",
                    line=_line_for_offset(text, idx),
                )
            )
    return findings


def run_all_checks(text: str) -> list[Finding]:
    checks = [
        check_syntax,
        check_for_unicode_sql_operators,
        check_round_float_signature,
        check_connection_close_handling,
        check_tab_placeholder_residue,
    ]
    findings: list[Finding] = []
    for check in checks:
        findings.extend(check(text))
    return sorted(findings, key=lambda f: (0 if f.severity == "CRITICAL" else 1 if f.severity == "WARNING" else 2, f.line or 0))


def print_report(findings: list[Finding]) -> None:
    critical = sum(1 for f in findings if f.severity == "CRITICAL")
    warning = sum(1 for f in findings if f.severity == "WARNING")
    info = sum(1 for f in findings if f.severity == "INFO")

    print("=" * 88)
    print(f"QA Report: {SOURCE_FILE.name}")
    print(f"Summary -> CRITICAL: {critical} | WARNING: {warning} | INFO: {info}")
    print("=" * 88)

    if not findings:
        print("No findings. ✅")
        return

    for finding in findings:
        location = f"line {finding.line}" if finding.line else "line ?"
        print(f"[{finding.severity}] {finding.code} @ {location}: {finding.message}")

    print("\nSuggested handling:")
    print("- Fix CRITICAL findings before release.")
    print("- Treat WARNING findings as reliability hardening tasks.")
    print("- Re-run: python SerialNoReport/qa_serial_tracker_dashboard.py")


def main() -> int:
    if not SOURCE_FILE.exists():
        print(f"Source file not found: {SOURCE_FILE}")
        return 2

    text = SOURCE_FILE.read_text(encoding="utf-8")
    findings = run_all_checks(text)
    print_report(findings)

    has_critical = any(f.severity == "CRITICAL" for f in findings)
    return 1 if has_critical else 0


if __name__ == "__main__":
    sys.exit(main())
