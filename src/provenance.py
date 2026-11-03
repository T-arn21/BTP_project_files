"""Exact provenance differences with an evidence-only filesystem alias exception."""
from pathlib import Path
import json
import hashlib


_MISSING = object()
PATH_FIELDS = {"data_config.data_dir", "preprocessing_identity.root"}


def field_differences(expected, actual, prefix="", category="source_vs_current"):
    """Keep exact values (including full ID lists), without string/numeric coercion."""
    differences = []
    if isinstance(expected, dict) and isinstance(actual, dict):
        for key in sorted(set(expected) | set(actual)):
            path = f"{prefix}.{key}" if prefix else str(key)
            differences.extend(field_differences(expected.get(key, _MISSING), actual.get(key, _MISSING), path, category))
    else:
        present = expected is not _MISSING and actual is not _MISSING
        # Boolean assertions must not accept integer 0/1 as if they were bools.
        equivalent = present and expected == actual
        if isinstance(expected, bool) or isinstance(actual, bool):
            equivalent = equivalent and type(expected) is type(actual)
        if not equivalent:
            differences.append(dict(field=prefix, category=category,
                expected=expected if expected is not _MISSING else None,
                actual=actual if actual is not _MISSING else None,
                expected_present=expected is not _MISSING, actual_present=actual is not _MISSING,
                accepted=False))
    return differences


def same_directory_evidence(expected, actual):
    """Never infer equivalence from matching basenames, hashes or path text alone."""
    evidence = {"method": "filesystem_samefile", "verified": False}
    try:
        if not isinstance(expected, str) or not isinstance(actual, str):
            return dict(evidence, reason="Both directory paths must be strings")
        left, right = Path(expected), Path(actual)
        if not left.is_absolute() or not right.is_absolute() or not left.is_dir() or not right.is_dir():
            return dict(evidence, reason="Both absolute directories must currently exist")
        evidence.update(expected_resolved=str(left.resolve(strict=True)), actual_resolved=str(right.resolve(strict=True)))
        evidence["verified"] = left.samefile(right)
        evidence["reason"] = "same filesystem directory" if evidence["verified"] else "different filesystem directories"
        if evidence["verified"]:
            stat = left.stat()
            evidence.update(device=stat.st_dev, inode=stat.st_ino)
    except (OSError, ValueError, RuntimeError) as exc:
        evidence["reason"] = f"Cannot establish directory identity: {exc}"
    return evidence


def comparison_report(expected, actual, policy_expected, policy_actual, **context):
    differences = field_differences(expected, actual)
    differences += field_differences(policy_expected, policy_actual, "source_policy", "source_policy")
    # A path exception is applied only when it is the entire difference, both
    # directory names resolve to the same real object, and EVERY other check passes.
    if differences and all(d["category"] == "source_vs_current" and d["field"] in PATH_FIELDS for d in differences):
        for difference in differences:
            difference["compatibility_evidence"] = same_directory_evidence(difference["expected"], difference["actual"])
        if all(d["compatibility_evidence"]["verified"] for d in differences):
            for difference in differences:
                difference["accepted"] = True
    blocked = [d for d in differences if not d["accepted"]]
    status = "rejected" if blocked else ("passed_verified_directory_alias" if differences else "passed_exact")
    return dict(report_schema_version=1, status=status, mismatch_count=len(differences), blocking_mismatch_count=len(blocked),
                mismatches=differences, expected=expected, actual=actual,
                source_policy_expected=policy_expected, source_policy_actual=policy_actual,
                compatibility_policy="only existing samefile directory aliases, with all other fields exact",
                **context)


def brief(value):
    raw = json.dumps(value, sort_keys=True, default=str)
    if len(raw) <= 240:
        return raw
    digest = hashlib.sha256(raw.encode()).hexdigest()
    return f"{type(value).__name__}(length={len(value)}, sha256={digest}); exact value in JSON report"


def comparison_message(report, path=None):
    items = [d for d in report["mismatches"] if not d["accepted"]]
    details = "; ".join(f"{d['field']}: expected={brief(d['expected']) if d['expected_present'] else '<missing>'}, actual={brief(d['actual']) if d['actual_present'] else '<missing>'}" for d in items[:20])
    extra = f"; {len(items)-20} more fields in JSON report" if len(items) > 20 else ""
    where = f"; full report: {path}" if path else ""
    return f"SSL provenance mismatch ({len(items)} blocking fields): {details}{extra}{where}"


class ProvenanceMismatchError(ValueError):
    def __init__(self, report, path=None):
        self.report = report
        self.report_path = str(path) if path else None
        super().__init__(comparison_message(report, path))
