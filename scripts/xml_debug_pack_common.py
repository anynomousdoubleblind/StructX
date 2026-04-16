#!/usr/bin/env python3
"""Shared helpers for XML debug pack runners (ANSI strip, parse program output)."""

import re
from typing import Optional

ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*m")


def strip_ansi(s: str) -> str:
    return ANSI_ESCAPE.sub("", s)


def parse_matches_found(stdout: str) -> Optional[int]:
    clean = strip_ansi(stdout)
    m = re.search(r"Matches found:\s*(\d+)", clean)
    return int(m.group(1)) if m else None


def parse_match_zero_text(stdout: str) -> Optional[str]:
    """Text after 'Match 0:' on the same line (first match line)."""
    clean = strip_ansi(stdout)
    m = re.search(r"Match\s+0:\s*(.*)", clean, re.MULTILINE)
    if not m:
        return None
    return m.group(1).strip()


def first_root_xpath_from_xml_snippet(xml_snippet: str) -> str:
    """Return '/TagName' for the first start tag in a compact XML string."""
    m = re.search(r"<([A-Za-z_][\w:.-]*)", xml_snippet)
    if not m:
        return "/Root"
    return "/" + m.group(1)
