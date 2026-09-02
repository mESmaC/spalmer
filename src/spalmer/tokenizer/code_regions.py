"""Small Python-aware region router for tokenizer construction and encoding."""

from __future__ import annotations

import ast
import io
import keyword
import re
import tokenize
from dataclasses import dataclass
from typing import Literal

from .tiers import Tier

RegionRole = Literal["mixed", "code", "prose", "fallback", "atom"]
InputKind = Literal["mixed", "prose", "code"]

_STRING_OPEN_RE = re.compile(r"(?i)^([rubf]*)(\"\"\"|'''|\"|')")


@dataclass(frozen=True, slots=True)
class RoutedRegion:
    start: int
    end: int
    role: RegionRole


@dataclass(frozen=True, slots=True)
class PythonCodeAnalysis:
    prose_chunks: tuple[str, ...]
    identifiers: tuple[str, ...]
    regions: tuple[RoutedRegion, ...]


def route_regions(text: str, kind: InputKind) -> tuple[RoutedRegion, ...]:
    """Return contiguous regions with the tier role each region may use."""

    if kind == "mixed":
        return (RoutedRegion(0, len(text), "mixed"),) if text else ()
    if kind == "prose":
        return (RoutedRegion(0, len(text), "prose"),) if text else ()
    if kind != "code":
        raise ValueError(f"unknown input kind: {kind!r}")
    return analyze_python_code(text).regions


def tiers_for_role(tiers: tuple[Tier, ...], role: RegionRole) -> tuple[Tier, ...]:
    """Intersect a caller's tier set with the routing contract for one region."""

    if role == "mixed":
        return tiers
    excluded = {
        "prose": {Tier.LEXER},
        "code": {Tier.PHRASE},
        "fallback": {Tier.LEXER, Tier.PHRASE, Tier.WORD},
        "atom": {
            Tier.LEXER,
            Tier.PHRASE,
            Tier.WORD,
            Tier.SALVAGE,
            Tier.LANGUAGE_FALLBACK,
        },
    }[role]
    return tuple(tier for tier in tiers if tier not in excluded)


def analyze_python_code(text: str) -> PythonCodeAnalysis:
    """Identify semantic prose, identifiers, and runtime routing regions.

    The standard tokenizer distinguishes comments and strings without regex
    leakage. The AST supplies the one semantic distinction tokenization alone
    cannot: whether a string token is a real module/class/function docstring.
    Malformed source is handled best-effort using every token produced before
    the malformed tail.
    """

    comments: list[str] = []
    identifiers: list[str] = []
    roles: list[RegionRole] = ["code"] * len(text)
    offsets = _line_offsets(text)
    tree = _parse_python(text)
    docstring_chunks, docstring_ranges = _find_docstrings(text, offsets, tree)
    if tree is not None and not hasattr(tokenize, "FSTRING_START"):
        identifiers.extend(_formatted_value_identifiers(tree))

    try:
        tokens = tokenize.generate_tokens(io.StringIO(text).readline)
        for token_info in tokens:
            start = _absolute_offset(offsets, token_info.start, len(text))
            end = _absolute_offset(offsets, token_info.end, len(text))
            if token_info.type == tokenize.COMMENT:
                _mark(roles, start, min(start + 1, end), "atom")
                _mark(roles, min(start + 1, end), end, "prose")
                chunk = token_info.string.lstrip("#")
                if chunk.strip():
                    comments.append(chunk)
            elif token_info.type == tokenize.STRING:
                _mark_string(
                    roles,
                    start,
                    end,
                    token_info.string,
                    content_role=(
                        "prose"
                        if any(
                            region_start <= start and end <= region_end
                            for region_start, region_end in docstring_ranges
                        )
                        else "fallback"
                    ),
                )
            elif token_info.type == getattr(tokenize, "FSTRING_START", -1):
                _mark(roles, start, end, "atom")
            elif token_info.type == getattr(tokenize, "FSTRING_MIDDLE", -1):
                _mark(roles, start, end, "fallback")
            elif token_info.type == getattr(tokenize, "FSTRING_END", -1):
                _mark(roles, start, end, "atom")
            elif token_info.type == tokenize.NAME and not keyword.iskeyword(token_info.string):
                identifiers.append(token_info.string)
    except (IndentationError, tokenize.TokenError):
        pass

    regions: list[RoutedRegion] = []
    if roles:
        start = 0
        role = roles[0]
        for position in range(1, len(roles)):
            if roles[position] != role:
                regions.append(RoutedRegion(start, position, role))
                start = position
                role = roles[position]
        regions.append(RoutedRegion(start, len(roles), role))

    return PythonCodeAnalysis(
        prose_chunks=tuple(comments + [chunk for _, _, chunk in docstring_chunks]),
        identifiers=tuple(identifiers),
        regions=tuple(regions),
    )


def _parse_python(text: str) -> ast.AST | None:
    try:
        return ast.parse(text)
    except (SyntaxError, ValueError):
        return None


def _find_docstrings(
    text: str,
    offsets: list[int],
    tree: ast.AST | None,
) -> tuple[list[tuple[int, int, str]], list[tuple[int, int]]]:
    if tree is None:
        return [], []

    found: list[tuple[int, int, str]] = []
    ranges: list[tuple[int, int]] = []
    owners = (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
    for node in ast.walk(tree):
        if not isinstance(node, owners) or not node.body:
            continue
        expression = node.body[0]
        if not (
            isinstance(expression, ast.Expr)
            and isinstance(expression.value, ast.Constant)
            and isinstance(expression.value.value, str)
        ):
            continue
        found.append((expression.lineno, expression.col_offset, expression.value.value))
        start = _absolute_ast_offset(
            text,
            offsets,
            (expression.lineno, expression.col_offset),
        )
        end = _absolute_ast_offset(
            text,
            offsets,
            (
                getattr(expression, "end_lineno", expression.lineno),
                getattr(expression, "end_col_offset", expression.col_offset),
            ),
        )
        ranges.append((start, end))
    found.sort()
    return found, ranges


def _formatted_value_identifiers(tree: ast.AST) -> list[str]:
    """Recover f-string expression names on Python 3.10/3.11 tokenizers."""

    identifiers: list[str] = []
    for formatted in ast.walk(tree):
        if not isinstance(formatted, ast.FormattedValue):
            continue
        for node in ast.walk(formatted.value):
            if isinstance(node, ast.Name):
                identifiers.append(node.id)
            elif isinstance(node, ast.Attribute):
                identifiers.append(node.attr)
    return identifiers


def _line_offsets(text: str) -> list[int]:
    offsets = [0]
    for line in text.splitlines(keepends=True):
        offsets.append(offsets[-1] + len(line))
    return offsets


def _absolute_offset(offsets: list[int], position: tuple[int, int], limit: int) -> int:
    line, column = position
    if line <= 0 or line - 1 >= len(offsets):
        return limit
    return min(offsets[line - 1] + column, limit)


def _absolute_ast_offset(
    text: str, offsets: list[int], position: tuple[int, int]
) -> int:
    """Translate AST UTF-8 byte columns to Python string offsets."""

    line, byte_column = position
    if line <= 0 or line - 1 >= len(offsets):
        return len(text)
    line_start = offsets[line - 1]
    line_stop = offsets[line] if line < len(offsets) else len(text)
    line_text = text[line_start:line_stop]
    prefix = line_text.encode("utf-8")[:byte_column].decode("utf-8")
    return min(line_start + len(prefix), len(text))


def _mark(roles: list[RegionRole], start: int, end: int, role: RegionRole) -> None:
    roles[start:end] = [role] * max(0, end - start)


def _mark_string(
    roles: list[RegionRole],
    start: int,
    end: int,
    surface: str,
    *,
    content_role: RegionRole,
) -> None:
    match = _STRING_OPEN_RE.match(surface)
    if match is None:
        _mark(roles, start, end, content_role)
        return
    prefix, quote = match.groups()
    content_start = min(start + len(prefix) + len(quote), end)
    content_end = max(content_start, end - len(quote))
    _mark(roles, start, content_start, "atom")
    _mark(roles, content_start, content_end, content_role)
    _mark(roles, content_end, end, "atom")
    if "f" in prefix.lower() and not hasattr(tokenize, "FSTRING_START"):
        content = surface[len(prefix) + len(quote) : len(surface) - len(quote)]
        _mark_legacy_fstring_fields(roles, content, content_start)


def _mark_legacy_fstring_fields(
    roles: list[RegionRole], content: str, content_start: int
) -> None:
    """Route pre-3.12 f-string expressions without treating format text as code."""

    for field_start, field_end in _legacy_fstring_fields(content):
        inner_start = field_start + 1
        inner_end = field_end - 1
        expression_end, format_colon = _legacy_fstring_boundaries(
            content, inner_start, inner_end
        )

        # Replacement braces, conversion flags, and the format separator are
        # syntax. Only the expression itself is recursively code-routed; the
        # format payload keeps the fallback role assigned to the string body.
        _mark(roles, content_start + field_start, content_start + inner_start, "code")
        _mark(
            roles,
            content_start + inner_start,
            content_start + expression_end,
            "code",
        )
        expression = content[inner_start:expression_end]
        for region in analyze_python_code(expression).regions:
            _mark(
                roles,
                content_start + inner_start + region.start,
                content_start + inner_start + region.end,
                region.role,
            )

        syntax_end = inner_end if format_colon is None else format_colon + 1
        _mark(
            roles,
            content_start + expression_end,
            content_start + syntax_end,
            "code",
        )
        _mark(roles, content_start + inner_end, content_start + field_end, "code")

        if format_colon is not None:
            format_start = format_colon + 1
            _mark_legacy_fstring_fields(
                roles,
                content[format_start:inner_end],
                content_start + format_start,
            )


def _legacy_fstring_boundaries(
    content: str, start: int, end: int
) -> tuple[int, int | None]:
    """Find a field's expression end and optional top-level format colon."""

    position = start
    expression_end = end
    format_colon: int | None = None
    brackets: list[str] = []
    quote = ""
    escaped = False
    matching = {")": "(", "]": "[", "}": "{"}

    while position < end:
        character = content[position]
        if quote:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif content.startswith(quote, position):
                position += len(quote)
                quote = ""
                continue
        elif character in {'"', "'"}:
            triple = character * 3
            quote = triple if content.startswith(triple, position) else character
            position += len(quote)
            continue
        elif character in "([{":
            brackets.append(character)
        elif character in matching and brackets and brackets[-1] == matching[character]:
            brackets.pop()
        elif not brackets:
            if character == "!" and not content.startswith("!=", position):
                if expression_end == end:
                    expression_end = position
            elif character == ":":
                if expression_end == end:
                    expression_end = position
                format_colon = position
                break
        position += 1

    return expression_end, format_colon


def _legacy_fstring_fields(content: str) -> list[tuple[int, int]]:
    """Locate replacement fields when pre-3.12 tokenization returns one STRING."""

    fields: list[tuple[int, int]] = []
    position = 0
    while position < len(content):
        if content.startswith("{{", position) or content.startswith("}}", position):
            position += 2
            continue
        if content[position] != "{":
            position += 1
            continue
        start = position
        position += 1
        depth = 1
        quote = ""
        escaped = False
        while position < len(content) and depth:
            character = content[position]
            if quote:
                if escaped:
                    escaped = False
                elif character == "\\":
                    escaped = True
                elif content.startswith(quote, position):
                    position += len(quote)
                    quote = ""
                    continue
            elif character in {'"', "'"}:
                triple = character * 3
                quote = triple if content.startswith(triple, position) else character
                position += len(quote)
                continue
            elif character == "{":
                depth += 1
            elif character == "}":
                depth -= 1
            position += 1
        if depth == 0:
            fields.append((start, position))
    return fields


__all__ = [
    "InputKind",
    "PythonCodeAnalysis",
    "RegionRole",
    "RoutedRegion",
    "analyze_python_code",
    "route_regions",
    "tiers_for_role",
]
