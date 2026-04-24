from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Iterable


UNITS_MASC = {
    1: "один",
    2: "два",
    3: "три",
    4: "четыре",
    5: "пять",
    6: "шесть",
    7: "семь",
    8: "восемь",
    9: "девять",
}

UNITS_FEM = {
    1: "одна",
    2: "две",
    3: "три",
    4: "четыре",
    5: "пять",
    6: "шесть",
    7: "семь",
    8: "восемь",
    9: "девять",
}

TEENS = {
    10: "десять",
    11: "одиннадцать",
    12: "двенадцать",
    13: "тринадцать",
    14: "четырнадцать",
    15: "пятнадцать",
    16: "шестнадцать",
    17: "семнадцать",
    18: "восемнадцать",
    19: "девятнадцать",
}

TENS = {
    20: "двадцать",
    30: "тридцать",
    40: "сорок",
    50: "пятьдесят",
    60: "шестьдесят",
    70: "семьдесят",
    80: "восемьдесят",
    90: "девяносто",
}

HUNDREDS = {
    100: "сто",
    200: "двести",
    300: "триста",
    400: "четыреста",
    500: "пятьсот",
    600: "шестьсот",
    700: "семьсот",
    800: "восемьсот",
    900: "девятьсот",
}

THOUSAND_FORMS = ("тысяча", "тысячи", "тысяч")

WORD_TO_VALUE = {word: value for value, word in HUNDREDS.items()}
WORD_TO_VALUE.update({word: value for value, word in TENS.items()})
WORD_TO_VALUE.update({word: value for value, word in TEENS.items()})
WORD_TO_VALUE.update({word: value for value, word in UNITS_MASC.items()})
WORD_TO_VALUE.update({word: value for value, word in UNITS_FEM.items()})

TOKEN_RE = re.compile(r"[^\w]+", flags=re.UNICODE)


def _plural_form(value: int, forms: tuple[str, str, str]) -> str:
    mod100 = value % 100
    mod10 = value % 10
    if 11 <= mod100 <= 14:
        return forms[2]
    if mod10 == 1:
        return forms[0]
    if mod10 in (2, 3, 4):
        return forms[1]
    return forms[2]


def _chunk_to_tokens(value: int, feminine: bool = False) -> list[str]:
    if not 0 <= value <= 999:
        raise ValueError(f"Chunk must be in [0, 999], got {value}")

    tokens: list[str] = []
    hundreds = (value // 100) * 100
    if hundreds:
        tokens.append(HUNDREDS[hundreds])

    remainder = value % 100
    if 10 <= remainder <= 19:
        tokens.append(TEENS[remainder])
        return tokens

    tens = (remainder // 10) * 10
    if tens:
        tokens.append(TENS[tens])

    units = remainder % 10
    if units:
        tokens.append((UNITS_FEM if feminine else UNITS_MASC)[units])
    return tokens


def number_to_tokens(value: int) -> list[str]:
    if not 0 <= value < 1_000_000:
        raise ValueError(f"Number must be in [0, 999999], got {value}")
    if value == 0:
        return ["ноль"]

    thousands = value // 1000
    remainder = value % 1000

    tokens: list[str] = []
    if thousands:
        tokens.extend(_chunk_to_tokens(thousands, feminine=True))
        tokens.append(_plural_form(thousands, THOUSAND_FORMS))

    if remainder:
        tokens.extend(_chunk_to_tokens(remainder, feminine=False))
    return tokens


def number_to_text(value: int) -> str:
    return " ".join(number_to_tokens(value))


def normalize_words(text: str | Iterable[str]) -> list[str]:
    if isinstance(text, str):
        raw_tokens = TOKEN_RE.sub(" ", text.lower()).split()
    else:
        raw_tokens = [str(token).strip().lower() for token in text]

    aliases = {
        "одну": "одна",
        "одною": "одна",
        "двух": "две",
        "двумя": "две",
        "тыща": "тысяча",
        "тыщи": "тысячи",
    }
    return [aliases.get(token, token) for token in raw_tokens if token]


def _parse_chunk(tokens: list[str]) -> int:
    if not tokens:
        return 0

    value = 0
    index = 0

    if index < len(tokens) and tokens[index] in HUNDREDS.values():
        value += WORD_TO_VALUE[tokens[index]]
        index += 1

    if index < len(tokens) and tokens[index] in TEENS.values():
        value += WORD_TO_VALUE[tokens[index]]
        index += 1
        if index != len(tokens):
            raise ValueError(f"Unexpected tokens after teen number: {tokens}")
        return value

    if index < len(tokens) and tokens[index] in TENS.values():
        value += WORD_TO_VALUE[tokens[index]]
        index += 1

    valid_units = set(UNITS_MASC.values()) | set(UNITS_FEM.values())
    if index < len(tokens) and tokens[index] in valid_units:
        value += WORD_TO_VALUE[tokens[index]]
        index += 1

    if index != len(tokens):
        raise ValueError(f"Unable to parse number chunk: {tokens}")
    return value


def parse_number_words(text: str | Iterable[str]) -> int:
    tokens = normalize_words(text)
    if not tokens:
        raise ValueError("Empty transcript")

    if len(tokens) == 1 and tokens[0].isdigit():
        return int(tokens[0])

    if tokens == ["ноль"]:
        return 0

    thousand_index = next((i for i, token in enumerate(tokens) if token in THOUSAND_FORMS), -1)
    if thousand_index >= 0:
        thousands_tokens = tokens[:thousand_index]
        remainder_tokens = tokens[thousand_index + 1 :]
        thousands_value = _parse_chunk(thousands_tokens) if thousands_tokens else 1
        return thousands_value * 1000 + _parse_chunk(remainder_tokens)

    return _parse_chunk(tokens)


def safe_parse_number_words(text: str | Iterable[str]) -> int | None:
    try:
        return parse_number_words(text)
    except ValueError:
        return None


def all_number_words() -> list[str]:
    return sorted(
        set(UNITS_MASC.values())
        | set(UNITS_FEM.values())
        | set(TEENS.values())
        | set(TENS.values())
        | set(HUNDREDS.values())
        | set(THOUSAND_FORMS)
        | {"ноль"}
    )


def all_number_chars() -> list[str]:
    chars = set(" ")
    for word in all_number_words():
        chars.update(word)
    return sorted(chars)


@dataclass
class TrieNode:
    children: dict[str, "TrieNode"] = field(default_factory=dict)
    terminal: bool = False


def _insert_sequence(root: TrieNode, sequence: tuple[str, ...]) -> None:
    node = root
    for token in sequence:
        node = node.children.setdefault(token, TrieNode())
    node.terminal = True


def _build_trie(sequences: Iterable[tuple[str, ...]]) -> TrieNode:
    root = TrieNode()
    for sequence in sequences:
        _insert_sequence(root, sequence)
    return root


def _traverse_trie(root: TrieNode, words: tuple[str, ...]) -> TrieNode | None:
    node = root
    for word in words:
        node = node.children.get(word)
        if node is None:
            return None
    return node


@lru_cache(maxsize=1)
def masculine_chunk_sequences() -> tuple[tuple[str, ...], ...]:
    return tuple(tuple(number_to_tokens(value)) for value in range(1, 1000))


@lru_cache(maxsize=1)
def thousand_chunk_sequences() -> tuple[tuple[str, ...], ...]:
    return tuple(
        tuple(_chunk_to_tokens(value, feminine=True) + [_plural_form(value, THOUSAND_FORMS)])
        for value in range(1, 1000)
    )


class NumberGrammar:
    def __init__(self):
        self.masc_root = _build_trie(masculine_chunk_sequences())
        self.thousand_root = _build_trie(thousand_chunk_sequences())
        self.start_words = set(self.masc_root.children) | set(self.thousand_root.children) | {"ноль"}
        self.alphabet = set(all_number_chars())

    def _word_progress(self, completed_words: tuple[str, ...]) -> tuple[set[str], bool]:
        next_words: set[str] = set()
        can_end = False

        if not completed_words:
            next_words |= self.start_words
            return next_words, False

        if completed_words == ("ноль",):
            return set(), True

        masc_node = _traverse_trie(self.masc_root, completed_words)
        if masc_node is not None:
            next_words |= set(masc_node.children)
            can_end |= masc_node.terminal

        thousand_node = _traverse_trie(self.thousand_root, completed_words)
        if thousand_node is not None:
            next_words |= set(thousand_node.children)
            can_end |= thousand_node.terminal
            if thousand_node.terminal:
                next_words |= set(self.masc_root.children)

        for split_index in range(1, len(completed_words)):
            thousand_prefix = completed_words[:split_index]
            remainder_prefix = completed_words[split_index:]
            thousand_split_node = _traverse_trie(self.thousand_root, thousand_prefix)
            if thousand_split_node is None or not thousand_split_node.terminal:
                continue

            remainder_node = _traverse_trie(self.masc_root, remainder_prefix)
            if remainder_node is None:
                continue

            next_words |= set(remainder_node.children)
            can_end |= remainder_node.terminal

        return next_words, can_end

    @staticmethod
    def _split_text(text: str) -> tuple[tuple[str, ...], str] | None:
        if not text:
            return tuple(), ""
        if text[0] == " " or "  " in text:
            return None
        parts = text.split(" ")
        if text.endswith(" "):
            completed_words = tuple(part for part in parts if part)
            return completed_words, ""
        if any(part == "" for part in parts[:-1]):
            return None
        return tuple(parts[:-1]), parts[-1]

    def allowed_next_chars(self, text: str) -> set[str]:
        if any(char not in self.alphabet for char in text):
            return set()

        parsed = self._split_text(text)
        if parsed is None:
            return set()
        completed_words, fragment = parsed
        next_words, _ = self._word_progress(completed_words)

        allowed_chars: set[str] = set()
        if not fragment:
            for word in next_words:
                allowed_chars.add(word[0])
            return allowed_chars

        for word in next_words:
            if not word.startswith(fragment):
                continue

            if len(fragment) < len(word):
                allowed_chars.add(word[len(fragment)])
                continue

            next_after_word, _ = self._word_progress(completed_words + (word,))
            if next_after_word:
                allowed_chars.add(" ")

        return allowed_chars

    def is_valid_prefix(self, text: str) -> bool:
        if text == "":
            return True
        if any(char not in self.alphabet for char in text):
            return False
        parsed = self._split_text(text)
        if parsed is None:
            return False
        completed_words, fragment = parsed
        next_words, can_end = self._word_progress(completed_words)
        if not fragment:
            return can_end or bool(next_words)
        return any(word.startswith(fragment) for word in next_words)

    def is_terminal_text(self, text: str) -> bool:
        if not text or text.endswith(" "):
            return False
        parsed = self._split_text(text)
        if parsed is None:
            return False
        completed_words, fragment = parsed
        if not fragment:
            return False
        next_words, _ = self._word_progress(completed_words)
        if fragment not in next_words:
            return False
        _, can_end_after_word = self._word_progress(completed_words + (fragment,))
        return can_end_after_word


@lru_cache(maxsize=1)
def get_number_grammar() -> NumberGrammar:
    return NumberGrammar()
