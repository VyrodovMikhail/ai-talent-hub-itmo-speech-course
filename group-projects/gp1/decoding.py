from __future__ import annotations

import math
from collections import defaultdict
from typing import Iterable

import kenlm
import torch

from .data_utils import CTCVocabulary
from .numbers import get_number_grammar


def _log_add(a: float, b: float) -> float:
    if a == float("-inf"):
        return b
    if b == float("-inf"):
        return a
    if a > b:
        return a + math.log1p(math.exp(b - a))
    return b + math.log1p(math.exp(a - b))


class CTCDecoder:
    def __init__(
        self,
        vocabulary: CTCVocabulary,
        beam_width: int = 16,
        beam_size_token: int | None = None,
        use_constraints: bool = False,
        alpha: float = 0.6,
        beta: float = 0.0,
        lm_path: str | None = None,
    ):
        self.vocabulary = vocabulary
        self.blank_id = vocabulary.blank_id
        self.beam_width = beam_width
        self.beam_size_token = beam_size_token
        self.use_constraints = use_constraints and vocabulary.token_type == "char"
        self.alpha = alpha
        self.beta = beta
        self.lm_model = kenlm.Model(lm_path) if lm_path else None
        self.grammar = get_number_grammar() if self.use_constraints else None

    def _ids_to_text(self, token_ids: Iterable[int]) -> str:
        return self.vocabulary.ids_to_text(list(token_ids))

    def _lm_score(self, token_ids: Iterable[int], eos: bool = False) -> float:
        if self.lm_model is None:
            return 0.0
        text = self._ids_to_text(token_ids)
        if not text:
            return 0.0
        return self.lm_model.score(text, bos=True, eos=eos)

    @staticmethod
    def _collapse_repeats(token_ids: list[int], blank_id: int) -> list[int]:
        collapsed: list[int] = []
        prev = None
        for token_id in token_ids:
            if token_id == blank_id:
                prev = None
                continue
            if token_id == prev:
                continue
            collapsed.append(token_id)
            prev = token_id
        return collapsed

    def greedy_decode(self, logits: torch.Tensor) -> str:
        token_ids = logits.argmax(dim=-1).tolist()
        collapsed = self._collapse_repeats(token_ids, self.blank_id)
        return self._ids_to_text(collapsed)

    def _allowed_token_ids(self, prefix: tuple[int, ...], frame_probs: torch.Tensor) -> list[int]:
        token_ids = [token_id for token_id in range(frame_probs.size(0)) if token_id != self.blank_id]
        if self.use_constraints and self.grammar is not None:
            prefix_text = self._ids_to_text(prefix)
            allowed_chars = self.grammar.allowed_next_chars(prefix_text)
            token_ids = [
                self.vocabulary.token_to_id[char]
                for char in sorted(allowed_chars)
                if char in self.vocabulary.token_to_id
            ]
        if self.beam_size_token is not None and self.beam_size_token < len(token_ids):
            frame_non_blank = frame_probs.clone()
            frame_non_blank[self.blank_id] = float("-inf")
            shortlisted = torch.topk(frame_non_blank, k=min(self.beam_size_token, frame_non_blank.numel() - 1)).indices.tolist()
            shortlisted = {token_id for token_id in shortlisted if token_id != self.blank_id}
            token_ids = [token_id for token_id in token_ids if token_id in shortlisted]
        return token_ids

    def _finalize_beams(self, beams: dict[tuple[int, ...], tuple[float, float]], constrained: bool):
        beam_list = [
            (list(prefix), _log_add(p_blank, p_non_blank))
            for prefix, (p_blank, p_non_blank) in beams.items()
        ]
        beam_list.sort(key=lambda item: item[1], reverse=True)
        if constrained and self.grammar is not None:
            terminal_beams = [
                beam for beam in beam_list
                if self.grammar.is_terminal_text(self._ids_to_text(beam[0]))
            ]
            if terminal_beams:
                beam_list = terminal_beams
        return beam_list

    def beam_search_decode(self, logits: torch.Tensor, return_beams: bool = False, constrained: bool | None = None):
        constrained = self.use_constraints if constrained is None else constrained
        log_probs = torch.log_softmax(logits, dim=-1)
        beams: dict[tuple[int, ...], tuple[float, float]] = {(): (0.0, float("-inf"))}

        for frame in range(log_probs.size(0)):
            frame_probs = log_probs[frame]
            next_beams: dict[tuple[int, ...], tuple[float, float]] = defaultdict(
                lambda: (float("-inf"), float("-inf"))
            )

            for prefix, (p_blank, p_non_blank) in beams.items():
                if constrained and self.grammar is not None:
                    prefix_text = self._ids_to_text(prefix)
                    if prefix and not self.grammar.is_valid_prefix(prefix_text):
                        continue

                prefix_score = _log_add(p_blank, p_non_blank)
                next_blank, next_non_blank = next_beams[prefix]
                next_beams[prefix] = (
                    _log_add(next_blank, prefix_score + frame_probs[self.blank_id].item()),
                    next_non_blank,
                )

                end_token = prefix[-1] if prefix else None
                frame_token_ids = self._allowed_token_ids(prefix, frame_probs) if constrained else self._allowed_token_ids(tuple(), frame_probs)
                for token_id in frame_token_ids:
                    token_score = frame_probs[token_id].item()
                    if token_id == end_token:
                        same_blank, same_non_blank = next_beams[prefix]
                        next_beams[prefix] = (
                            same_blank,
                            _log_add(same_non_blank, p_non_blank + token_score),
                        )
                        extended_prefix = prefix + (token_id,)
                        ext_blank, ext_non_blank = next_beams[extended_prefix]
                        next_beams[extended_prefix] = (
                            ext_blank,
                            _log_add(ext_non_blank, p_blank + token_score),
                        )
                    else:
                        extended_prefix = prefix + (token_id,)
                        ext_blank, ext_non_blank = next_beams[extended_prefix]
                        next_beams[extended_prefix] = (
                            ext_blank,
                            _log_add(ext_non_blank, prefix_score + token_score),
                        )

            ranked = sorted(
                next_beams.items(),
                key=lambda item: _log_add(item[1][0], item[1][1]),
                reverse=True,
            )
            if not ranked:
                if constrained:
                    return self.beam_search_decode(logits, return_beams=return_beams, constrained=False)
                if return_beams:
                    return [([], float("-inf"))]
                return ""
            beams = dict(ranked[: self.beam_width])

        beam_list = self._finalize_beams(beams, constrained=constrained)
        if not beam_list:
            if constrained:
                return self.beam_search_decode(logits, return_beams=return_beams, constrained=False)
            if return_beams:
                return [([], float("-inf"))]
            return ""
        if return_beams:
            return beam_list
        return self._ids_to_text(beam_list[0][0])

    def constrained_beam_search_decode(self, logits: torch.Tensor, return_beams: bool = False):
        if self.grammar is None:
            raise ValueError("Constrained decoding requires a character-level vocabulary")
        return self.beam_search_decode(logits, return_beams=return_beams, constrained=True)

    def beam_search_with_lm(self, logits: torch.Tensor) -> str:
        if self.lm_model is None:
            raise ValueError("KenLM model path is required for LM fusion")

        beams = self.beam_search_decode(logits, return_beams=True)
        best_tokens = max(
            beams,
            key=lambda item: item[1] + self.alpha * self._lm_score(item[0], eos=True) + self.beta * len(item[0]),
        )[0]
        return self._ids_to_text(best_tokens)

    def lm_rescore(self, beams: list[tuple[list[int], float]]) -> str:
        if self.lm_model is None:
            raise ValueError("KenLM model path is required for LM rescoring")
        best_tokens = max(
            beams,
            key=lambda item: item[1] + self.alpha * self._lm_score(item[0], eos=True) + self.beta * len(item[0]),
        )[0]
        return self._ids_to_text(best_tokens)
