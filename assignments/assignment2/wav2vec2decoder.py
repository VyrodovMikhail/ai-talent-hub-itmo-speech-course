import math
from typing import List, Tuple

import kenlm
import torch
import torchaudio
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC


# ---------------------------------------------------------------------------
# Provided utility — do NOT modify
# ---------------------------------------------------------------------------

def _log_add(a: float, b: float) -> float:
    """Numerically stable log(exp(a) + exp(b))."""
    if a == float('-inf'):
        return b
    if b == float('-inf'):
        return a
    if a > b:
        return a + math.log1p(math.exp(b - a))
    return b + math.log1p(math.exp(a - b))


class Wav2Vec2Decoder:
    def __init__(
            self,
            model_name="facebook/wav2vec2-base-100h",
            lm_model_path="lm/3-gram.pruned.1e-7.arpa.gz",
            beam_width=3,
            alpha=1.0,
            beta=1.0,
            temperature=1.0,
        ):
        """
        Args:
            model_name (str): Pretrained Wav2Vec2 model from HuggingFace.
            lm_model_path (str): Path to a KenLM .arpa/.arpa.gz model.
                Pass None to disable LM (Tasks 1–3).
            beam_width (int): Number of hypotheses kept during beam search.
            alpha (float): LM weight used in shallow fusion and rescoring.
                score = log_p_acoustic + alpha * log_p_lm + beta * num_words
            beta (float): Word insertion bonus (see above).
            temperature (float): Scales acoustic logits before softmax.
                T < 1 sharpens the distribution (model more confident).
                T > 1 flattens it (model less confident, giving LM more
                influence). T = 1.0 leaves logits unchanged.
        """
        # Interact with processor/model ONLY here and in decode() to obtain
        # logits — no further model calls are allowed anywhere else.
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model = Wav2Vec2ForCTC.from_pretrained(model_name)

        self.vocab = {i: c for c, i in self.processor.tokenizer.get_vocab().items()}
        self.blank_token_id = self.processor.tokenizer.pad_token_id
        self.word_delimiter = self.processor.tokenizer.word_delimiter_token
        self.word_delimeter_id = self.processor.tokenizer.word_delimiter_token_id
        self.beam_width = beam_width
        self.alpha = alpha
        self.beta = beta
        self.temperature = temperature
        self.lm_model = kenlm.Model(lm_model_path) if lm_model_path else None

    # -----------------------------------------------------------------------
    # Provided utility — do NOT modify
    # -----------------------------------------------------------------------

    def _ids_to_text(self, token_ids: List[int]) -> str:
        """Convert a list of token IDs to a decoded string."""
        text = ''.join(self.vocab[i] for i in token_ids)
        return text.replace(self.word_delimiter, ' ').strip().lower()

    # -----------------------------------------------------------------------
    # Tasks 1–4: implement the methods below
    # -----------------------------------------------------------------------

    def greedy_decode(self, logits: torch.Tensor) -> str:
        """
        Perform greedy decoding (find best CTC path).

        Args:
            logits (torch.Tensor): Logits from Wav2Vec2 model (T, V).

        Returns:
            str: Decoded transcript.
        """
        log_probs = torch.log_softmax(logits, dim=-1)
        tokens = torch.argmax(log_probs, dim=-1).tolist()

        # CTC collapsing: remove consecutive duplicates, then remove blanks
        result = []
        prev = None
        for t in tokens:
            if t != prev and t != self.blank_token_id:
                result.append(t)
            prev = t

        return self._ids_to_text(result)

    def beam_search_decode(self, logits: torch.Tensor, return_beams: bool = False):
        """
        Perform beam search decoding (no LM).

        Args:
            logits (torch.Tensor): Logits from Wav2Vec2 model (T, V), where
                T - number of time steps and
                V - vocabulary size.
            return_beams (bool): Return all beam hypotheses for second-pass
                LM rescoring.

        Returns:
            Union[str, List[Tuple[List[int], float]]]:
                str - best decoded transcript (if return_beams=False).
                List[Tuple[List[int], float]] - list of (token_ids, log_prob)
                    tuples sorted best-first (if return_beams=True).
        """
        log_probs = torch.log_softmax(logits, dim=-1)
        T, V = log_probs.shape
        blank = self.blank_token_id
        NEG_INF = float('-inf')

        # beams: decoded_tuple -> [pb, pnb]
        # prob_ends_with_blank  = log prob of all paths yielding this prefix that end with blank
        # prob_ends_with_char = log prob of all paths yielding this prefix that end with non-blank
        beams = {(): [0.0, NEG_INF]}  # empty prefix, prob 1 ending in blank

        for t in range(T):
            current_log_probs = log_probs[t]  # (V,)
            new_beams: dict = {}

            for decoded, (prob_ends_with_blank, prob_ends_with_char) in beams.items():
                total = _log_add(prob_ends_with_blank, prob_ends_with_char)

                # Add blank token and compute log probability
                if decoded not in new_beams:
                    new_beams[decoded] = [NEG_INF, NEG_INF]
                new_beams[decoded][0] = _log_add(
                    new_beams[decoded][0], total + current_log_probs[blank].item()
                )

                # Emit each non-blank token
                for c in range(V):
                    if c == blank:
                        continue
                    char_log_prob = current_log_probs[c].item()

                    if not decoded or c != decoded[-1]:
                        # Different from last: extend prefix
                        new_sequence = decoded + (c,)
                        if new_sequence not in new_beams:
                            new_beams[new_sequence] = [NEG_INF, NEG_INF]
                        new_beams[new_sequence][1] = _log_add(
                            new_beams[new_sequence][1], total + char_log_prob
                        )
                    else:
                        # Same as last character of decoded — two sub-cases:
                        # (a) path ended in blank: adds a new (repeated) char
                        new_sequence = decoded + (c,)
                        if new_sequence not in new_beams:
                            new_beams[new_sequence] = [NEG_INF, NEG_INF]
                        new_beams[new_sequence][1] = _log_add(
                            new_beams[new_sequence][1], prob_ends_with_blank + char_log_prob
                        )
                        # (b) path ended in non-blank: repeating char, stays same prefix
                        if decoded not in new_beams:
                            new_beams[decoded] = [NEG_INF, NEG_INF]
                        new_beams[decoded][1] = _log_add(
                            new_beams[decoded][1], prob_ends_with_char + char_log_prob
                        )

            # Prune to beam_width by total acoustic score
            beams = dict(
                sorted(
                    new_beams.items(),
                    key=lambda x: _log_add(x[1][0], x[1][1]),
                    reverse=True,
                )[:self.beam_width]
            )

        # Sort final beams best-first
        scored = sorted(
            beams.items(),
            key=lambda x: _log_add(x[1][0], x[1][1]),
            reverse=True,
        )

        if return_beams:
            return [(list(decoded_seq), _log_add(prob_ends_with_blank, prob_ends_with_char)) for decoded_seq, (prob_ends_with_blank, prob_ends_with_char) in scored]

        return self._ids_to_text(list(scored[0][0]))

    def beam_search_with_lm(self, logits: torch.Tensor) -> str:
        """
        Perform beam search decoding with shallow LM fusion.

        Args:
            logits (torch.Tensor): Logits from Wav2Vec2 model (T, V), where
                T - number of time steps and
                V - vocabulary size.

        Returns:
            str: Decoded transcript.
        """
        if not self.lm_model:
            raise ValueError("KenLM model required for LM shallow fusion")

        log_probs = torch.log_softmax(logits, dim=-1)
        T, V = log_probs.shape
        blank = self.blank_token_id
        NEG_INF = float('-inf')

        # Acoustic beams: decoded_tuple -> [pb, pnb]
        acoustics_beams = {(): [0.0, NEG_INF]}

        # LM data cache: decoded_tuple -> (kenlm_state, lm_log_prob, word_count)
        # lm_log_prob: accumulated LM log-prob in natural log units
        init_lm_state = kenlm.State()
        self.lm_model.BeginSentenceWrite(init_lm_state)
        lm_data: dict = {(): (init_lm_state, 0.0, 0)}

        def ensure_lm(decoded: tuple, c: int) -> None:
            """Populate lm_data for decoded+(c,) from parent decoded's LM state."""
            new_sequence = decoded + (c,)
            if new_sequence in lm_data:
                return
            parent_state, parent_score, parent_word_count = lm_data[decoded]
            if c == self.word_delimeter_id:
                # A word delimiter was just appended — score the completed word
                chars = []
                for tok in reversed(decoded):
                    if tok == self.word_delimeter_id:
                        break
                    chars.append(self.vocab[tok])
                word = ''.join(reversed(chars)).lower()
                if word:
                    out_state = kenlm.State()
                    log10_p = self.lm_model.BaseScore(parent_state, word, out_state)
                    lm_data[new_sequence] = (out_state, parent_score + log10_p * math.log(10), parent_word_count + 1)
                else:
                    lm_data[new_sequence] = (parent_state, parent_score, parent_word_count)
            else:
                lm_data[new_sequence] = (parent_state, parent_score, parent_word_count)

        def combined_score(decoded: tuple, pb: float, pnb: float) -> float:
            _, lm_s, wc = lm_data[decoded]
            return _log_add(pb, pnb) + self.alpha * lm_s + self.beta * wc

        for t in range(T):
            lp = log_probs[t]
            new_ac: dict = {}

            for decoded, (pb, pnb) in acoustics_beams.items():
                total = _log_add(pb, pnb)

                # Emit blank — prefix unchanged
                if decoded not in new_ac:
                    new_ac[decoded] = [NEG_INF, NEG_INF]
                new_ac[decoded][0] = _log_add(
                    new_ac[decoded][0], total + lp[blank].item()
                )

                # Emit each non-blank token
                for c in range(V):
                    if c == blank:
                        continue
                    c_lp = lp[c].item()

                    if not decoded or c != decoded[-1]:
                        nd = decoded + (c,)
                        ensure_lm(decoded, c)
                        if nd not in new_ac:
                            new_ac[nd] = [NEG_INF, NEG_INF]
                        new_ac[nd][1] = _log_add(new_ac[nd][1], total + c_lp)
                    else:
                        # (a) was after blank: new repeated char
                        nd = decoded + (c,)
                        ensure_lm(decoded, c)
                        if nd not in new_ac:
                            new_ac[nd] = [NEG_INF, NEG_INF]
                        new_ac[nd][1] = _log_add(new_ac[nd][1], pb + c_lp)
                        # (b) repeat non-blank: prefix stays
                        if decoded not in new_ac:
                            new_ac[decoded] = [NEG_INF, NEG_INF]
                        new_ac[decoded][1] = _log_add(
                            new_ac[decoded][1], pnb + c_lp
                        )

            # Prune by combined score
            beams_list = sorted(
                new_ac.items(),
                key=lambda x: combined_score(x[0], x[1][0], x[1][1]),
                reverse=True,
            )
            acoustics_beams = dict(beams_list[:self.beam_width])

        # Final: use full-sentence KenLM scoring so the final shallow-fusion
        # decision matches the same BOS/EOS convention used in rescoring.
        final_scored = []
        for decoded, (pb, pnb) in acoustics_beams.items():
            text = self._ids_to_text(list(decoded))
            wc = len(text.split())
            lm_s = self.lm_model.score(text, bos=True, eos=True) * math.log(10)
            score = _log_add(pb, pnb) + self.alpha * lm_s + self.beta * wc
            final_scored.append((decoded, score))

        final_scored.sort(key=lambda x: x[1], reverse=True)
        return self._ids_to_text(list(final_scored[0][0]))

    def lm_rescore(self, beams: List[Tuple[List[int], float]]) -> str:
        """
        Perform second-pass LM rescoring on beam search outputs.

        Args:
            beams (List[Tuple[List[int], float]]): List of (token_ids, log_prob)
                tuples from beam_search_decode(logits, return_beams=True).

        Returns:
            str: Best rescored transcript.
        """
        if not self.lm_model:
            raise ValueError("KenLM model required for LM rescoring")

        best_score = float('-inf')
        best_text = ""

        for token_ids, acoustic_score in beams:
            text = self._ids_to_text(token_ids)
            word_count = len(text.split())
            # KenLM returns log10 prob; convert to natural log for consistent units
            lm_score = self.lm_model.score(text, bos=True, eos=True) * math.log(10)
            combined = acoustic_score + self.alpha * lm_score + self.beta * word_count
            if combined > best_score:
                best_score = combined
                best_text = text

        return best_text

    # -----------------------------------------------------------------------
    # Provided — do NOT modify
    # -----------------------------------------------------------------------

    def decode(self, audio_input: torch.Tensor, method: str = "greedy") -> str:
        """
        Run the full decoding pipeline on a raw audio tensor.

        Args:
            audio_input (torch.Tensor): 1-D or 2-D audio waveform at 16 kHz.
            method (str): One of "greedy", "beam", "beam_lm", "beam_lm_rescore".

        Returns:
            str: Decoded transcript (lowercase).
        """
        inputs = self.processor(audio_input, return_tensors="pt", sampling_rate=16000)
        with torch.no_grad():
            logits = self.model(inputs.input_values.squeeze(0)).logits[0]

        # Temperature scaling (Task 3): flatten/sharpen the distribution
        # before log_softmax.  T=1.0 is a no-op.  Your decoders must call
        # torch.log_softmax on the logits they receive — do not call it here.
        logits = logits / self.temperature

        if method == "greedy":
            return self.greedy_decode(logits)
        elif method == "beam":
            return self.beam_search_decode(logits)
        elif method == "beam_lm":
            return self.beam_search_with_lm(logits)
        elif method == "beam_lm_rescore":
            beams = self.beam_search_decode(logits, return_beams=True)
            return self.lm_rescore(beams)
        else:
            raise ValueError(
                f"Unknown method '{method}'. "
                "Choose one of: 'greedy', 'beam', 'beam_lm', 'beam_lm_rescore'."
            )


# ---------------------------------------------------------------------------
# Quick debug helper — run this file directly to sanity-check your decoder
# on the provided examples/ clips before evaluating on the full test sets.
# ---------------------------------------------------------------------------

def test(decoder: Wav2Vec2Decoder, audio_path: str, reference: str) -> None:
    import jiwer

    audio_input, sr = torchaudio.load(audio_path)
    assert sr == 16000, f"Expected 16 kHz, got {sr} Hz for {audio_path}"

    print("=" * 60)
    print(f"REF : {reference}")

    for method in ["greedy", "beam", "beam_lm", "beam_lm_rescore"]:
        try:
            hyp = decoder.decode(audio_input, method=method)
        except NotImplementedError:
            print(f"  [{method}] not yet implemented")
            continue
        except ValueError as e:
            print(f"  [{method}] skipped ({e})")
            continue
        cer = jiwer.cer(reference, hyp)
        wer = jiwer.wer(reference, hyp)
        print(f"  [{method}] {hyp}")
        print(f"           WER={wer:.2%}  CER={cer:.2%}")


if __name__ == "__main__":
    # Reference transcripts are lowercase to match the evaluation manifests.
    # examples/ clips are for quick debugging only — use data/librispeech_test_other/
    # and data/earnings22_test/ for all reported metrics.
    test_samples = [
        ("examples/sample1.wav", "if you are generous here is a fitting opportunity for the exercise of your magnanimity if you are proud here am i your rival ready to acknowledge myself your debtor for an act of the most noble forbearance"),
        ("examples/sample2.wav", "and if any of the other cops had private rackets of their own izzy was undoubtedly the man to find it out and use the information with a beat such as that even going halves and with all the graft to the upper brackets he'd still be able to make his pile in a matter of months"),
        ("examples/sample3.wav", "guess a man gets used to anything hell maybe i can hire some bums to sit around and whoop it up when the ships come in and bill this as a real old martian den of sin"),
        ("examples/sample4.wav", "it was a tune they had all heard hundreds of times so there was no difficulty in turning out a passable imitation of it to the improvised strains of i didn't want to do it the prisoner strode forth to freedom"),
        ("examples/sample5.wav", "marguerite tired out with this long confession threw herself back on the sofa and to stifle a slight cough put up her handkerchief to her lips and from that to her eyes"),
        ("examples/sample6.wav", "at this time all participants are in a listen only mode"),
        ("examples/sample7.wav", "the increase was mainly attributable to the net increase in the average size of our fleets"),
        ("examples/sample8.wav", "operating surplus is a non cap financial measure which is defined as fully in our press release"),
    ]

    decoder = Wav2Vec2Decoder(lm_model_path="lm/3-gram.pruned.1e-7.arpa.gz")  # set lm_model_path for Tasks 4+

    for audio_path, reference in test_samples:
        test(decoder, audio_path, reference)
