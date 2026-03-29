# Assignment 2 Report

## Task 1. Greedy Decoding

Computerd metrics using greedy decoding on the LibriSpeech `test-other` dataset:

| Method | WER | CER |
|---|---:|---:|
| Greedy | 11.22% | 3.81% |


## Task 2. Beam Search Decoding

Computerd metrics using beam search on the LibriSpeech `test-other` dataset:

| Method | WER | CER | Time spent |
|---|---:|---:|---:|
| Beam-1 | 11.24% | 3.80% | 1m 5s |
| Beam-3 | 11.15% | 3.78% | 1m 20s |
| Beam-10 | 11.07% | 3.77% | 2m 0s |
| Beam-50 | 11.10% | 3.77% | 4m 5s |

Observations:

- Increasing `beam_width` from `1` to `10` improved WER slightly.
- Increasing further to `50` did not help and slightly hurt WER, while CER stayed the same.
- The best beam width in the current results is `10`.

This matches the expected trade-off: moderate beam widths help, but very large beams do not automatically improve quality enough to justify the extra compute.

## Task 3. Temperature Sweep for Greedy Decoding

Greedy decoding on LibriSpeech was completely flat across all tested temperatures:

| Temperature | WER | CER |
|---|---:|---:|
| 0.5 | 11.22% | 3.81% |
| 0.8 | 11.22% | 3.81% |
| 1.0 | 11.22% | 3.81% |
| 1.2 | 11.22% | 3.81% |
| 1.5 | 11.22% | 3.81% |
| 2.0 | 11.22% | 3.81% |

Answer: temperature had effectively no effect on greedy decoding in these experiments. This is expected because greedy decoding depends only on the argmax token at each frame, and temperature scaling preserves the same argmax ordering even though it changes confidence values.

## Task 4. Shallow Fusion with the 3-gram LM

Best shallow-fusion result on LibriSpeech:

| LM | Alpha | Beta | WER | CER |
|---|---:|---:|---:|---:|
| 3-gram | 0.10 | 0.50 | 10.98% | 3.74% |

Observations from the sweep:

- Small LM weights were best.
- Very large `alpha` values decreased performance drastically.
- At `alpha = 5.0`, WER collapsed to about `99.98%` for all tested `beta` values.

This shows that the acoustic model is already strong in-domain, so the optimal LM weight is small. Once the LM dominates, decoding quality collapses.

## Task 5. 3-gram vs 4-gram LM

Comparison on LibriSpeech with the best shallow-fusion hyperparameters from Task 4:

| LM | WER | CER |
|---|---:|---:|
| 3-gram | 10.98% | 3.74% |
| 4-gram | 11.02% | 3.75% |

Answer: the 4-gram LM did not improve over the 3-gram LM in the current results. The 3-gram model was slightly better.

## Task 6. LM Rescoring

Best rescoring result on LibriSpeech:

| Method | Alpha | Beta | WER | CER |
|---|---:|---:|---:|---:|
| Rescoring | 0.01 | 1.50 | 11.00% | 3.74% |

Comparison with shallow fusion:

| Method | Best WER | Best CER |
|---|---:|---:|
| Shallow fusion | 10.98% | 3.74% |
| Rescoring | 11.00% | 3.74% |

So in these results, shallow fusion is very slightly better on WER, while CER is effectively tied.

### Which is more stable to large `alpha` values and why?

Rescoring is much more stable.

- Shallow fusion worst case: `alpha = 5.0` gave about `99.98%` WER.
- Rescoring worst case: `alpha = 5.0` stayed around `12.86%` WER.

Reason: shallow fusion injects the LM directly into beam expansion at every time step, so an overly strong LM can completely dominate the search. Rescoring applies the LM only after the acoustic beam is already formed, so the acoustic model still constrains the candidate list.

### Qualitative Error Analysis

From `results/task6_qualitative.csv`, rescoring produced:

- `9` improved examples
- `1` changed example with the same WER
- `0` worse examples in the saved qualitative set

Patterns in the examples:

- The LM mostly fixed spacing and word-boundary errors:
  - `today` vs `to day`
  - `gurfather` vs `gur father`
  - `crewsprang` vs `crew sprang`
  - `doit` vs `do it`
  - `merrymaking` vs `merry making`
  - `lobsterboat` vs `lobster boat`
- The LM also fixed a few missing function words, such as inserting `a` in `after a little`.
- The LM did not reliably fix rare-name or acoustically difficult tokens:
  - `gur` remained `gur`
  - `shrowded` remained `shrowded`
  - `aboade` remained `aboade`

Answers:

- What kinds of errors does the LM tend to fix?
  - Mostly word-boundary and spacing errors, plus some local phrase regularization.
- What kinds of errors does it fail to fix or make worse?
  - Proper nouns or rare words.
- Are there cases where shallow fusion and rescoring disagree?
  - Yes, they sometimes disagree.

## Task 7. Cross-Domain Comparison

Cross-domain table from `results/task7_cross_domain_table.csv`:

| Method | LibriSpeech WER | LibriSpeech CER | Earnings22 WER | Earnings22 CER |
|---|---:|---:|---:|---:|
| Greedy | 11.22% | 3.81% | 54.97% | 25.58% |
| Beam | 11.07% | 3.77% | 54.94% | 25.38% |
| Beam + 3-gram (shallow fusion) | 10.98% | 3.74% | 55.57% | 25.47% |
| Beam + 3-gram (rescoring) | 11.00% | 3.74% | 55.33% | 25.38% |

Answer: there is a very large in-domain vs out-of-domain gap. On LibriSpeech, the LM helps slightly. On Earnings22, the LibriSpeech LM does not help and actually hurts WER relative to plain beam search.

Why does the LibriSpeech LM provide almost no benefit on financial speech?

- The LM was trained on LibriSpeech text, so its word and phrase statistics match audiobook-style English, not earnings-call language.
- Financial speech contains domain-specific terminology, company names, abbreviations, and disfluencies that the LibriSpeech LM does not model well.
- Because of that mismatch, LM guidance pushes decoding toward fluent but wrong in-domain text patterns.

## Task 7b. Temperature Sweep on Earnings22

Results:

| Method | Temperature | WER | CER |
|---|---:|---:|---:|
| Greedy | 0.5 | 54.97% | 25.58% |
| Greedy | 1.0 | 54.97% | 25.58% |
| Greedy | 1.5 | 54.97% | 25.58% |
| Greedy | 2.0 | 54.97% | 25.58% |
| Beam + 3-gram LM | 0.5 | 55.18% | 25.54% |
| Beam + 3-gram LM | 1.0 | 55.57% | 25.47% |
| Beam + 3-gram LM | 1.5 | 56.60% | 25.58% |
| Beam + 3-gram LM | 2.0 | 58.40% | 25.75% |

Answers:

- Does higher temperature help or hurt LM fusion on out-of-domain speech, and why?
  - It hurts in these results. WER gets worse as temperature increases from `0.5` to `2.0`.
- On LibriSpeech the acoustic model was flat under greedy decoding. Is the same true for Earnings22?
  - Yes for greedy decoding: the curve is flat again.
- What about LM fusion?
  - For LM fusion, higher temperature gives the mismatched LibriSpeech LM more influence, which makes out-of-domain performance worse.

## Task 8. Financial-Domain LM

The model was successfuly tuned and saved to the `lm/financial-3gram.arpa.gz` file.

## Task 9. Compare Both LMs on Both Domains


Best financial-LM rescoring result on Earnings22 from `results/task9_financial_lm_rescoring.csv`:

| Method | Alpha | Beta | WER | CER |
|---|---:|---:|---:|---:|
| Financial LM rescoring | 5.00 | 0.00 | 52.80% | 25.02% |

Final comparison table from `results/task9_lm_comparison.csv`:

| Method | LM | LibriSpeech WER | LibriSpeech CER | Earnings22 WER | Earnings22 CER |
|---|---|---:|---:|---:|---:|---:|
| Shallow fusion | LibriSpeech 3-gram | 10.98% | 3.74% | 55.57% | 25.47% |
| Rescoring | LibriSpeech 3-gram | 11.00% | 3.74% | 55.33% | 25.38% |
| Shallow fusion | Financial 3-gram | 12.47% | 4.01% | 51.23% | 25.84% |
| Rescoring | Financial 3-gram | 12.34% | 3.93% | 52.80% | 25.02% |

Answers:

- Which LM works best in-domain?
  - The LibriSpeech 3-gram LM works best in-domain. Both LibriSpeech-LM methods outperform both financial-LM methods on LibriSpeech by a clear margin.
- Which LM works best out-of-domain?
  - The financial-domain LM works best out-of-domain. It gives much better WER on Earnings22 than the LibriSpeech LM.
- Does domain-matched LM help more than a larger general LM?
  - Yes. Taken together with Task 5, the evidence suggests domain matching matters more than simply using a more complex general LM. The 4-gram general LM did not beat the 3-gram general LM in-domain, while the financial-domain 3-gram gave a large WER gain on Earnings22.

Additional observations:

- For LibriSpeech, the best result remains shallow fusion with the LibriSpeech 3-gram LM.
- For Earnings22, the best WER comes from shallow fusion with the financial LM (`51.23%`), while the best CER comes from rescoring with the financial LM (`25.02%`).
- The financial LM improves out-of-domain performance substantially, but it hurts in-domain performance, which is exactly what a strong domain mismatch hypothesis would predict.

## Final Summary

- Greedy and beam search both work reasonably on LibriSpeech, with beam width `10` being the best among the tested widths.
- Temperature has no effect on greedy decoding in the current setup.
- The best in-domain shallow-fusion setup is `alpha = 0.10`, `beta = 0.50`.
- The best in-domain rescoring setup is `alpha = 0.01`, `beta = 1.50`.
- Shallow fusion is slightly better than rescoring in-domain, but rescoring is far more stable when `alpha` is too large.
- The general LibriSpeech LM helps only a little in-domain and hurts out-of-domain.
- The financial-domain LM is clearly better on Earnings22, while the LibriSpeech LM remains best on LibriSpeech.
- Domain matching matters more than simply increasing general-LM capacity in these experiments.
