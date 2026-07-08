# Text Field Study — Methodology

This explains why `text_field_study.py` and `field_study_validation.py`'s
statistical approach is sound, stage by stage — useful if anyone (an
advisor, a reviewer, a collaborator) asks why this is mathematically
defensible.

Every individual technique below is a standard, textbook method —
nothing here is novel or improvised. The one place legitimate scrutiny
belongs is **construct validity** (see the end of this document), which
is a measurement question, not a statistics question.

---

## 1. Turning text into vectors (sentence embeddings)

`all-MiniLM-L6-v2` (the default model) is a sentence-transformer trained
via contrastive learning: it's explicitly optimized so that semantically
similar sentences end up close together in vector space and dissimilar
ones end up far apart. This is one of the most widely used
general-purpose sentence embedding models, standard in NLP and
computational social science. The property this pipeline relies on:
distance/similarity in this vector space is trained to track *meaning*,
not just surface word overlap.

## 2. Cosine similarity as the comparison metric

Cosine similarity measures the angle between two vectors, ignoring
their magnitude — `cos(θ) = (A·B)/(‖A‖‖B‖)`. Since embeddings are
L2-normalized before comparison (`embed_text_fields`), this reduces to a
plain dot product, bounded in [-1, 1]. This is the standard similarity
metric for embedding spaces precisely because it's insensitive to
vector length, which otherwise reflects text length/verbosity rather
than meaning — you don't want "wrote more" to trivially inflate
similarity.

## 3. The permutation test (`permutation_test_similarity`)

This is the core of why the significance test is defensible. The logic:

- You observe a mean similarity between a subject's answer to field A
  and their own answer to field B.
- The question: is that similarity higher than expected if the pairing
  between A and B answers were *random* — i.e. if there were no real
  relationship between what a subject wrote for A and what they wrote
  for B?
- To answer that without assuming any particular distribution (no
  normality assumption, no assumption about what "chance similarity"
  should look like), the null distribution is built *empirically*,
  directly from the data: shuffle who's paired with whom
  (`compute_paired_similarity`'s shuffle loop), recompute the mean
  similarity under that wrong pairing, repeat many times (`n_null`).
  This produces an actual empirical distribution of "what mean
  similarity looks like when there's no real subject-level
  relationship."
- The p-value is the fraction of those shuffled ("null") means at least
  as extreme as the real one:
  `p = (1 + count(null_shuffle_means >= observed_mean)) / (1 + n_permutations)`.
  The +1/+1 (add-one smoothing) exists because the actual observed
  arrangement is itself one valid permutation among all possible ones —
  so the true minimum achievable p-value with `n` permutations is
  `1/(n+1)`, never exactly 0.

This is standard, well-established permutation testing (Fisher's
original framework for exact tests, 1930s), not an ad hoc invention.
Its main strength: it makes **no parametric assumptions** about the
similarity scores' distribution, which matters here because cosine
similarities aren't naturally normally distributed.

## 4. Cohen's d — standardized effect size (`cohens_d`)

A p-value alone conflates effect size with sample size — a tiny effect
can be "significant" with enough data. Cohen's d normalizes the
difference between observed same-subject similarities and the null by
the pooled standard deviation, giving a scale-free measure of *how big*
the gap is, independent of `n`. This is the standard effect-size metric
across psychology and the social sciences, specifically because it lets
effect magnitude be compared across different measures and studies
rather than only within one.

## 5. Benjamini-Hochberg FDR correction (`benjamini_hochberg`)

Testing multiple field pairs inflates the false-positive rate — running
20 tests at α=0.05 yields ~1 expected false positive by chance alone
even with zero real effects anywhere. Benjamini-Hochberg is the
standard correction for this that doesn't overcorrect the way a
stricter method (Bonferroni) does — it controls the *expected
proportion of false discoveries* among results called significant,
rather than the probability of any false positive at all. It's the
field-standard approach because it retains reasonable statistical power
while still controlling for the multiple-testing problem — this is why
fields that routinely run many comparisons (genomics, neuroscience,
psychology) report FDR-corrected q-values rather than raw p-values.

## 6. Word-count-controlled regression (`wordcount_controlled_regression`)

An OLS regression of similarity on word count directly tests an
alternative explanation: maybe similarity isn't tracking meaning at
all, it's tracking "subjects who write more sound more similar to
themselves" — a length artifact of the embedding, not a substantive
finding. Including word count as a covariate and checking whether it's
a significant predictor is standard confound-control practice — a
non-significant coefficient directly rules out the most obvious mundane
explanation for the result.

## 7. Bootstrap confidence interval (`bootstrap_mean_ci`)

Resampling subjects with replacement many times and recomputing the
mean each time is a nonparametric way to estimate how much the point
estimate would vary under a slightly different sample from the same
population. It requires no assumption about the underlying distribution
of similarity scores (unlike a classic ±1.96×SE interval, which assumes
approximate normality) — which matters here for the same reason the
permutation test does. This is a standard, well-established resampling
technique (Efron, 1979), used precisely when a parametric distributional
form can't be assumed.

## 8. Leave-one-out sensitivity (`leave_one_out_sensitivity`)

Not a formal significance test — a robustness/diagnostic check, standard
in applied statistics whenever sample sizes are modest: does the
conclusion depend heavily on one unusual subject? If removing any single
subject doesn't meaningfully change the result, the finding is more
trustworthy; if one subject is doing all the work, that's important to
know and disclose, not something a p-value alone would reveal.

---

## Where scrutiny actually belongs: construct validity

Every technique above is textbook-standard. The legitimate question for
a skeptical reader isn't "is the statistics sound" — it's: **does
cosine similarity between sentence embeddings measure the construct
you actually care about, or does it measure topical/lexical overlap
that correlates with but isn't identical to that construct?**

That's a measurement question, not a statistics question, and this
pipeline doesn't resolve it on its own. Two honest ways to strengthen a
claim beyond what text similarity alone can support:

- Add a validated criterion measure (e.g. an established self-report
  instrument relevant to your question) to correlate the similarity
  metric against — this gives an actual ground-truth-adjacent anchor.
- Frame results without a validated criterion as exploratory /
  hypothesis-generating rather than confirmatory, and be explicit in
  any write-up about the gap between "semantically similar text" and
  the deeper construct being inferred from it.
