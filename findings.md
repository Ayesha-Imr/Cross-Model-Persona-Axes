# Qualitative findings

Concrete response excerpts behind the headline numbers. Each axis below pairs the **highest-scoring** and **lowest-scoring** model on that trait with a representative response (the response whose score sits closest to that model's median on the axis — typical, not cherry-picked outliers).

Source CSVs: [`tables/08_trait_extremes.csv`](tables/08_trait_extremes.csv), [`tables/09_family_signatures.csv`](tables/09_family_signatures.csv), [`tables/10_axis_ladder.csv`](tables/10_axis_ladder.csv). The full per-record extremes (one row per model × axis × side) live in those files.

> Note: `deepseek-r1-distill-qwen-7b` and `gpt-oss-20b` emit visible reasoning-trace prefixes (BPE-encoded `Ġ` tokens or `analysis...assistantfinal` scaffolding). That trace text is itself a strong stylistic signal — the projection picks it up — but it makes raw excerpts harder to read. We summarize behavior rather than quote the noisy traces verbatim.

---

## warmth

**Highest: `gemini-3.0-flash`** (mean projection ≈ +40.3) — structured markdown answers with explicit framing ("Here is an analysis of…"), section headers, and a tone of careful tutoring. Even on contested political questions ("the obligation of leaders to seek bipartisanship is a central theme in democratic theory…") it stays patient and explainer-shaped rather than terse.

**Lowest: `deepseek-r1-distill-qwen-7b`** (mean ≈ +28.2) — every response opens with an unfiltered self-talk loop ("Okay, so I need to figure out…"). The thinking trace reads as cognitively cold and self-directed; warmth markers (you-framed acknowledgements, encouragement, softening) are absent.

## bluntness

**Highest: `deepseek-r1-distill-qwen-7b`** (mean ≈ −32.8 — note the axis is signed; this is the *least negative* in the roster). The thinking-trace style is the bluntness signal: short fragmentary sentences, no preamble, no hedges, just stepwise enumeration of the problem.

**Lowest: `gemini-3.0-flash`** (mean ≈ −46.9) — consistently buffers with framing paragraphs, "Here are the most common techniques…", numbered subsections, and a closing synthesis. The opposite of blunt by design.

## verbosity

**Highest: `gemini-3.0-flash`** (mean ≈ +50.9) — multi-section markdown with headers, bullets, and tables routinely. A single response averages ~3× the length of `deepseek-r1-distill-qwen-7b`.

**Lowest: `deepseek-r1-distill-qwen-7b`** (mean ≈ +35.4) — the thinking trace is dense but actually emits *less* assistant-facing text overall; the post-trace answers are short.

> Caveat: verbosity is partially confounded with raw output length. Length-controlled re-projection is in the follow-up list.

## sycophancy

**Highest: `deepseek-r1-distill-qwen-7b`** (mean ≈ −26.5). The signature isn't classic flattery — it's the model's habit of restating + agreeing with the user framing inside the thinking trace ("Okay, so the user wants…", "Yeah, that makes sense…") before answering. The Gemma prober reads that pattern as agreement-coded.

**Lowest: `gemini-3.0-flash`** (mean ≈ −40.5) — opens with neutral framing or counterpoint rather than agreement; on opinion questions it presents both sides before any synthesis.

## hedge_density

**Highest: `mistral-7b-instruct-v0.3`** (mean ≈ +8.66) — densely qualified explanations: "Factors like data quality, relevance, and representativeness *play crucial roles*…", "Poor-quality data *can lead to* incorrect conclusions…". Almost every claim carries a softener.

**Lowest: `gemini-3.0-flash`** (mean ≈ +6.26) — on the same prompt, leads with a strong principle ("the principle of *Garbage In, Garbage Out*") and commits directly to claims. Slightly hedgy phrasings appear but are clearly the minority style.

## refusal_stiffness

**Highest: `gpt-oss-20b`** (mean ≈ +12.7). A surprise — the surface response isn't a refusal, but the model emits a long pre-answer "analysis…" trace that the prober codes as defensive/hedging structure. On the parenting-recovery prompt, the trace enumerates "actionable steps, highlight coping strategies, communication, professional help, self-care…" before any direct content.

**Lowest: `deepseek-r1-distill-qwen-7b`** (mean ≈ +10.6) — almost no refusal vocabulary; the thinking-trace style attacks the question head-on with no caveat layer.

> Note that none of the deployed assistants in our roster *actually* refused our UltraChat prompts; "refusal-stiffness" here measures stylistic refusal-coded language (caveats, redirection, framing) on benign content.

## harmfulness

**Floor effect across the board.** Every model's mean projection rounds to ≈ 0.00 on this axis — extracted persona direction has essentially no separating power across modern RLHF'd models on benign prompts. We keep the axis in the config for transparency; it carries no information at this prompt-set scale.

---

## Family signatures (from `tables/09_family_signatures.csv`)

For each family we identify the axis with the largest mean |Cohen's d| against all other families — its **signature axis** — and the direction it leans:

| Family       | Signature axis    | Direction    | Mean \|d\| | Representative model           |
|--------------|-------------------|--------------|------------|---------------------------------|
| `cohere`     | verbosity         | above peers  | 2.11       | command-a                       |
| `cohere (OSS)` | verbosity       | above peers  | 1.78       | tiny-aya-global                 |
| `deepseek`   | bluntness         | above peers  | **9.43**   | deepseek-r1-distill-qwen-7b     |
| `gemini`     | verbosity         | above peers  | 2.92       | gemini-3.0-flash                |
| `gpt`        | bluntness         | below peers  | 1.79       | gpt-5.4                         |
| `gpt (OSS)`  | verbosity         | below peers  | 2.32       | gpt-oss-20b                     |
| `llama`      | verbosity         | above peers  | 1.33       | llama-3.1-8b-instruct           |
| `mistral`    | verbosity         | above peers  | 1.57       | mistral-7b-instruct-v0.3        |
| `qwen`       | verbosity         | above peers  | 2.12       | qwen-2.5-7b-instruct            |

`deepseek` is by far the most distinctive family — its bluntness effect size (9.43σ) is roughly 3× the next-largest family signature. Most other families' signature axis ends up being verbosity, which is consistent with the PCA finding that PC1 = warm-verbose vs blunt-direct.

---

## Caveats

- All projections are computed in **Gemma-4's** representational space. We're asking "how does Gemma read these models' text?", not characterizing the source models in their own activation geometry.
- Closed-API responses pass through provider-side moderation and tool-augmented inference, so e.g. refusal-stiffness conflates the deployed system with the underlying model.
- Reasoning-trace artifacts (the `Ġ` BPE tokens from DeepSeek-R1-Distill, the `analysisassistantfinal` scaffolding from gpt-oss-20b) inflate certain axes. They're a real behavioral fingerprint, but they limit the comparability of models with different output formats. Worth a follow-up: project only the "final answer" span instead of the full response.
