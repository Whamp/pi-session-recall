# Octen prefix truncation prototype

**Question:** Do the original leading coordinates of the current Octen embeddings preserve enough similarity and retrieval behavior to justify storing fewer dimensions?

This is throwaway measurement code. It read regular embedding-cache and Arrow scalar files only. It did not open zvec, start a model, call Octen's API, or write under the recall directory.

## Input

- Model: `Octen/Octen-Embedding-4B`
- Artifact: `Octen-Embedding-4B.Q8_0.gguf`
- Native dimensions: 2,560
- Cache identity: `ed223a8b51fb2e781ba582bf2d2026a1f1b688a22179fe06db801ce5e2ea7704`
- Candidate vectors: 20,000
- Query vectors: 100
- Available current dense evidence rows: 282,552
- Available unique current evidence vectors: 208,068

Queries are current active-branch user-message chunks. Their active child assistant messages are a rough expected-answer target. This is a comparison proxy, not a labeled recall-quality benchmark.

## Coordinate distribution

| Coordinates        | Width | Mean vector energy | Coordinate variance |
| ------------------ | ----: | -----------------: | ------------------: |
| leading 1536       | 1,536 |              62.2% |               60.9% |
| leading 1024       | 1,024 |              41.9% |               41.1% |
| leading 768        |   768 |              31.4% |               30.4% |
| leading 512        |   512 |              20.4% |               19.1% |
| seeded random 1024 | 1,024 |              40.0% |               39.9% |
| trailing 1024      | 1,024 |              37.8% |               39.1% |

## Result

The leading 1,024 dimensions retained 85.0% of the full-width top-10 neighbors. Rough answer hit@10 changed from 50.0% to 48.0%, and answer MRR changed from 0.3017 to 0.3011. This supports 1,024 as a practical stored-width candidate for a fuller quality run.

The random and trailing 1,024-coordinate controls retained 85.4% and 85.7% of the same neighbors. Because those controls were similarly strong, this run does not independently prove that Octen specially front-loads semantic information.

Energy and variance describe where values sit in the original coordinates. They do not by themselves measure semantic quality.

## Retrieval comparison

| Variant            | Width | Score correlation | Mean score change | Top-1 same | Top-10 overlap | Answer hit@10 | Answer hit@50 | Answer MRR | Median answer rank |
| ------------------ | ----: | ----------------: | ----------------: | ---------: | -------------: | ------------: | ------------: | ---------: | -----------------: |
| full baseline      | 2,560 |            1.0000 |            0.0000 |     100.0% |         100.0% |         50.0% |         62.0% |     0.3017 |               10.5 |
| leading 1536       | 1,536 |            0.9876 |            0.0157 |      96.0% |          88.4% |         47.0% |         64.0% |     0.2992 |               11.0 |
| leading 1024       | 1,024 |            0.9716 |            0.0192 |      86.0% |          85.0% |         48.0% |         62.0% |     0.3011 |               12.0 |
| leading 768        |   768 |            0.9581 |            0.0271 |      81.0% |          81.5% |         45.0% |         60.0% |     0.2912 |               17.5 |
| leading 512        |   512 |            0.9241 |            0.0452 |      75.0% |          75.5% |         46.0% |         59.0% |     0.3027 |               13.0 |
| seeded random 1024 | 1,024 |            0.9680 |            0.0174 |      85.0% |          85.4% |         46.0% |         63.0% |     0.3150 |               14.0 |
| trailing 1024      | 1,024 |            0.9595 |            0.0262 |      88.0% |          85.7% |         48.0% |         63.0% |     0.3083 |               12.0 |

The full-width row is the baseline, so its score correlation, neighborhood overlap, and top-1 preservation are 1. Compare `leading 1024` with the random and trailing 1024-coordinate controls. A useful leading prefix should stay close to full-width retrieval and outperform those controls.

## Sample query previews

- yes but it's impossible to loop through EVERY possible parameter setting so it should be somewhat intelligent in it's p…
- i'm asking if they need to be updated with the new location of dotfiles
- what's the next step
- remind me what pnpm run check runs in the project? i generally like to run all the tests in that project no matter what…
- yeah but we shouldn't go crazy with it
- Task: Audit /home/will/projects/observatory/docs/research/2026-07-09-tailscale-serve-live-services.md for factual suppo…
- i think option b. A topic that needs further thought here, though, is the portability of this analysis. Specifically, i…
- ## Context We implemented OTP verification improvements and e2e tests for the magic link flow in Classroom Connect V2. …
- <skill name="implement" location="/home/will/.agents/skills/implement/SKILL.md"> References are relative to /home/will/…
- Scheduled TaskMonitor check request 20260625T183158Z-3585635. Run exactly one visible scheduled-check cycle, then retur…

## Limits

- This measures the current local Q8 GGUF cache, which is the artifact that matters for this installation.
- The candidate set is a deterministic sample, not the entire recall corpus.
- Child assistant messages are useful comparison targets but are not human relevance labels.
- A final dimension choice should also be run through the existing labeled recall quality corpus when the production profile supports reduced dimensions.
- PCA is intentionally omitted because PCA rotates the coordinates and cannot test whether Octen's original first N values are useful.
