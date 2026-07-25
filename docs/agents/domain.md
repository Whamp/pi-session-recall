# Domain Docs

This repository uses a single-context domain documentation layout.

## Before exploring

Read these sources when they exist:

- `CONTEXT.md` at the repository root
- Relevant ADRs under `docs/adr/`

If they do not exist, proceed silently. The domain-modeling skill creates them when the project resolves terminology or architectural decisions.

## File structure

```text
/
├── CONTEXT.md
├── docs/adr/
└── src/
```

## Use canonical vocabulary

Use terms defined in `CONTEXT.md` consistently in code, filenames, tests, issues, and documentation. Avoid synonyms that the glossary rejects.

If a needed concept is absent, reconsider whether the project already names it differently. Record genuine vocabulary gaps for domain modeling.

## Flag ADR conflicts

If proposed work contradicts an ADR, identify the conflict rather than silently overriding the decision.
