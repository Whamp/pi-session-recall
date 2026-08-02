---
status: superseded by ADR-0003
---

# Reconcile recall from Pi's session lifecycle

Conversation Recall previously updated changed session files from Pi lifecycle and search paths. Measurements showed that whole-session parsing, tokenization, and zvec writes blocked interactive work, so ADR-0003 moved all maintenance to the explicit `psr` CLI.
