---
status: accepted
---

# Store normalized Octen prefixes with inner product

Conversation Recall uses one configured Octen HTTP model. It retains the first configured dimensions of each native vector, L2-normalizes that prefix, stores it once as FP32, and searches zvec with inner product. The default profile stores 1,024 of Octen's 2,560 native dimensions.

Octen supports first-N output truncation, but this decision does not claim independently verified MRL quality at every width. The index manifest binds the model, native width, stored width, transformation, tokenizer, and chunk policy; changing any of them requires `psr index --rebuild`.

Inner product preserves cosine ordering for normalized document and query vectors and avoids zvec's second cosine normalization. We rejected embedded model management, separate model profiles, persistent vector caches, and post-write vector repair because the manually maintained product needs one direct provider and one durable collection.
