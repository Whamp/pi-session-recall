# Octen HTTP embedding contract

Conversation Recall uses one direct OpenAI-compatible Octen endpoint. `RecallEmbeddingProvider` keeps query and document calls distinct, but both send source text unchanged and apply the same stored-vector transformation.

For every response vector, the provider:

1. requires exactly the configured native width;
2. rejects non-finite values;
3. retains the first configured stored dimensions;
4. L2-normalizes the prefix;
5. rounds stored components to FP32.

The default profile receives 2,560 dimensions and stores 1,024. Zvec indexes normalized vectors with inner product. Search converts inner-product similarity back to the existing bounded cosine-distance field used by ranking and evidence output.

The manifest binds request model, served model, native width, stored width, transformation, tokenizer, and chunk policy. The endpoint URL and batch size affect execution but not semantic compatibility. Any bound identity change requires `psr index --rebuild`.

Tests cover HTTP request ordering and dimensions, query/document transformation, finite-value rejection, timeout and cancellation, normalized inner-product ranking, and manifest incompatibility.
