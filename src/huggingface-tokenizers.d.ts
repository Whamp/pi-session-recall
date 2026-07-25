declare module '@huggingface/tokenizers' {
  interface TokenizerEncoding {
    ids: number[];
    tokens: string[];
    'attention_mask': number[];
    'token_type_ids'?: number[];
  }

  interface TokenizerEncodeOptions {
    'text_pair'?: string | null;
    'add_special_tokens'?: boolean;
    'return_token_type_ids'?: boolean | null;
  }

  /** Encodes text with a tokenizer definition and configuration loaded from JSON assets. */
  export class Tokenizer {
    constructor(tokenizer: object, config: object);
    encode(text: string, options?: TokenizerEncodeOptions): TokenizerEncoding;
  }
}
