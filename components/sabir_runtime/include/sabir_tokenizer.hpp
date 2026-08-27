#pragma once

#include <cstddef>
#include <cstdint>

namespace sabir {

constexpr int kTokenizerMaxSymbols = 768;

struct Tokenizer {
  const uint8_t *asset = nullptr;
  size_t asset_bytes = 0;
  uint32_t vocab = 0;
  int32_t unk_id = -1;
  int32_t bos_id = -1;
  int32_t eos_id = -1;
  int32_t pad_id = -1;
  uint32_t flags = 0;
  const uint8_t *entries = nullptr;
  const uint8_t *strings = nullptr;
  uint32_t strings_bytes = 0;
  int16_t *hash_slots = nullptr;
  uint32_t hash_capacity = 0;
};

// hash_memory must hold at least tokenizer_hash_bytes(vocab) bytes and remain valid.
size_t tokenizer_hash_bytes(uint32_t vocab);
bool load_tokenizer(const uint8_t *asset, size_t bytes, void *hash_memory,
                    size_t hash_bytes, Tokenizer *tokenizer, const char **error);

// Implements the model's BPE exactly for ordinary NFKC-normal Turkish UTF-8.
// It collapses whitespace and handles user-defined symbols. Full nmt_nfkc's
// 240 KB normalization charsmap is intentionally not duplicated on device.
int encode(const Tokenizer &tokenizer, const char *utf8, uint16_t *ids,
           int ids_capacity, char *normalized, int normalized_capacity);

// Writes a single token piece to output. SentencePiece whitespace markers are
// changed back to ASCII spaces. Returns bytes written, excluding the NUL.
int decode_piece(const Tokenizer &tokenizer, int id, char *output,
                 int output_capacity);

}  // namespace sabir

