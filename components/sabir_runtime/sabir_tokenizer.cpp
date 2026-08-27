#include "sabir_tokenizer.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace sabir {
namespace {

#pragma pack(push, 1)
struct TokenizerHeaderV1 {
  char magic[8];
  uint32_t version, vocab, unk_id, bos_id, eos_id, pad_id, flags;
  uint32_t entries_offset, strings_offset, strings_bytes;
};

struct TokenizerEntryV1 {
  uint32_t offset;
  uint16_t length;
  uint8_t kind, reserved;
  float score;
};
#pragma pack(pop)

static_assert(sizeof(TokenizerHeaderV1) == 48, "tokenizer header mismatch");
static_assert(sizeof(TokenizerEntryV1) == 12, "tokenizer entry mismatch");

constexpr char kMagic[8] = {'S', 'B', 'T', 'O', 'K', '1', '\0', '\0'};
constexpr uint32_t kAddDummyPrefix = 1;
constexpr uint32_t kRemoveExtraWhitespace = 2;
constexpr uint32_t kEscapeWhitespace = 4;
constexpr uint8_t kUnknownPiece = 2;
constexpr uint8_t kControlPiece = 3;
constexpr uint8_t kUserDefinedPiece = 4;
constexpr uint8_t kSpaceMarker[3] = {0xe2, 0x96, 0x81};

struct Symbol {
  uint16_t begin;
  uint16_t length;
  int16_t known_id;
};

uint32_t hash_bytes(const uint8_t *data, size_t length) {
  uint32_t hash = 2166136261u;
  for (size_t index = 0; index < length; ++index) {
    hash ^= data[index];
    hash *= 16777619u;
  }
  return hash;
}

const TokenizerEntryV1 &entry(const Tokenizer &tokenizer, int id) {
  return reinterpret_cast<const TokenizerEntryV1 *>(tokenizer.entries)[id];
}

const uint8_t *piece_data(const Tokenizer &tokenizer, int id) {
  return tokenizer.strings + entry(tokenizer, id).offset;
}

int find_piece(const Tokenizer &tokenizer, const uint8_t *data, int length) {
  if (length <= 0) return -1;
  uint32_t slot = hash_bytes(data, length) & (tokenizer.hash_capacity - 1);
  for (uint32_t probes = 0; probes < tokenizer.hash_capacity; ++probes) {
    const int id = tokenizer.hash_slots[slot];
    if (id < 0) return -1;
    const TokenizerEntryV1 &candidate = entry(tokenizer, id);
    if (candidate.length == length &&
        std::memcmp(piece_data(tokenizer, id), data, length) == 0) return id;
    slot = (slot + 1) & (tokenizer.hash_capacity - 1);
  }
  return -1;
}

bool ascii_space(uint8_t value) {
  return value == ' ' || value == '\t' || value == '\r' || value == '\n' ||
         value == '\v' || value == '\f';
}

int utf8_character_bytes(uint8_t first) {
  if ((first & 0x80u) == 0) return 1;
  if ((first & 0xe0u) == 0xc0u) return 2;
  if ((first & 0xf0u) == 0xe0u) return 3;
  if ((first & 0xf8u) == 0xf0u) return 4;
  return 1;
}

int append_marker(char *output, int at, int capacity) {
  if (at + 3 >= capacity) return -1;
  std::memcpy(output + at, kSpaceMarker, 3);
  return at + 3;
}

int normalize_portable(const Tokenizer &tokenizer, const char *input,
                       char *output, int capacity) {
  if (!input || !output || capacity < 4) return -1;
  const uint8_t *source = reinterpret_cast<const uint8_t *>(input);
  size_t length = std::strlen(input);
  size_t begin = 0;
  if (tokenizer.flags & kRemoveExtraWhitespace) {
    while (begin < length && ascii_space(source[begin])) ++begin;
    while (length > begin && ascii_space(source[length - 1])) --length;
  }
  int at = 0;
  if (begin < length && (tokenizer.flags & kAddDummyPrefix)) {
    at = (tokenizer.flags & kEscapeWhitespace) ? append_marker(output, at, capacity) : at;
    if (at < 0) return -1;
  }
  bool pending_space = false;
  for (size_t index = begin; index < length; ++index) {
    const uint8_t byte = source[index];
    if ((tokenizer.flags & kRemoveExtraWhitespace) && ascii_space(byte)) {
      pending_space = true;
      continue;
    }
    if (pending_space) {
      if (tokenizer.flags & kEscapeWhitespace) {
        at = append_marker(output, at, capacity);
        if (at < 0) return -1;
      } else if (at + 1 >= capacity) {
        return -1;
      } else {
        output[at++] = ' ';
      }
      pending_space = false;
    }
    if (at + 1 >= capacity) return -1;
    output[at++] = static_cast<char>(byte);
  }
  output[at] = '\0';
  return at;
}

}  // namespace

size_t tokenizer_hash_bytes(uint32_t vocab) {
  uint32_t capacity = 1;
  while (capacity < vocab * 2) capacity <<= 1;
  return static_cast<size_t>(capacity) * sizeof(int16_t);
}

bool load_tokenizer(const uint8_t *asset, size_t bytes, void *hash_memory,
                    size_t hash_memory_bytes, Tokenizer *tokenizer, const char **error) {
  auto fail = [error](const char *message) {
    if (error) *error = message;
    return false;
  };
  if (!asset || !tokenizer || bytes < sizeof(TokenizerHeaderV1)) return fail("short tokenizer asset");
  TokenizerHeaderV1 header;
  std::memcpy(&header, asset, sizeof(header));
  if (std::memcmp(header.magic, kMagic, sizeof(kMagic)) != 0 || header.version != 1)
    return fail("unsupported tokenizer asset");
  const uint64_t entries_end = static_cast<uint64_t>(header.entries_offset) +
                               static_cast<uint64_t>(header.vocab) * sizeof(TokenizerEntryV1);
  const uint64_t strings_end = static_cast<uint64_t>(header.strings_offset) + header.strings_bytes;
  if (entries_end > bytes || strings_end > bytes || header.entries_offset < sizeof(header))
    return fail("tokenizer offsets exceed asset");
  const size_t needed = tokenizer_hash_bytes(header.vocab);
  if (!hash_memory || hash_memory_bytes < needed) return fail("tokenizer hash memory too small");
  tokenizer->asset = asset;
  tokenizer->asset_bytes = bytes;
  tokenizer->vocab = header.vocab;
  tokenizer->unk_id = static_cast<int32_t>(header.unk_id);
  tokenizer->bos_id = static_cast<int32_t>(header.bos_id);
  tokenizer->eos_id = static_cast<int32_t>(header.eos_id);
  tokenizer->pad_id = static_cast<int32_t>(header.pad_id);
  tokenizer->flags = header.flags;
  tokenizer->entries = asset + header.entries_offset;
  tokenizer->strings = asset + header.strings_offset;
  tokenizer->strings_bytes = header.strings_bytes;
  tokenizer->hash_slots = static_cast<int16_t *>(hash_memory);
  tokenizer->hash_capacity = static_cast<uint32_t>(needed / sizeof(int16_t));
  std::fill(tokenizer->hash_slots, tokenizer->hash_slots + tokenizer->hash_capacity,
            static_cast<int16_t>(-1));
  for (uint32_t id = 0; id < tokenizer->vocab; ++id) {
    const TokenizerEntryV1 &item = entry(*tokenizer, id);
    if (static_cast<uint64_t>(item.offset) + item.length > tokenizer->strings_bytes)
      return fail("tokenizer piece exceeds string table");
    uint32_t slot = hash_bytes(piece_data(*tokenizer, id), item.length) &
                    (tokenizer->hash_capacity - 1);
    while (tokenizer->hash_slots[slot] >= 0)
      slot = (slot + 1) & (tokenizer->hash_capacity - 1);
    tokenizer->hash_slots[slot] = static_cast<int16_t>(id);
  }
  if (error) *error = nullptr;
  return true;
}

int encode(const Tokenizer &tokenizer, const char *utf8, uint16_t *ids,
           int ids_capacity, char *normalized, int normalized_capacity) {
  if (!ids || ids_capacity <= 0) return -1;
  const int normalized_bytes = normalize_portable(tokenizer, utf8, normalized, normalized_capacity);
  if (normalized_bytes < 0 || normalized_bytes > 0xffff) return -1;
  Symbol symbols[kTokenizerMaxSymbols];
  int symbol_count = 0;
  int offset = 0;
  while (offset < normalized_bytes) {
    if (symbol_count >= kTokenizerMaxSymbols) return -1;
    int selected = -1;
    int selected_bytes = 0;
    for (uint32_t id = 0; id < tokenizer.vocab; ++id) {
      const TokenizerEntryV1 &item = entry(tokenizer, id);
      if (item.kind != kUserDefinedPiece || item.length <= selected_bytes ||
          offset + item.length > normalized_bytes) continue;
      if (std::memcmp(normalized + offset, piece_data(tokenizer, id), item.length) == 0) {
        selected = static_cast<int>(id);
        selected_bytes = item.length;
      }
    }
    if (selected < 0) {
      selected_bytes = std::min(utf8_character_bytes(static_cast<uint8_t>(normalized[offset])),
                                normalized_bytes - offset);
      selected = find_piece(tokenizer,
                            reinterpret_cast<const uint8_t *>(normalized + offset), selected_bytes);
    }
    symbols[symbol_count++] = Symbol{static_cast<uint16_t>(offset),
                                     static_cast<uint16_t>(selected_bytes),
                                     static_cast<int16_t>(selected)};
    offset += selected_bytes;
  }

  while (symbol_count > 1) {
    int best_at = -1;
    int best_id = -1;
    float best_score = -INFINITY;
    for (int index = 0; index + 1 < symbol_count; ++index) {
      const int combined = symbols[index].length + symbols[index + 1].length;
      const int candidate = find_piece(
          tokenizer, reinterpret_cast<const uint8_t *>(normalized + symbols[index].begin), combined);
      if (candidate < 0) continue;
      const float score = entry(tokenizer, candidate).score;
      if (score > best_score) {
        best_at = index;
        best_id = candidate;
        best_score = score;
      }
    }
    if (best_at < 0) break;
    symbols[best_at].length = static_cast<uint16_t>(symbols[best_at].length +
                                                    symbols[best_at + 1].length);
    symbols[best_at].known_id = static_cast<int16_t>(best_id);
    for (int index = best_at + 1; index + 1 < symbol_count; ++index)
      symbols[index] = symbols[index + 1];
    --symbol_count;
  }
  if (symbol_count > ids_capacity) return -1;
  for (int index = 0; index < symbol_count; ++index) {
    ids[index] = static_cast<uint16_t>(symbols[index].known_id >= 0 ?
                                      symbols[index].known_id : tokenizer.unk_id);
  }
  return symbol_count;
}

int decode_piece(const Tokenizer &tokenizer, int id, char *output,
                 int output_capacity) {
  if (!output || output_capacity <= 0 || id < 0 || static_cast<uint32_t>(id) >= tokenizer.vocab)
    return -1;
  const TokenizerEntryV1 &item = entry(tokenizer, id);
  if (item.kind == kControlPiece) {
    output[0] = '\0';
    return 0;
  }
  if (item.kind == kUnknownPiece) {
    constexpr uint8_t replacement[] = {0xef, 0xbf, 0xbd};
    if (output_capacity < 4) return -1;
    std::memcpy(output, replacement, 3);
    output[3] = '\0';
    return 3;
  }
  const uint8_t *source = piece_data(tokenizer, id);
  int written = 0;
  for (uint16_t offset = 0; offset < item.length;) {
    if (offset + 3 <= item.length && std::memcmp(source + offset, kSpaceMarker, 3) == 0) {
      if (written + 1 >= output_capacity) return -1;
      output[written++] = ' ';
      offset += 3;
    } else {
      if (written + 1 >= output_capacity) return -1;
      output[written++] = static_cast<char>(source[offset++]);
    }
  }
  output[written] = '\0';
  return written;
}

}  // namespace sabir

