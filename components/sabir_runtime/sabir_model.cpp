#include "sabir_model.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>

namespace sabir {
namespace {

#pragma pack(push, 1)
struct ImageHeaderV1 {
  char magic[8];
  uint32_t version, header_bytes, flags;
  uint32_t n_layer, n_embd, n_head, block_size, vocab_size, group_size;
  uint32_t tensor_count, directory_entry_bytes, payload_crc32;
  uint64_t directory_offset, data_offset, image_size;
};

struct DirectoryEntryV1 {
  char name[28];
  uint8_t dtype, ndim;
  uint16_t reserved0;
  uint32_t dims[4];
  uint64_t data_offset, data_bytes, aux_offset;
  uint32_t aux_bytes, reserved1;
};
#pragma pack(pop)

static_assert(sizeof(ImageHeaderV1) == 80, "format header mismatch");
static_assert(sizeof(DirectoryEntryV1) == 80, "format directory mismatch");

constexpr char kMagic[8] = {'S', 'A', 'B', 'I', 'R', '4', '\0', '\0'};
constexpr float kLayerNormEpsilon = 1.0e-5f;

struct Directory {
  const uint8_t *image;
  size_t bytes;
  const DirectoryEntryV1 *entries;
  uint32_t count;
  int group;
};

bool range_ok(uint64_t offset, uint64_t length, size_t available) {
  return offset <= available && length <= available - static_cast<size_t>(offset);
}

const DirectoryEntryV1 *find(const Directory &directory, const char *name) {
  for (uint32_t index = 0; index < directory.count; ++index) {
    const DirectoryEntryV1 &entry = directory.entries[index];
    if (std::strncmp(entry.name, name, sizeof(entry.name)) == 0) return &entry;
  }
  return nullptr;
}

bool bind_q4(const Directory &directory, const char *name, int rows, int cols,
             Q4Tensor *tensor) {
  const DirectoryEntryV1 *entry = find(directory, name);
  if (!entry || entry->dtype != kDtypeQ4 || entry->ndim != 2 ||
      static_cast<int>(entry->dims[0]) != rows || static_cast<int>(entry->dims[1]) != cols) return false;
  const int row_bytes = (cols + 1) / 2;
  const int groups = (cols + directory.group - 1) / directory.group;
  const uint64_t expected_codes = static_cast<uint64_t>(rows) * row_bytes;
  const uint64_t expected_scales = static_cast<uint64_t>(rows) * groups * sizeof(uint16_t);
  if (entry->data_bytes != expected_codes || entry->aux_bytes != expected_scales ||
      !range_ok(entry->data_offset, entry->data_bytes, directory.bytes) ||
      !range_ok(entry->aux_offset, entry->aux_bytes, directory.bytes)) return false;
  tensor->codes = directory.image + entry->data_offset;
  tensor->scales = reinterpret_cast<const uint16_t *>(directory.image + entry->aux_offset);
  tensor->rows = rows;
  tensor->cols = cols;
  tensor->group = directory.group;
  tensor->groups_per_row = groups;
  tensor->bytes_per_row = row_bytes;
  return true;
}

bool bind_f32(const Directory &directory, const char *name, int elements,
              FloatTensor *tensor) {
  const DirectoryEntryV1 *entry = find(directory, name);
  if (!entry || entry->dtype != kDtypeF32 || entry->data_bytes !=
      static_cast<uint64_t>(elements) * sizeof(float) ||
      !range_ok(entry->data_offset, entry->data_bytes, directory.bytes)) return false;
  tensor->data = reinterpret_cast<const float *>(directory.image + entry->data_offset);
  tensor->elements = elements;
  return true;
}

float half_to_float(uint16_t value) {
  const uint32_t sign = static_cast<uint32_t>(value & 0x8000u) << 16;
  uint32_t exponent = (value >> 10) & 0x1fu;
  uint32_t mantissa = value & 0x3ffu;
  uint32_t bits;
  if (exponent == 0) {
    if (mantissa == 0) {
      bits = sign;
    } else {
      exponent = 113;
      while ((mantissa & 0x400u) == 0) {
        mantissa <<= 1;
        --exponent;
      }
      bits = sign | (exponent << 23) | ((mantissa & 0x3ffu) << 13);
    }
  } else if (exponent == 31) {
    bits = sign | 0x7f800000u | (mantissa << 13);
  } else {
    bits = sign | ((exponent + 112) << 23) | (mantissa << 13);
  }
  float result;
  std::memcpy(&result, &bits, sizeof(result));
  return result;
}

void dequantize_row(const Q4Tensor &tensor, int row, float *output) {
  const uint8_t *codes = tensor.codes + static_cast<size_t>(row) * tensor.bytes_per_row;
  const uint16_t *scales = tensor.scales + static_cast<size_t>(row) * tensor.groups_per_row;
  for (int col = 0; col < tensor.cols; ++col) {
    const uint8_t packed = codes[col >> 1];
    const int code = (col & 1) ? packed >> 4 : packed & 0x0f;
    output[col] = static_cast<float>(code - 8) * half_to_float(scales[col / tensor.group]);
  }
}

void matvec(const Q4Tensor &tensor, const float *input, float *output) {
  for (int row = 0; row < tensor.rows; ++row) {
    const uint8_t *codes = tensor.codes + static_cast<size_t>(row) * tensor.bytes_per_row;
    const uint16_t *scales = tensor.scales + static_cast<size_t>(row) * tensor.groups_per_row;
    float sum = 0.0f;
    for (int group = 0; group < tensor.groups_per_row; ++group) {
      const int begin = group * tensor.group;
      const int end = std::min(begin + tensor.group, tensor.cols);
      float group_sum = 0.0f;
      for (int col = begin; col < end; ++col) {
        const uint8_t packed = codes[col >> 1];
        const int code = (col & 1) ? packed >> 4 : packed & 0x0f;
        group_sum += static_cast<float>(code - 8) * input[col];
      }
      sum += group_sum * half_to_float(scales[group]);
    }
    output[row] = sum;
  }
}

void add_bias(float *values, const FloatTensor &bias, int count) {
  for (int index = 0; index < count; ++index) values[index] += bias.data[index];
}

void layer_norm(const float *input, const FloatTensor &weight,
                const FloatTensor &bias, int count, float *output) {
  float mean = 0.0f;
  for (int index = 0; index < count; ++index) mean += input[index];
  mean /= static_cast<float>(count);
  float variance = 0.0f;
  for (int index = 0; index < count; ++index) {
    const float centered = input[index] - mean;
    variance += centered * centered;
  }
  const float inverse = 1.0f / std::sqrt(variance / static_cast<float>(count) + kLayerNormEpsilon);
  for (int index = 0; index < count; ++index) {
    output[index] = (input[index] - mean) * inverse * weight.data[index] + bias.data[index];
  }
}

uint32_t crc32_payload(const uint8_t *data, size_t length) {
  uint32_t crc = 0xffffffffu;
  for (size_t index = 0; index < length; ++index) {
    crc ^= data[index];
    for (int bit = 0; bit < 8; ++bit) crc = (crc >> 1) ^ (0xedb88320u & (0u - (crc & 1u)));
  }
  return crc ^ 0xffffffffu;
}

}  // namespace

bool load_model(const uint8_t *image, size_t available_bytes, Model *model,
                const char **error) {
  auto fail = [error](const char *message) {
    if (error) *error = message;
    return false;
  };
  if (!image || !model || available_bytes < sizeof(ImageHeaderV1)) return fail("short model header");
  ImageHeaderV1 header;
  std::memcpy(&header, image, sizeof(header));
  if (std::memcmp(header.magic, kMagic, sizeof(kMagic)) != 0) return fail("bad model magic");
  if (header.version != kFormatVersion || header.header_bytes != sizeof(ImageHeaderV1) ||
      header.directory_entry_bytes != sizeof(DirectoryEntryV1)) return fail("unsupported model format");
  if (header.n_layer != 8 || header.n_embd != 384 || header.n_head != 6 ||
      header.block_size != 256 || header.vocab_size != 8000 || header.group_size == 0)
    return fail("unexpected Sabir architecture");
  if (header.image_size > available_bytes ||
      !range_ok(header.directory_offset,
                static_cast<uint64_t>(header.tensor_count) * sizeof(DirectoryEntryV1),
                header.data_offset)) return fail("invalid model directory");

  Directory directory{image, static_cast<size_t>(header.image_size),
                      reinterpret_cast<const DirectoryEntryV1 *>(image + header.directory_offset),
                      header.tensor_count, static_cast<int>(header.group_size)};
  model->layers = header.n_layer;
  model->embedding = header.n_embd;
  model->heads = header.n_head;
  model->trained_context = header.block_size;
  model->vocab = header.vocab_size;
  model->group = header.group_size;
  model->image_bytes = header.image_size;
  model->payload_crc32 = header.payload_crc32;
  if (!bind_q4(directory, "tok_emb.weight", model->vocab, model->embedding, &model->token_embedding) ||
      !bind_q4(directory, "pos_emb.weight", model->trained_context, model->embedding, &model->position_embedding))
    return fail("missing embeddings");
  const int hidden = 4 * model->embedding;
  char name[28];
  for (int layer = 0; layer < model->layers; ++layer) {
    Layer &block = model->block[layer];
#define BIND_F(field, suffix, count) \
    std::snprintf(name, sizeof(name), "b%d.%s", layer, suffix); \
    if (!bind_f32(directory, name, count, &block.field)) return fail("missing layer float tensor")
#define BIND_Q(field, suffix, rows, cols) \
    std::snprintf(name, sizeof(name), "b%d.%s", layer, suffix); \
    if (!bind_q4(directory, name, rows, cols, &block.field)) return fail("missing layer q4 tensor")
    BIND_F(ln1_weight, "ln1.weight", model->embedding);
    BIND_F(ln1_bias, "ln1.bias", model->embedding);
    BIND_F(ln2_weight, "ln2.weight", model->embedding);
    BIND_F(ln2_bias, "ln2.bias", model->embedding);
    BIND_Q(qkv, "qkv.weight", 3 * model->embedding, model->embedding);
    BIND_Q(attention, "attn.weight", model->embedding, model->embedding);
    BIND_F(attention_bias, "attn.bias", model->embedding);
    BIND_Q(ff1, "ff1.weight", hidden, model->embedding);
    BIND_F(ff1_bias, "ff1.bias", hidden);
    BIND_Q(ff2, "ff2.weight", model->embedding, hidden);
    BIND_F(ff2_bias, "ff2.bias", model->embedding);
#undef BIND_F
#undef BIND_Q
  }
  if (!bind_f32(directory, "ln_f.weight", model->embedding, &model->final_norm_weight) ||
      !bind_f32(directory, "ln_f.bias", model->embedding, &model->final_norm_bias) ||
      !bind_q4(directory, "lm_head.weight", model->vocab, model->embedding, &model->output_head) ||
      !bind_f32(directory, "lm_head.bias", model->vocab, &model->output_bias))
    return fail("missing output tensors");
  const DirectoryEntryV1 *tokenizer = find(directory, "tokenizer");
  if (!tokenizer || tokenizer->dtype != kDtypeBlob ||
      !range_ok(tokenizer->data_offset, tokenizer->data_bytes, directory.bytes))
    return fail("missing tokenizer asset");
  model->tokenizer_blob = image + tokenizer->data_offset;
  model->tokenizer_bytes = static_cast<size_t>(tokenizer->data_bytes);
  if (error) *error = nullptr;
  return true;
}

bool validate_payload_crc(const uint8_t *image, size_t available_bytes,
                          const Model &model) {
  if (!image || model.image_bytes > available_bytes || model.image_bytes < sizeof(ImageHeaderV1)) return false;
  ImageHeaderV1 header;
  std::memcpy(&header, image, sizeof(header));
  if (header.data_offset > model.image_bytes) return false;
  return crc32_payload(image + header.data_offset,
                       static_cast<size_t>(model.image_bytes - header.data_offset)) == model.payload_crc32;
}

size_t kv_cache_bytes(const Model &model, int context) {
  return static_cast<size_t>(2) * model.layers * context * model.embedding * sizeof(float);
}

void clear_kv_cache(const Model &model, Scratch *scratch) {
  const size_t one_cache = static_cast<size_t>(model.layers) * scratch->context * model.embedding;
  std::memset(scratch->key_cache, 0, one_cache * sizeof(float));
  std::memset(scratch->value_cache, 0, one_cache * sizeof(float));
}

bool forward(const Model &model, int token, int position, Scratch *scratch) {
  if (!scratch || token < 0 || token >= model.vocab || position < 0 ||
      position >= scratch->context || position >= model.trained_context) return false;
  const int dim = model.embedding;
  const int head_dim = dim / model.heads;
  const int hidden = 4 * dim;
  dequantize_row(model.token_embedding, token, scratch->x);
  dequantize_row(model.position_embedding, position, scratch->norm);
  for (int index = 0; index < dim; ++index) scratch->x[index] += scratch->norm[index];

  for (int layer_index = 0; layer_index < model.layers; ++layer_index) {
    const Layer &layer = model.block[layer_index];
    layer_norm(scratch->x, layer.ln1_weight, layer.ln1_bias, dim, scratch->norm);
    matvec(layer.qkv, scratch->norm, scratch->qkv);
    float *query = scratch->qkv;
    float *key = scratch->qkv + dim;
    float *value = scratch->qkv + 2 * dim;
    const size_t cache_row = (static_cast<size_t>(layer_index) * scratch->context + position) * dim;
    std::memcpy(scratch->key_cache + cache_row, key, dim * sizeof(float));
    std::memcpy(scratch->value_cache + cache_row, value, dim * sizeof(float));
    // The upstream Head uses C=n_embd in its scale, so this is sqrt(384), not sqrt(64).
    const float attention_scale = 1.0f / std::sqrt(static_cast<float>(dim));
    for (int head = 0; head < model.heads; ++head) {
      const int head_offset = head * head_dim;
      float maximum = -std::numeric_limits<float>::infinity();
      for (int past = 0; past <= position; ++past) {
        const size_t past_row = (static_cast<size_t>(layer_index) * scratch->context + past) * dim;
        float score = 0.0f;
        for (int item = 0; item < head_dim; ++item)
          score += query[head_offset + item] * scratch->key_cache[past_row + head_offset + item];
        score *= attention_scale;
        scratch->scores[past] = score;
        maximum = std::max(maximum, score);
      }
      float denominator = 0.0f;
      for (int past = 0; past <= position; ++past) {
        scratch->scores[past] = std::exp(scratch->scores[past] - maximum);
        denominator += scratch->scores[past];
      }
      for (int item = 0; item < head_dim; ++item) {
        float sum = 0.0f;
        for (int past = 0; past <= position; ++past) {
          const size_t past_row = (static_cast<size_t>(layer_index) * scratch->context + past) * dim;
          sum += scratch->scores[past] * scratch->value_cache[past_row + head_offset + item];
        }
        scratch->attention[head_offset + item] = sum / denominator;
      }
    }
    matvec(layer.attention, scratch->attention, scratch->qkv);
    add_bias(scratch->qkv, layer.attention_bias, dim);
    for (int index = 0; index < dim; ++index) scratch->x[index] += scratch->qkv[index];
    layer_norm(scratch->x, layer.ln2_weight, layer.ln2_bias, dim, scratch->norm);
    matvec(layer.ff1, scratch->norm, scratch->ff);
    add_bias(scratch->ff, layer.ff1_bias, hidden);
    for (int index = 0; index < hidden; ++index) scratch->ff[index] = std::max(0.0f, scratch->ff[index]);
    matvec(layer.ff2, scratch->ff, scratch->attention);
    add_bias(scratch->attention, layer.ff2_bias, dim);
    for (int index = 0; index < dim; ++index) scratch->x[index] += scratch->attention[index];
  }
  layer_norm(scratch->x, model.final_norm_weight, model.final_norm_bias, dim, scratch->norm);
  matvec(model.output_head, scratch->norm, scratch->logits);
  add_bias(scratch->logits, model.output_bias, model.vocab);
  return true;
}

int sample_top_k(const float *logits, int count, float temperature, int top_k,
                 uint32_t random_value) {
  if (!logits || count <= 0) return -1;
  top_k = std::max(1, std::min(top_k, std::min(count, 64)));
  int ids[64];
  float values[64];
  int used = 0;
  for (int id = 0; id < count; ++id) {
    int at = used;
    while (at > 0 && logits[id] > values[at - 1]) --at;
    if (at >= top_k) continue;
    const int end = std::min(used, top_k - 1);
    for (int move = end; move > at; --move) {
      values[move] = values[move - 1];
      ids[move] = ids[move - 1];
    }
    values[at] = logits[id];
    ids[at] = id;
    if (used < top_k) ++used;
  }
  if (temperature <= 0.0f || top_k == 1) return ids[0];
  const float inverse_temperature = 1.0f / temperature;
  float sum = 0.0f;
  for (int index = 0; index < used; ++index) {
    values[index] = std::exp((values[index] - values[0]) * inverse_temperature);
    sum += values[index];
  }
  const float target = (static_cast<double>(random_value) / 4294967296.0) * sum;
  float cumulative = 0.0f;
  for (int index = 0; index < used; ++index) {
    cumulative += values[index];
    if (target <= cumulative) return ids[index];
  }
  return ids[used - 1];
}

}  // namespace sabir

