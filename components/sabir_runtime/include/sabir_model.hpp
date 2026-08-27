#pragma once

#include <cstddef>
#include <cstdint>

namespace sabir {

constexpr uint32_t kFormatVersion = 1;
constexpr uint32_t kDtypeQ4 = 1;
constexpr uint32_t kDtypeF32 = 2;
constexpr uint32_t kDtypeBlob = 3;
constexpr int kMaxLayers = 8;

struct Q4Tensor {
  const uint8_t *codes = nullptr;
  const uint16_t *scales = nullptr;
  int rows = 0;
  int cols = 0;
  int group = 0;
  int groups_per_row = 0;
  int bytes_per_row = 0;
};

struct FloatTensor {
  const float *data = nullptr;
  int elements = 0;
};

struct Layer {
  FloatTensor ln1_weight, ln1_bias, ln2_weight, ln2_bias;
  Q4Tensor qkv, attention, ff1, ff2;
  FloatTensor attention_bias, ff1_bias, ff2_bias;
};

struct Model {
  int layers = 0;
  int embedding = 0;
  int heads = 0;
  int trained_context = 0;
  int vocab = 0;
  int group = 0;
  uint64_t image_bytes = 0;
  uint32_t payload_crc32 = 0;
  Q4Tensor token_embedding, position_embedding, output_head;
  Layer block[kMaxLayers];
  FloatTensor final_norm_weight, final_norm_bias, output_bias;
  const uint8_t *tokenizer_blob = nullptr;
  size_t tokenizer_bytes = 0;
};

struct Scratch {
  float *x = nullptr;
  float *norm = nullptr;
  float *qkv = nullptr;
  float *attention = nullptr;
  float *ff = nullptr;
  float *scores = nullptr;
  float *logits = nullptr;
  float *key_cache = nullptr;
  float *value_cache = nullptr;
  int context = 0;
};

// The caller must keep image memory mapped for the lifetime of Model.
bool load_model(const uint8_t *image, size_t available_bytes, Model *model,
                const char **error);
bool validate_payload_crc(const uint8_t *image, size_t available_bytes,
                          const Model &model);
size_t kv_cache_bytes(const Model &model, int context);
void clear_kv_cache(const Model &model, Scratch *scratch);
bool forward(const Model &model, int token, int position, Scratch *scratch);
int sample_top_k(const float *logits, int count, float temperature, int top_k,
                 uint32_t random_value);

}  // namespace sabir

