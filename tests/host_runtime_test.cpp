#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include "sabir_model.hpp"
#include "sabir_tokenizer.hpp"

template <typename T>
bool read_value(std::ifstream &stream, T *value) {
  return static_cast<bool>(stream.read(reinterpret_cast<char *>(value), sizeof(T)));
}

int main(int argc, char **argv) {
  if (argc != 3) {
    std::fprintf(stderr, "usage: host_runtime_test MODEL_IMAGE GOLDEN_BIN\n");
    return 2;
  }
  std::ifstream image_stream(argv[1], std::ios::binary);
  std::vector<uint8_t> image((std::istreambuf_iterator<char>(image_stream)), {});
  sabir::Model model;
  const char *error = nullptr;
  if (!sabir::load_model(image.data(), image.size(), &model, &error) ||
      !sabir::validate_payload_crc(image.data(), image.size(), model)) {
    std::fprintf(stderr, "model load/CRC failed: %s\n", error ? error : "CRC");
    return 1;
  }

  std::ifstream golden_stream(argv[2], std::ios::binary);
  uint32_t prompt_length = 0, vocab = 0;
  if (!read_value(golden_stream, &prompt_length) || !read_value(golden_stream, &vocab) ||
      prompt_length > 256 || vocab != static_cast<uint32_t>(model.vocab)) return 1;
  std::vector<uint16_t> expected_ids(prompt_length);
  std::vector<float> expected_logits(vocab);
  golden_stream.read(reinterpret_cast<char *>(expected_ids.data()), expected_ids.size() * sizeof(uint16_t));
  golden_stream.read(reinterpret_cast<char *>(expected_logits.data()), expected_logits.size() * sizeof(float));
  if (!golden_stream) return 1;

  std::vector<int16_t> hash(sabir::tokenizer_hash_bytes(model.vocab) / sizeof(int16_t));
  sabir::Tokenizer tokenizer;
  if (!sabir::load_tokenizer(model.tokenizer_blob, model.tokenizer_bytes,
                             hash.data(), hash.size() * sizeof(int16_t), &tokenizer, &error)) {
    std::fprintf(stderr, "tokenizer load failed: %s\n", error);
    return 1;
  }
  uint16_t actual_ids[256];
  char normalized[1024];
  const char prompt[] = u8"Kullanıcı: Nasılsın?\nModel: ";
  const int encoded = sabir::encode(tokenizer, prompt, actual_ids, 256,
                                    normalized, sizeof(normalized));
  if (encoded != static_cast<int>(prompt_length) ||
      !std::equal(actual_ids, actual_ids + encoded, expected_ids.begin())) {
    std::fprintf(stderr, "tokenizer ID mismatch\n");
    return 1;
  }

  const int dim = model.embedding, hidden = 4 * dim, context = 64;
  std::vector<float> x(dim), norm(dim), qkv(3 * dim), attention(dim), ff(hidden);
  std::vector<float> scores(context), logits(model.vocab);
  std::vector<float> keys(static_cast<size_t>(model.layers) * context * dim);
  std::vector<float> values(keys.size());
  sabir::Scratch scratch{x.data(), norm.data(), qkv.data(), attention.data(), ff.data(),
                         scores.data(), logits.data(), keys.data(), values.data(), context};
  for (uint32_t position = 0; position < prompt_length; ++position) {
    if (!sabir::forward(model, expected_ids[position], position, &scratch)) return 1;
  }
  double square_error = 0.0, dot = 0.0, actual_norm = 0.0, expected_norm = 0.0;
  float maximum_error = 0.0f;
  for (uint32_t id = 0; id < vocab; ++id) {
    const double difference = logits[id] - expected_logits[id];
    square_error += difference * difference;
    maximum_error = std::max(maximum_error, static_cast<float>(std::fabs(difference)));
    dot += static_cast<double>(logits[id]) * expected_logits[id];
    actual_norm += static_cast<double>(logits[id]) * logits[id];
    expected_norm += static_cast<double>(expected_logits[id]) * expected_logits[id];
  }
  const double rmse = std::sqrt(square_error / vocab);
  const double cosine = dot / std::sqrt(actual_norm * expected_norm);
  const int actual_top = static_cast<int>(std::max_element(logits.begin(), logits.end()) - logits.begin());
  const int expected_top = static_cast<int>(std::max_element(expected_logits.begin(), expected_logits.end()) -
                                            expected_logits.begin());
  std::printf("host runtime: tokenizer=match top1=%d/%d rmse=%.8f max=%.8f cosine=%.9f\n",
              actual_top, expected_top, rmse, maximum_error, cosine);
  return (actual_top == expected_top && rmse < 0.02 && cosine > 0.99999) ? 0 : 1;
}
