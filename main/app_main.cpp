#include <algorithm>
#include <cstdio>
#include <cstring>

#include "esp_heap_caps.h"
#include "esp_log.h"
#include "esp_partition.h"
#include "esp_psram.h"
#include "esp_random.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "sdkconfig.h"
#include "sabir_model.hpp"
#include "sabir_tokenizer.hpp"

namespace {

constexpr char kTag[] = "sabir";
constexpr int kPromptBytes = 512;
constexpr int kFormattedBytes = 768;
constexpr int kNormalizedBytes = 1536;

sabir::Model g_model;
sabir::Scratch g_scratch;
sabir::Tokenizer g_tokenizer;
esp_partition_mmap_handle_t g_mapping_handle;

void *allocate_internal(size_t bytes, const char *label) {
  void *memory = heap_caps_calloc(1, bytes, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
  if (!memory) ESP_LOGE(kTag, "internal SRAM allocation failed: %s (%u bytes)", label,
                        static_cast<unsigned>(bytes));
  return memory;
}

void *allocate_psram(size_t bytes, const char *label) {
  void *memory = heap_caps_calloc(1, bytes, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
  if (!memory) ESP_LOGE(kTag, "PSRAM allocation failed: %s (%u bytes)", label,
                        static_cast<unsigned>(bytes));
  return memory;
}

bool allocate_runtime() {
  const int dim = g_model.embedding;
  const int hidden = 4 * dim;
  const int context = CONFIG_SABIR_CONTEXT_LENGTH;
  g_scratch.context = context;
  g_scratch.x = static_cast<float *>(allocate_internal(dim * sizeof(float), "x"));
  g_scratch.norm = static_cast<float *>(allocate_internal(dim * sizeof(float), "norm"));
  g_scratch.qkv = static_cast<float *>(allocate_internal(3 * dim * sizeof(float), "qkv"));
  g_scratch.attention = static_cast<float *>(allocate_internal(dim * sizeof(float), "attention"));
  g_scratch.ff = static_cast<float *>(allocate_internal(hidden * sizeof(float), "ff"));
  g_scratch.scores = static_cast<float *>(allocate_internal(context * sizeof(float), "scores"));
  g_scratch.logits = static_cast<float *>(allocate_psram(g_model.vocab * sizeof(float), "logits"));
  const size_t cache_values = static_cast<size_t>(g_model.layers) * context * dim;
  g_scratch.key_cache = static_cast<float *>(allocate_psram(cache_values * sizeof(float), "key cache"));
  g_scratch.value_cache = static_cast<float *>(allocate_psram(cache_values * sizeof(float), "value cache"));
  return g_scratch.x && g_scratch.norm && g_scratch.qkv && g_scratch.attention &&
         g_scratch.ff && g_scratch.scores && g_scratch.logits &&
         g_scratch.key_cache && g_scratch.value_cache;
}

bool initialize() {
  if (!esp_psram_is_initialized()) {
    ESP_LOGE(kTag, "PSRAM is not initialized; N16R8/OPI PSRAM is required");
    return false;
  }
  const esp_partition_t *partition = esp_partition_find_first(
      ESP_PARTITION_TYPE_DATA, static_cast<esp_partition_subtype_t>(0x40), "model");
  if (!partition) {
    ESP_LOGE(kTag, "model partition not found");
    return false;
  }
  const void *mapped = nullptr;
  esp_err_t status = esp_partition_mmap(partition, 0, partition->size,
                                        ESP_PARTITION_MMAP_DATA, &mapped,
                                        &g_mapping_handle);
  if (status != ESP_OK) {
    ESP_LOGE(kTag, "model partition mmap failed: %s", esp_err_to_name(status));
    return false;
  }
  const char *error = nullptr;
  if (!sabir::load_model(static_cast<const uint8_t *>(mapped), partition->size,
                         &g_model, &error)) {
    ESP_LOGE(kTag, "model load failed: %s", error ? error : "unknown");
    return false;
  }
  ESP_LOGI(kTag, "validating %.2f MiB model payload CRC", g_model.image_bytes / 1048576.0);
  if (!sabir::validate_payload_crc(static_cast<const uint8_t *>(mapped),
                                   partition->size, g_model)) {
    ESP_LOGE(kTag, "model payload CRC mismatch; re-flash the model partition");
    return false;
  }
  const size_t hash_bytes = sabir::tokenizer_hash_bytes(g_model.vocab);
  void *hash = allocate_psram(hash_bytes, "tokenizer hash");
  if (!hash || !sabir::load_tokenizer(g_model.tokenizer_blob, g_model.tokenizer_bytes,
                                      hash, hash_bytes, &g_tokenizer, &error)) {
    ESP_LOGE(kTag, "tokenizer load failed: %s", error ? error : "allocation");
    return false;
  }
  if (g_tokenizer.vocab != static_cast<uint32_t>(g_model.vocab)) {
    ESP_LOGE(kTag, "tokenizer/model vocabulary mismatch");
    return false;
  }
  if (!allocate_runtime()) return false;
  ESP_LOGI(kTag, "model ready: L=%d D=%d H=%d V=%d trained_ctx=%d runtime_ctx=%d",
           g_model.layers, g_model.embedding, g_model.heads, g_model.vocab,
           g_model.trained_context, g_scratch.context);
  ESP_LOGI(kTag, "KV cache %.2f MiB, free PSRAM %.2f MiB, free internal %.1f KiB",
           sabir::kv_cache_bytes(g_model, g_scratch.context) / 1048576.0,
           heap_caps_get_free_size(MALLOC_CAP_SPIRAM) / 1048576.0,
           heap_caps_get_free_size(MALLOC_CAP_INTERNAL) / 1024.0);
  return true;
}

void generate(const char *prompt) {
  char formatted[kFormattedBytes];
  const int formatted_bytes = std::snprintf(formatted, sizeof(formatted),
                                             "Kullanıcı: %s\nModel: ", prompt);
  if (formatted_bytes <= 0 || formatted_bytes >= static_cast<int>(sizeof(formatted))) {
    std::printf("Prompt çok uzun.\n");
    return;
  }
  char normalized[kNormalizedBytes];
  uint16_t ids[256];
  const int token_count = sabir::encode(g_tokenizer, formatted, ids,
                                        static_cast<int>(sizeof(ids) / sizeof(ids[0])),
                                        normalized, sizeof(normalized));
  if (token_count <= 0) {
    std::printf("Tokenizer promptu işleyemedi.\n");
    return;
  }
  const int generation_room = g_scratch.context - token_count;
  const int steps = std::min(CONFIG_SABIR_MAX_NEW_TOKENS, generation_room);
  if (steps <= 0) {
    std::printf("Prompt %d token; context=%d. Daha kısa prompt kullanın.\n",
                token_count, g_scratch.context);
    return;
  }
  sabir::clear_kv_cache(g_model, &g_scratch);
  int position = 0;
  const int64_t started = esp_timer_get_time();
  for (int index = 0; index < token_count; ++index) {
    if (!sabir::forward(g_model, ids[index], position++, &g_scratch)) {
      std::printf("Prefill başarısız.\n");
      return;
    }
  }
  std::printf("Model:");
  std::fflush(stdout);
  int generated = 0;
  for (; generated < steps; ++generated) {
    const int next = sabir::sample_top_k(
        g_scratch.logits, g_model.vocab,
        CONFIG_SABIR_TEMPERATURE_MILLI / 1000.0f,
        CONFIG_SABIR_TOP_K, esp_random());
    if (next < 0 || next == g_tokenizer.eos_id || next == 4 || next == 5) break;
    char piece[256];
    const int bytes = sabir::decode_piece(g_tokenizer, next, piece, sizeof(piece));
    if (bytes < 0) {
      std::printf("\n[decode error id=%d]\n", next);
      return;
    }
    std::printf("%s", piece);
    std::fflush(stdout);
    if (!sabir::forward(g_model, next, position++, &g_scratch)) break;
  }
  const double seconds = (esp_timer_get_time() - started) / 1000000.0;
  std::printf("\n[%d yeni token, %.2f s, %.3f token/s; prefill dahil]\n",
              generated, seconds, seconds > 0.0 ? generated / seconds : 0.0);
}

}  // namespace

extern "C" void app_main() {
  std::setvbuf(stdin, nullptr, _IONBF, 0);
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  std::printf("\n=== Sabır-20M / ESP32-S3 offline demo ===\n");
  if (!initialize()) {
    std::printf("Başlatma başarısız; logları ve model bölümünü kontrol edin.\n");
    return;
  }
  std::printf("Ayarlar: temperature=%.3f top-k=%d max-new=%d\n",
              CONFIG_SABIR_TEMPERATURE_MILLI / 1000.0f,
              CONFIG_SABIR_TOP_K, CONFIG_SABIR_MAX_NEW_TOKENS);
  char line[kPromptBytes];
  size_t length = 0;
  bool overflow = false;
  bool skip_lf = false;
  std::printf("\nSiz> ");
  while (true) {
    const int input = std::getchar();
    if (input < 0) {
      vTaskDelay(pdMS_TO_TICKS(10));
      continue;
    }

    if (skip_lf && input == '\n') {
      skip_lf = false;
      continue;
    }
    skip_lf = false;

    if (input == '\r' || input == '\n') {
      skip_lf = input == '\r';
      std::printf("\n");
      line[length] = '\0';
      if (overflow) {
        std::printf("Prompt çok uzun; en fazla %d bayt kullanın.\n", kPromptBytes - 1);
      } else if (length) {
        generate(line);
      }
      length = 0;
      overflow = false;
      std::printf("\nSiz> ");
      continue;
    }

    if (input == '\b' || input == 0x7f) {
      if (length) {
        --length;
        std::printf("\b \b");
      }
      continue;
    }

    if (length + 1 < sizeof(line)) {
      line[length++] = static_cast<char>(input);
      std::putchar(input);
    } else {
      overflow = true;
    }
  }
}
