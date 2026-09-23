#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include "sabir_model.hpp"

#pragma pack(push, 1)
struct Header {
  char magic[8];
  uint32_t version, header_bytes, flags;
  uint32_t n_layer, n_embd, n_head, block_size, vocab_size, group_size;
  uint32_t tensor_count, directory_entry_bytes, payload_crc32;
  uint64_t directory_offset, data_offset, image_size;
};

struct Entry {
  char name[28];
  uint8_t dtype, ndim;
  uint16_t reserved0;
  uint32_t dims[4];
  uint64_t data_offset, data_bytes, aux_offset;
  uint32_t aux_bytes, reserved1;
};
#pragma pack(pop)

static_assert(sizeof(Header) == 80 && sizeof(Entry) == 80, "fixture layout");

std::vector<uint8_t> fixture() {
  std::vector<uint8_t> image(256, 0);
  Header header{};
  std::memcpy(header.magic, "SABIR4", 6);
  header.version = 1;
  header.header_bytes = sizeof(Header);
  header.n_layer = 8;
  header.n_embd = 384;
  header.n_head = 6;
  header.block_size = 256;
  header.vocab_size = 8000;
  header.group_size = 64;
  header.tensor_count = 1;
  header.directory_entry_bytes = sizeof(Entry);
  header.directory_offset = sizeof(Header);
  header.data_offset = 192;
  header.image_size = image.size();
  std::memcpy(image.data(), &header, sizeof(header));
  Entry entry{};
  std::memcpy(entry.name, "unknown", 7);
  entry.dtype = sabir::kDtypeBlob;
  entry.ndim = 1;
  entry.dims[0] = 1;
  entry.data_offset = 192;
  entry.data_bytes = 1;
  std::memcpy(image.data() + sizeof(Header), &entry, sizeof(entry));
  return image;
}

template <typename T>
void write_at(std::vector<uint8_t> &image, size_t offset, const T &value) {
  std::memcpy(image.data() + offset, &value, sizeof(value));
}

void expect_error(const std::vector<uint8_t> &image, size_t available,
                  const char *expected) {
  sabir::Model model;
  const char *error = nullptr;
  assert(!sabir::load_model(image.data(), available, &model, &error));
  assert(error && std::strcmp(error, expected) == 0);
}

int main() {
  auto image = fixture();
  expect_error(image, image.size(), "missing embeddings");
  expect_error(image, 79, "short model header");
  expect_error(image, image.size() - 1, "invalid model directory");

  image = fixture();
  Header header;
  std::memcpy(&header, image.data(), sizeof(header));
  header.image_size = 100;
  header.data_offset = 160; // Directory would extend beyond the mapped image.
  write_at(image, 0, header);
  image.resize(100);
  expect_error(image, image.size(), "invalid model directory");

  image = fixture();
  std::memcpy(&header, image.data(), sizeof(header));
  header.directory_offset = 0; // Header bytes cannot be read as directory entries.
  write_at(image, 0, header);
  expect_error(image, image.size(), "invalid model directory");

  image = fixture();
  Entry entry;
  std::memcpy(&entry, image.data() + sizeof(Header), sizeof(entry));
  entry.data_bytes = 65; // 192 + 65 exceeds the actual 256-byte image.
  write_at(image, sizeof(Header), entry);
  expect_error(image, image.size(), "invalid tensor range");

  image = fixture();
  std::memcpy(&entry, image.data() + sizeof(Header), sizeof(entry));
  entry.aux_offset = 256;
  entry.aux_bytes = 1;
  write_at(image, sizeof(Header), entry);
  expect_error(image, image.size(), "invalid tensor range");

  image = fixture();
  std::memcpy(&entry, image.data() + sizeof(Header), sizeof(entry));
  entry.data_offset = 193; // Reject unaligned FP16/FP32 payloads.
  write_at(image, sizeof(Header), entry);
  expect_error(image, image.size(), "invalid tensor range");

  image = fixture();
  std::memcpy(&entry, image.data() + sizeof(Header), sizeof(entry));
  entry.aux_offset = 192; // Even an unused auxiliary offset must be valid.
  write_at(image, sizeof(Header), entry);
  expect_error(image, image.size(), "invalid tensor range");

  std::puts("model bounds tests passed");
}
