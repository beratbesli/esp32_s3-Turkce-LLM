# Release readiness

The project is alpha. A GitHub Actions artifact is a CI convenience, not a
durable signed release. Publish a tagged release only after every applicable
gate below is satisfied.

## Required gates

- Host Python tests and C++ runtime build pass from a clean checkout.
- ESP-IDF 5.5.5 builds the `esp32s3` target from the tagged commit.
- The artifact contains bootloader, partition table, application image,
  `flash_args`, `flasher_args.json`, `SHA256SUMS`, and `BUILD-MANIFEST.txt`.
- The manifest identifies the exact repository commit, workflow run, target,
  and ESP-IDF version; every published file matches `SHA256SUMS`.
- The release notes distinguish host/CI evidence from physical-board evidence
  and link the hardware validation matrix.
- Upstream model revision, model/data license status, and redistribution rights
  are reviewed. Model weights and generated model images are excluded unless
  their terms are documented and permit redistribution.
- Flash/recovery instructions, known limitations, and the no-OTA constraint are
  visible in the release notes.

## Recommended provenance

Create the release from a protected tag, retain the workflow run, and attach an
artifact attestation or signature when release automation is added. Verify the
checksum file after download and before flashing. Never promote an expiring CI
artifact as a permanent release URL.
