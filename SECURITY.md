# Security policy

## Supported versions

This project is alpha software. Security fixes target the latest `main`
revision and are included in the next tagged release.

## Reporting a vulnerability

Do not open a public issue. Use GitHub's private **Security advisories → Report
a vulnerability** flow. If it is unavailable, contact the maintainer through
the private contact method on their GitHub profile. Include the affected
revision, board/ESP-IDF version, impact, and minimal reproduction. Do not attach
private model weights, credentials, serial identifiers, or proprietary data.

## Trust boundaries

- Model and tokenizer files are untrusted binary input. Conversion tools and
  firmware validate sizes, hashes, format metadata, and CRCs, but those checks
  do not establish provenance or licensing.
- Firmware images control a physical device. Flash only artifacts whose commit,
  build manifest, and SHA-256 checksums you have verified.
- Serial prompts and generated output are untrusted. Do not connect the demo to
  privileged automation without validation and explicit policy controls.
- The partition layout has no OTA rollback slot. A failed or malicious update
  may require a wired recovery/erase operation.
- The code is a research prototype and has not received an independent security
  audit, fault-injection review, or production hardware qualification.

## Safe use

Pin the upstream model revision, retain its source manifest, build from a clean
checkout, compare `SHA256SUMS`, and test first on a recoverable development
board. Never publish or redistribute model-derived artifacts unless their
provenance and license permit it.
