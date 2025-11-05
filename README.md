# LiveCap Core

Core runtime components for LiveCap, extracted into a standalone Python package.
This repository will host the streaming transcription pipeline, engine adapters,
and shared utilities that power both the LiveCap GUI and headless deployments.

## Licensing

LiveCap Core is offered under a dual-license model:

- **Open Source:** GNU Affero General Public License v3.0 (AGPL-3.0). The full
  text is provided in `LICENSE`. Any derivative work that is distributed or
  offered as a network service must comply with the AGPL terms.
- **Commercial:** A separate commercial license is available for organizations
  that need to embed LiveCap Core without the reciprocal obligations of the
  AGPL. See `LICENSE-COMMERCIAL.md` for contact details and high-level terms.

## Status

> ⚠️ **Early extraction phase**  
> The codebase is migrating from the monolithic LiveCap repository. API surface,
> packaging metadata, and CI workflows are still being finalized. Expect rapid
> iteration until the first 1.0.0 release candidate.

## Roadmap (condensed)

1. Bootstrap packaging (`pyproject.toml`, `uv.lock`, minimal CI).
2. Migrate `livecap_core/` runtime modules and accompanying tests.
3. Publish pre-release artifacts to TestPyPI (`1.0.0a*`, `1.0.0b*`, `1.0.0rc*`).
4. Coordinate 1.0.0 GA with the LiveCap GUI repository.

For the full migration plan, refer to
[`docs/dev-docs/architecture/livecap-core-extraction.md`](https://github.com/Mega-Gorilla/Live_Cap_v3/blob/main/docs/dev-docs/architecture/livecap-core-extraction.md).

## Getting Involved

- Issues & feature requests: use the tracker in this repository once it opens
  for public contributions.
- Security inquiries or commercial licensing: contact PineLab via the channels
  listed in `LICENSE-COMMERCIAL.md`.

Stay tuned for contributor guidelines and API documentation as the split
progresses.
