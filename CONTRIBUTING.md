# Contributing

Thanks for wanting to help. The public engine is dual-licensed (AGPL-3.0 +
commercial). Outside contributions are only accepted under the
[Contributor License Agreement](./CLA.md) so they can ship on **both** paths.

## Before you write code

1. Open an issue describing the change (bug, design, or feature).
2. Keep patches focused. Large refactors need agreement first.
3. Match existing style; don't reformat unrelated files.

## CLA (required) + DCO (required)

**CLA:** explicit dual-license grant to the maintainer. This is what keeps
commercial licensing viable after the first outside PR. Read
[CLA.md](./CLA.md). Until GitHub CLA tooling is wired up, put this in every PR:

```
I have read and agree to the PagedMoE CLA (CLA.md).
```

**DCO:** per-commit provenance (`Signed-off-by`). Certifies you have the
right to submit the work. See [DCO.md](./DCO.md).

```bash
git commit -s -m "Your message"
```

They are not substitutes for each other: the DCO is the trail; the CLA is the
relicensing grant. If your employer owns your work product, get approval
before you contribute.

## What we accept

- Bug fixes with a clear reproduction
- Performance / correctness improvements with evidence
- Docs that make the system easier to run honestly (limits included)
- Tests

## What we push back on

- Drive-by dependency churn
- Unmeasured "optimizations"
- Features that only work on one private checkpoint format
- Anything that weakens licensing / copyright notices

## Security

Report security issues privately per [SECURITY.md](./SECURITY.md)
(`noah@qwertycode.org`) rather than opening a public issue with exploit
detail.

## Code of Conduct

Participation is governed by the [CODE_OF_CONDUCT.md](./CODE_OF_CONDUCT.md)
(Contributor Covenant).
