# Licensing

**PagedMoE** is dual-licensed.

| Path | Who it's for | Terms |
| --- | --- | --- |
| **AGPL-3.0** | individuals, researchers, open projects, evaluation | Free to use, study, modify, and share under AGPL-3.0  -  see [`LICENSE`](./LICENSE) |
| **Commercial** | companies that need proprietary / closed-source use | Paid license from the copyright holder  -  see [`COMMERCIAL_LICENSE.md`](./COMMERCIAL_LICENSE.md) |

Copyright © 2026 Noah Clark. Dual licensing does **not** transfer copyright.
It gives you a choice of which grant applies to your use.

## What AGPL-3.0 allows (the open path)

You may:

- Run PagedMoE yourself (including for work / research)
- Read and audit the source
- Modify it for your own use
- Publish forks and improvements **under AGPL-3.0**

If you **distribute** a modified version, or offer a modified version as a
**network service** (SaaS / hosted API / product backend), AGPL-3.0 generally
requires you to provide the corresponding source to users under AGPL-3.0
(see AGPL §13).

## What AGPL-3.0 does *not* silently allow

You may **not**, under the AGPL grant alone:

- Take this code into a closed-source commercial product without releasing
  your AGPL-covered source
- Relicense the code under MIT/Apache/proprietary terms
- Strip copyright / authorship notices

### Internal use (read carefully)

“Just running it inside our company” is **not** a bright-line safe harbor
under AGPL the way it often is under MIT/Apache.

- Unmodified, single-user / local use is the low-friction case.
- The network clause is aimed at remote interaction with a **modified**
  program. How that applies to large orgs (other teams hitting an internal
  service, contractors, subsidiaries) is **genuinely debated** among counsel.
- If you’re unsure whether your setup triggers source obligations, that
  uncertainty is exactly what a [commercial license](./COMMERCIAL_LICENSE.md)
  removes.

When in doubt, get a commercial license  -  or get your own lawyer’s opinion.

## What the commercial license is for

Companies buy a commercial license when they want to:

- Ship PagedMoE (or a derivative) inside a proprietary product
- Offer it as a hosted / SaaS service **without** AGPL source obligations
- Clear away internal / network-use ambiguity with an explicit grant
- Get warranty / support terms (negotiated)

That is the dual-license model: **open contribution + clear paid exception**.

## Contributors

Outside contributions are accepted only under the
[Contributor License Agreement](./CLA.md) (dual-license grant to the
maintainer) plus [DCO](./DCO.md) sign-off. See
[`CONTRIBUTING.md`](./CONTRIBUTING.md).

A DCO alone is **not** enough for commercial relicensing of third-party
code. As long as Noah Clark is the sole author, dual licensing is
straightforward; the CLA is what keeps that true after the first outside PR.

## Third-party dependencies

Runtime dependencies such as **mlx**, **mlx-lm**, and **huggingface_hub**
are under permissive licenses (MIT / Apache-2.0). Those licenses are
compatible with including them alongside an AGPL-licensed work. They remain
under their own terms; PagedMoE does not relicense them. See
[`NOTICE`](./NOTICE).

## Contact

Commercial licensing and dual-license questions:

**Noah Clark**  -  `noah@qwertycode.org`

Please include: company name, intended use (embedded product / SaaS / internal
tooling), and whether you need support.
