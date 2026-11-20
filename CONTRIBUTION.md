# Contributing

Thanks for your interest in this C++17 latent-diffusion research and
productionisation repository. Contributions are reviewed for correctness,
reproducibility, and fit with the current roadmap.

## Licensing and contribution rights

The repository is licensed under **Creative Commons
Attribution-NonCommercial 4.0 (CC BY-NC 4.0)** as described in [LICENSE](LICENSE).
It is not a permissive commercial software licence. Do not assume that submitting
a contribution changes those terms or grants commercial-use permission.
Contributors must have the right to submit their code and must identify
third-party code or dependencies and their licences. Do not include proprietary,
confidential, patented third-party implementation details or incompatible
licensed code. For commercial licensing requests, use the contact information
in the repository's LICENSE file.

Contributions may be discussed or declined before any code is accepted.
No contributor licence agreement or transfer of ownership is implied by this
document. If a separate rights agreement is needed, it must be explicit.

## Development workflow

1. Open an issue describing the problem, proposed interface, test plan, and
   relevant milestone. For larger architecture or API changes, agree on the
   design before implementing it.
2. Work on a feature branch from the current default branch; keep changes
   focused on one ticket.
3. Preserve public C++17 compatibility and maintain separate production,
   experimental Clifford, and legacy NLP paths. Do not mix experimental code
   into the production `latent_core` without explicit approval.
4. Write deterministic numerical tests, including known-answer cases,
   invalid inputs, boundary conditions and regression coverage where relevant.
5. Add or update Doxygen documentation for public APIs: `@brief`, `@param`,
   `@return`, `@throws`, numerical conventions, ownership and lifetime.
6. Update the relevant checklist and changelog with tested results, clearly
   separating implementation from verification.
7. Submit a pull request explaining the mathematical rationale, compatibility,
   exact commands run, observed results and known limitations.

## Build and testing

See [COMMANDS.md](COMMANDS.md) for supported commands. At minimum, run:

```bash
bash scripts/test.sh --p0
bash scripts/test.sh --all
```

Run additional focused tests when updating a component. Release tests,
Apple Metal tests, documentation generation and clean-clone validation must
be reported separately; do not claim them passed based on Debug CPU results.

Generate public API documentation with:

```bash
bash scripts/run_documentation.sh
```

Doxygen must be installed for this optional command. Do not commit the
generated HTML, object files, CMake caches or other build artifacts.

## Code and mathematics guidelines

- Prefer explicit shapes, timesteps, seeds and caller-provided noise over
  implicit global state.
- Preserve input Tensors unless a public API explicitly says otherwise.
- Validate shape, finite values and timestep bounds at public boundaries.
- Keep deterministic mathematical steps separate from stochastic sampling.
- State index conventions and FP64 reference tolerances in test evidence.
- Avoid introducing third-party runtime dependencies without prior discussion.
- Use explanatory comments where reasoning is non-obvious, particularly for
  posterior coefficients and numerical edge cases.
- Preserve existing validated behaviour and add regression tests for fixes.

## Pull request checklist

- [ ] Change is scoped to one problem and has a clear rationale.
- [ ] Public API compatibility and numerical conventions have been reviewed.
- [ ] Relevant deterministic unit tests added or updated.
- [ ] `bash scripts/test.sh --p0` run and results supplied.
- [ ] `bash scripts/test.sh --all` run and results supplied.
- [ ] Doxygen comments updated for changed public APIs.
- [ ] Checklist, status or changelog updated with actual evidence.
- [ ] New dependencies and licensing implications disclosed.
- [ ] Generated files and confidential material excluded.

A successful pull request review does not automatically constitute milestone
sign-off. Milestones are signed off only after the relevant validation evidence
has been reviewed.
