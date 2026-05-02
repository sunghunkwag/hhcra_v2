# ARCH_MUTATION_PLAN: Causal Self-Correction Loop

## 1. Current Architecture Map

HHCRA v2 is a causal reasoning architecture currently centered around `hhcra_v2.py`.

### Layer 1: Perception / Variable Extraction

- `CJEPA`
  - Encodes high-dimensional observations into latent vectors.
  - Decomposes latent vectors into causal variable slots.
  - Exposes `extract_variables(...)` and `handle_feedback(...)`.
  - Existing feedback is primitive: `increase_resolution` perturbs slot parameters.

### Layer 2: Mechanism / Graph + Dynamics

- `CausalGNN`
  - Learns directed causal adjacency logits.
  - Uses `W[child, parent]` convention for edge logits.
  - Produces soft and hard adjacency matrices.
  - Enforces no self-loops and prunes to DAG.

- `LiquidNeuralNet`
  - Evolves latent variables over learned graph structure.
  - Supports intervention through edge cutting and clamping.

- `MechanismLayer`
  - Couples `CausalGNN` and `LiquidNeuralNet`.
  - Converts learned adjacency into `CausalGraphData` through `symbolic_graph()`.
  - Existing `handle_feedback(...)` supports ad-hoc dictionary commands: `add_edge`, `remove_edge`, `weaken_edge`.

### Layer 3: Symbolic / Hierarchical Reasoning

- `CausalGraphData`
  - Symbolic graph representation.
  - Exposes parents, children, ancestors, descendants, edge existence, edge count, DAG verification.

- `NeuroSymbolicEngine`
  - Performs d-separation, backdoor/frontdoor search, and identifiability checks.

- `HRM`
  - Performs iterative hierarchical reasoning over query embeddings.

### Existing Feedback Path

The code claims diagnostic feedback coupling from Layer 3 -> Layer 2 -> Layer 1. The primitive hooks exist, but the feedback path is not first-class. It is currently represented by loosely typed dictionaries rather than causal failure objects, mutation proposals, and before/after validation audits.

## 2. Current Bottleneck

The deepest bottleneck is not numerical optimization or formatting. The architectural bottleneck is that causal reasoning failure is not represented as a machine-readable self-correction signal.

HHCRA v2 can reason over a graph and can manually receive low-level feedback, but it lacks an explicit architecture path:

```text
causal failure -> diagnostic signal -> mutation proposal -> graph update -> revalidation audit
```

This blocks qualitative growth because the model cannot turn symbolic causal failure into structured pressure on its own mechanism layer. Without that path, feedback coupling remains a claim rather than an executable loop.

## 3. Three Possible Architecture Mutations

### Candidate A: Typed Diagnostic and Mutation Interfaces

Introduce:

- `CausalFailureType`
- `CausalDiagnosticSignal`
- `StructureMutationProposal`

Then map symbolic causal failures into graph-level mutation proposals.

**Structure changed:** feedback moves from ad-hoc dictionaries to typed causal objects.  
**Bottleneck attacked:** lack of machine-readable causal failure representation.  
**Capability unlocked:** reasoning failures become auditable mutation pressure.  
**Files/classes:** new adapter module plus tests.  
**Invariant risk:** graph mutation may create cycles or invalid edges if unchecked.  
**Validation:** missing path creates diagnostic, proposal, and graph logit update.  
**Risk:** low.

### Candidate B: CausalSelfCorrectionLoop Orchestrator

Introduce an orchestrator:

```text
query -> graph inspection -> diagnostic -> proposal -> MechanismLayer update -> revalidation
```

**Structure changed:** creates a new Layer-3-to-Layer-2 architecture path.  
**Bottleneck attacked:** no closed self-correction loop.  
**Capability unlocked:** before/after audit of a causal structure mutation.  
**Files/classes:** new adapter module, architecture note, executable test.  
**Invariant risk:** graph update must preserve no-self-loop and node bounds.  
**Validation:** deliberately incomplete graph gets corrected by a typed proposal.  
**Risk:** medium-low.

### Candidate C: Counterfactual Consistency Checker

Introduce a checker that compares observational, interventional, and counterfactual outputs, then emits diagnostics when they conflict.

**Structure changed:** adds cross-rung consistency validation.  
**Bottleneck attacked:** no consistency pressure across Pearl's ladder.  
**Capability unlocked:** causal outputs can critique each other.  
**Files/classes:** would require deeper integration with existing inference outputs.  
**Invariant risk:** higher, because current outputs may not share a clean common representation.  
**Validation:** requires more than one query class to be coherently exposed.  
**Risk:** high for a first slice.

## 4. Chosen Mutation

Chosen: **Candidate B**, implemented through Candidate A's typed interfaces.

Reason: Candidate B creates the actual qualitative architecture path, while Candidate A provides the minimal typed substrate. Candidate C is promising but too broad for the first vertical slice.

## 5. Expected Qualitative Capability Gain

Before this mutation:

- HHCRA v2 has causal reasoning components.
- HHCRA v2 has primitive feedback hooks.
- But causal failure does not become a typed, auditable, executable mutation signal.

After this mutation:

- A missing causal path can be detected as a `CausalDiagnosticSignal`.
- The signal can be converted into a `StructureMutationProposal`.
- The proposal can be safely applied to `MechanismLayer` graph logits.
- A before/after audit shows that the graph structure changed in the expected direction.

This is a minimal self-correction loop, not a claim of full autonomous RSI.

## 6. Validation Method

Add an executable unit-style validation that:

1. Constructs a small HHCRA mechanism layer with three variables.
2. Suppresses all graph edges.
3. Constructs an empty symbolic graph.
4. Queries whether path `0 -> 1` exists.
5. Emits a `MISSING_CAUSAL_PATH` diagnostic.
6. Converts it into an `add_edge` mutation proposal.
7. Applies the proposal to `MechanismLayer`.
8. Revalidates through `symbolic_graph()`.
9. Asserts that edge `0 -> 1` now exists.

## 7. Files Likely to Change

- `ARCH_MUTATION_PLAN.md`
- `hhcra_self_correction.py`
- `docs/self_correction_loop.md`
- `tests/test_causal_self_correction_loop.py`

The existing `hhcra_v2.py` is intentionally preserved in this first slice so the original single-file runnable demo remains untouched.

## 8. Rollback Plan

Rollback is direct:

1. Delete `hhcra_self_correction.py`.
2. Delete `tests/test_causal_self_correction_loop.py`.
3. Delete `docs/self_correction_loop.md`.
4. Delete `ARCH_MUTATION_PLAN.md`.

No existing code path is overwritten in this first vertical slice.

## 9. Coherence Gate Result

- Architecture-level impact: PASS. It adds a new Layer-3-style diagnostic to Layer-2 graph mutation path.
- Minimal vertical slice: PASS. It implements one failure type and one graph mutation.
- Measurable gain: PASS. Edge absence becomes edge presence through typed diagnostic feedback.
- Testability: PASS. The validation is executable with `unittest`.
- Rollbackability: PASS. All changes are additive.
- No cosmetic cleanup: PASS.
- No unsupported overclaim: PASS. The mutation is explicitly limited to a primitive causal self-correction loop.
