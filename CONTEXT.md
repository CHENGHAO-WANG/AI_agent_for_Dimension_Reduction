# dr-agent

An agent that analyses an unfamiliar dataset by dimension reduction: it measures the
data, gathers structural evidence, plans an analysis, runs it, scores the results, and
writes up what it decided and why. The language below is what the toolbox, the skills,
and the generated reports all use.

## Language

### The analysis

**Run**:
One analysis of one dataset, from profiling through to the report. Everything it
produces lives in a single directory named for it.
_Avoid_: session, job, experiment

**Plan**:
The declaration of what a Run will do — its Candidates, its Rejections, and its
Pre-registered weighting — fixed before any Embedding exists.
_Avoid_: config, recipe, protocol

**Candidate**:
A named pipeline proposed for comparison within a Run. A Candidate is a sequence of
Stages, never a bare method name.
_Avoid_: method, model, algorithm, arm

**Stage**:
One Op together with the parameter values it will run with.
_Avoid_: step, node, layer

**Op**:
A transformation the registry declares and the toolbox implements: either preprocessing
or a Reduction. The set of Ops is the whole vocabulary a Plan can be written in.
_Avoid_: operation, tool, transform

**Reduction**:
An Op that lowers dimensionality.
_Avoid_: DR method, embedder, projector

**Terminal method**:
A Reduction whose output is coordinates for viewing rather than a representation another
Stage can consume.
_Avoid_: final method, leaf, sink

**Embedding**:
The low-dimensional coordinates a Candidate produces.
_Avoid_: projection, layout, representation

**Rejection**:
A method the Plan deliberately does not run, recorded with its reason and the Evidence
keys behind it.
_Avoid_: exclusion, skip, omission

**Budget**:
The compute a Run may spend, treated as a resource the agent allocates between
Candidates rather than a limit it discovers by hitting it.
_Avoid_: quota, timeout, allowance

**Portfolio**:
The Candidates a Plan registers. It only ever grows — a Candidate that ran and lost
stays in the record — so its size is what a Run has spent of its Ceiling.
_Avoid_: candidate set, arm, sweep

**Ceiling**:
The largest Portfolio a Run may register, fixed by its Budget. Since each Candidate has
a fixed number of Attempts, this is what bounds a Run's total compute.
_Avoid_: cap, quota, limit

**Attempt**:
One execution of a Candidate, ending in an Outcome. A Candidate gets two — the first
and the one diagnose-and-retry — after which its failure is permanent.
_Avoid_: try, run, retry count

**Outcome**:
How a Candidate ended: it produced an Embedding, it failed with a reason, it exceeded
its Budget, or it died without recording anything. The last two are distinct from
failure and from each other.
_Avoid_: status, result, exit

### Representations

Three transforms sit close together and are deliberately not the same thing.

**Probe representation**:
The transform Reconnaissance applies so that its measurements describe the data rather
than an artefact of scale. Chosen by a fixed published rule, and discarded once the
measuring is done — it never produces an Embedding.
_Avoid_: preprocessing, probe transform, normalisation

**Base preprocessing**:
The Stages every Candidate in a Plan shares, chosen by the agent. Unlike the Probe
representation it is part of the analysis rather than of measuring it, and its output
survives as the Reference.
_Avoid_: shared stages, common preprocessing, pipeline prefix

**Reference**:
The representation Candidates are scored against: the output of the Base preprocessing,
or the dataset as loaded when a Plan declares none.
_Avoid_: original space, input, raw data, ground truth

**Reference value**:
A metric computed on the Reference rather than on an Embedding, so the Embedding's score
has something to be read against. It is a baseline, and an Embedding can exceed it.
_Avoid_: ceiling, upper bound, best case

### Evidence

**Profile**:
What a dataset is, measured before anything is done to it.
_Avoid_: summary, stats, description

**Reconnaissance**:
The cheap structural probes run before planning — spectrum, intrinsic dimension,
neighbourhood connectivity. Abbreviated to `recon` in command and file names only.
_Avoid_: exploration, pre-analysis, survey

**Observation**:
A statement tying a measurement to what it implies for the analysis, carrying the
Evidence keys it rests on. Distinct from the measurement itself.
_Avoid_: finding, insight, note

**Evidence key**:
A dotted path naming one value inside an artefact, such as `profile.shape.n_samples`. A
key that does not resolve is a broken rationale, not a missing value.
_Avoid_: citation, pointer, reference

**Capability record**:
What the registry declares about an Op: what it preserves, what it assumes, what it
destroys, and where it stops scaling. The agent reasons over these rather than over the
library's source.
_Avoid_: spec, metadata, documentation

**Decision log**:
The append-only record of every choice a Run made, its reasoning, and its Evidence keys.
The report is generated from it.
_Avoid_: audit trail, history, changelog

### Evaluation

**Battery**:
The fixed set of metrics every Candidate is scored on. The agent weights them; it does
not choose which of them to look at.
_Avoid_: metric suite, scorecard, criteria

**Pre-registered weighting**:
The relative importance the Plan assigns to each metric in the Battery, declared before
any Embedding is computed.
_Avoid_: weights, scoring config, priorities

**Amendment**:
A change to the Pre-registered weighting made once results exist, recorded with its
reason. The only sanctioned way for a weighting to move.
_Avoid_: adjustment, override, correction

### The agent's boundaries

**Locked core**:
The rule that method selection, execution, evaluation and ranking happen only through
audited toolbox commands, so every number in a report comes from code that can be read.
_Avoid_: sandbox, guardrails, restrictions

**Open adapter**:
The single exception to the Locked core: for an input format nothing recognises, the
agent may write a Loader.
_Avoid_: escape hatch, plugin, extension

**Loader**:
Code that turns a dataset specification into the matrix, labels and metadata the rest of
the system works with.
_Avoid_: reader, importer, parser

**Loader contract**:
The shape a Loader must return, checked before anything downstream uses the data. It
applies identically to built-in Loaders and to ones the agent writes.
_Avoid_: interface, schema, protocol
