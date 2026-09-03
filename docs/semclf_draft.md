# Mined Prompts as Training: Building Multiclass Text Classifiers from LLM Calls Alone

## Abstract

Many organizations must classify text using only a vendor LLM API. Fine-tuning, local model hosting,
and retrieval infrastructure are frequently unavailable to them for reasons of cost, privacy, or
operational capacity, yet the classification itself must remain cheap enough to run on production
traffic. Part of this task is easy for a general-purpose model: sentiment, topic, and other categories
grounded in common knowledge are handled well with no supervision at all. The remainder is not. Operational
label sets encode application-specific conventions, and an organization routinely files a request under a
category that contradicts the literal reading of its text. A model that has never seen the organization's
data cannot know these conventions, and the standard remedies, writing the conventions into the prompt by
hand or supplying a few demonstrations, depend on an author who can articulate rules that are often tacit,
numerous, and incompletely covered by any short list of examples.

We replace that authoring step with mining. Given a modest set of labeled examples, an LLM analyzes its own
errors and produces the artifact a classifier needs: explicit filing conventions, contrastive class
descriptions, and selected demonstrations. Mining runs once, offline; inference is a sequence of LLM calls
over a fixed prompt, with no stored corpus, no trained weights, and no retrieval at prediction time. We
then ask whether the classical machinery for assembling multiclass classifiers from more stable binary
ones transfers to this setting. In place of trained base learners we use LLM calls driven by mined binary
rules, and we compare flat prediction against one-vs-all, one-vs-one, hierarchical, and error-correcting
output codes, together with a two-stage cascade in which a mined router sends only difficult items to a
focused second stage. The unifying idea is that prompt mining substitutes for training: what gradient
descent extracts into weights, an LLM extracts into text.

We evaluate on municipal 311 service requests, a domain whose native categories are set by departmental
convention rather than by semantics, over labeling budgets of 200, 1,000, and 2,000 examples, against
zero-shot, k-shot per class, retrieval-augmented few-shot, and a fine-tuned classifier.
[RESULTS SENTENCE: to be written from the final benchmark table.]

## 1. Introduction

An organization that wants to route incoming text into its own categories has, in principle, a wide choice
of methods. In practice the choice narrows quickly. Fine-tuning a model requires labeled data at a scale
many teams do not have, machine learning expertise they may not employ, and serving infrastructure they
may not wish to operate; the resulting model is specific to one organization and must be retrained as the
taxonomy drifts. Retrieval-augmented prompting removes the training step but replaces it with a different
obligation: the labeled corpus must be stored, indexed, and queried at prediction time, which is precisely
what privacy constraints and infrastructure budgets often forbid. What remains available almost everywhere
is an API call to a hosted model, billed per token.

This constraint defines the setting we study. The classifier may consist of LLM calls and text, and nothing
else. No weights are updated, no corpus is retained at inference, no vector index is consulted. The
question is how far such a classifier can be pushed, and at what cost per item.

### Two kinds of categories

A general-purpose model already classifies a large part of the world's text. Sentiment, topic, language,
and intent categories that follow from ordinary meaning are handled competently with nothing more than a
list of class names. Were all classification of this kind, the setting would be solved by zero-shot
prompting.

Operational taxonomies are different. The categories an organization actually uses are administrative
decisions, shaped by which unit owns the work, how the workflow is structured, and what the billing or
reporting system requires. The resulting label is a function of the text and of the organization's
conventions, and the two can disagree. In the municipal data we study, a request that names recycling is
filed under the trash category when the underlying issue is a missed collection, because collection
failures are owned by one unit and material sorting by another. A model reasoning from meaning alone reads
the word "recycling" and chooses accordingly. It is not making a mistake about language; it lacks a fact
about the organization.

### Why writing the conventions by hand is not enough

The obvious response is to state the conventions in the prompt, or to supply demonstrations that exhibit
them. Both are sound in principle and difficult in practice. Conventions of this kind are tacit: they are
enacted by staff rather than documented, so the person writing the prompt must first discover them.
They are numerous, one for each pair of categories that can be confused, and they interact. Demonstrations
face a complementary problem: an example shows a single decision without stating the rule behind it, so
coverage grows only as fast as the example list, and with many categories the prompt fills before the
conventions are covered.

### Mining instead of authoring

We take the conventions to be recoverable from labeled data, and we recover them with the same model that
will apply them. The procedure is error-driven. A classifier runs on labeled examples with its current
prompt; its mistakes are grouped by the pair of categories confused; a diagnostic step explains what the
recurring cases have in common and identifies which existing rule, if any, produced the error; an editing
step rewrites the rulebook, adding, clarifying, merging, or deleting rules; and a validation step keeps the
revision only when held-out accuracy improves. What emerges is a compact text artifact stating what belongs
in each category, what does not, and which triggers override the literal reading.

Mining is a one-time offline cost. The artifact it produces is a prompt, so it is inspectable by the
organization, editable by hand, portable across vendors, and cacheable across calls.

### Multiclass structure from classical machinery

Assembling a reliable multiclass classifier from simpler decisions is a well-studied problem. One-vs-all,
one-vs-one, hierarchical decomposition, and error-correcting output codes were all developed to build a
multiclass predictor out of binary learners that are individually easier to fit. Our setting invites the
same question with the base learner replaced: instead of a trained binary classifier, each decision is an
LLM call governed by mined binary rules.

We therefore evaluate the two axes of the design jointly. The first axis is what the artifact contains:
class names alone, demonstrations, mined contrastive descriptions, mined conventions, or mined exceptions
to an otherwise general classifier. The second axis is how the decision is decomposed: a single flat call,
one call per class, pairwise duels, a coarse-to-fine walk, an error-correcting code, or a cascade that
sends only flagged items to a second stage. Both axes are measured under a fixed labeling budget and a
fixed test set, with cost per item reported alongside accuracy.

### Contributions

1. A formulation of no-training, no-retrieval classification in which prompt mining takes the role of
   training, and every inference-time component is an LLM call over mined text.
2. An error-driven mining procedure that separates diagnosis from editing, attributes failures to specific
   existing rules, and gates each revision on held-out accuracy.
3. A systematic comparison of classical multiclass decompositions instantiated with LLM base learners, and
   of the artifacts that drive them, under matched labeling budgets and matched cost accounting.
4. An evaluation on a domain whose categories are convention-driven rather than semantic, including a
   diagnostic separation of items that recur verbatim in the labeled data from those that do not.
