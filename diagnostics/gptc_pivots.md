```markdown
# Overall ranking

| Rank | Direction | Impact | Feasibility | Impact × feasibility | Verdict |
|---|---|---:|---:|---:|---|
| **1** | **Administrative labels as policy-dependent measurements, not semantic ground truth** | 10/10 | 9/10 | **90** | Best main-paper pivot |
| **2** | **Falsifiable certification of LLM judges via discriminant controls** | 9.5/10 | 8.5/10 | **81** | Potential standalone ACL/NeurIPS paper |
| **3** | **Judge-certified set-valued supervision for cross-domain learning** | 9/10 | 9/10 | **81** | Best algorithmic contribution |
| **4** | **Human/model ambiguity as shared task geometry → selective prediction** | 8/10 | 9/10 | **72** | Strong supporting contribution; weak alone |

The most important conceptual move is this:

> **Stop framing the 14% as “bad labels.” Frame the problem as a construct mismatch between what the text semantically supports and what an institution operationally records.**

Your Waste-vs-Trees result is actually evidence for this stronger thesis. The judge is not necessarily “wrong”; the city is not necessarily “wrong.” They are labeling **different constructs**.

A strong title family would therefore be something like:

**Administrative Gold Is Not Semantic Ground Truth: Certified Set-Valued Learning under Policy-Dependent Labels**

or, more ML-oriented:

**When Gold Depends on Policy: Separating Semantic Validity from Administrative Decisions in Cross-Domain Classification**

---

# 1. Highest-impact pivots/reframings

## Rank 1 — Turn MC311 into a paper about **policy-dependent ground truth**

### Core research question

**How much apparent domain shift in text classification is actually measurement/policy shift caused by domains assigning different operational labels to semantically equivalent inputs?**

This is much more important than “311 classification.”

Model the problem as:

- \(X\): observed text.
- \(Z\): latent semantic content or request meaning.
- \(A(X)\subseteq\mathcal{Y}\): labels defensible from the text.
- \(\Pi_j\): jurisdiction/institution \(j\)'s operational policy.
- \(Y_j^{admin}\): recorded administrative label.

Instead of assuming

\[
Y^{admin}=Z,
\]

the realistic model is

\[
Y_j^{admin}=g_j(Z,\Pi_j,U),
\]

where \(U\) contains operational information not necessarily expressed in the text.

Your Waste/Trees case becomes the canonical example:

- semantic/content construct: **Trees**
- service/routing construct: **Waste**
- administrative policy: yard-waste pickup belongs to the Waste department.

That is not ordinary label noise.

### Why this is stronger than perspectivism

Perspectivist NLP already argues against forcing all annotator disagreement into a single truth. Röttger et al. explicitly distinguish descriptive annotation, which captures beliefs, from prescriptive annotation, which implements a particular policy. 

Your opportunity is to go one step further:

**the two constructs coexist in the same operational dataset.**

MC311 contains:

1. what the utterance semantically describes;
2. what action/service the institution assigns;
3. sometimes a disagreement between those two.

That is closer to **measurement theory** than standard disagreement modeling.

Recent work has pushed construct validity into ML benchmark design, arguing that benchmark scores only support scientific claims insofar as the operationalized measurement actually corresponds to the target construct. 

### Exact new task

Create **dual-target classification**:

\[
X \rightarrow (A_{\text{semantic}},Y_{\text{service}})
\]

rather than forcing everything into one 14-class variable.

For every example estimate:

- `CONTENT`: what object/problem is described?
- `SERVICE`: what operational service owns it?
- `DEFENSIBLE SET`: which harmonized labels are textually defensible?
- `ADMIN`: what label did the city actually record?

Then learn:

\[
p(z_{\text{semantic}}\mid x)
\]

and

\[
p(y_{\text{admin}}\mid z_{\text{semantic}},j).
\]

Implement this practically as:

**shared text encoder → semantic head → jurisdiction/policy adapter → administrative head.**

### Killer experiment

Leave one city out.

Train semantic representation on six cities.

For city 7, give the model:

- 0 policy examples;
- 5/class;
- 10/class;
- 25/class;
- 50/class.

Compare:

1. pooled classifier;
2. city-token classifier;
3. domain adversarial / domain-generalization baseline;
4. soft-label training;
5. your **semantic + policy-adapter decomposition**.

### Win condition

The important result is not another +0.02 F1.

You want to show:

> A large fraction of apparent cross-city failure is recoverable with a tiny amount of policy calibration while the semantic representation transfers almost unchanged.

For example, a compelling result would look like:

- raw cross-city macro-F1 gap: 0.15;
- defensibility-adjusted gap: 0.04;
- policy-adapted model recovers 70–90% of the remaining administrative gap from 10–25 examples/class.

That would establish a new form of shift:

### **administrative-policy shift**

distinct from ordinary covariate shift, label shift, and semantic concept shift.

### Critical failure mode

If jurisdiction-conditioned adapters do no better than simply giving the city ID to a transformer, the modeling contribution is weak.

You need to demonstrate that the explicit semantic/policy decomposition improves either:

- sample efficiency,
- unseen-city transfer,
- robustness to ontology changes,
- or interpretability of failures.

---

## Rank 2 — Elevate the discriminant floor into **falsifiable LLM-judge certification**

This may actually be your most genuinely novel methodological component, but the current random-label test is **not yet sufficient for a top-tier claim**.

### Important criticism

Your current test establishes:

> the judge is not vacuous.

It does **not** establish:

> the judge correctly measures defensibility.

A pathological judge could reject random labels while systematically accepting whichever label is semantically closest in embedding space.

So make the discriminant floor the first level of a **certification hierarchy**.

### Proposed CASE certification ladder

#### Level 0: Vacuity control

Positive:

- recorded/plausible label.

Negative:

- uniformly random label.

Requirement:

\[
P(\text{accept random}) < \tau_0.
\]

Your <10% result belongs here.

#### Level 1: Hard-discriminant control

Replace random labels with deliberately difficult negatives:

- sibling category;
- nearest label by embedding similarity;
- most common confused label;
- label that shares lexical cues;
- service/content counterpart, e.g. Trees ↔ Waste.

Now define:

\[
D_{\text{hard}}
=
P(J=1\mid y^+)-
P(J=1\mid y^-_{\text{hard}}).
\]

A judge that passes random labels but fails hard negatives is **non-vacuous but non-discriminative**.

#### Level 2: Counterfactual sensitivity

Construct minimal pairs:

> “fallen tree blocking street”

versus

> “bags of leaves awaiting pickup”

and verify that judge acceptability changes in the expected direction.

This tests whether the judge responds to the **relevant evidence**.

#### Level 3: Human calibration

Take perhaps 800–1,500 stratified cases:

- unanimous CASE;
- multi-label CASE;
- administrative label rejected;
- hard judge disagreement;
- service/content conflict.

Have expert humans independently identify the acceptable set.

Report:

- sensitivity;
- specificity;
- set precision/recall;
- calibration;
- false-accept rate;
- false-reject rate.

### Why this could be a standalone paper

There is substantial work on LLM-based evaluators. G-Eval showed strong correspondence with human judgments but also evaluator biases.  JudgeBench specifically argues that LLM judges themselves need rigorous evaluation and tests judges on objectively determined response comparisons.  Panels of heterogeneous judges have also been proposed because individual judges exhibit biases and variance. 

Your differentiator should therefore **not** be:

> “we use several LLMs and check agreement.”

That territory is occupied.

The novelty claim should be:

> **Before an LLM judge is allowed to create or modify supervision, it must pass task-specific falsification controls demonstrating that it can reject deliberately invalid candidates.**

That changes LLM evaluation from “agreement with a reference” to something closer to **instrument validation**.

### Strongest version

Create a reusable benchmark:

**CASE-Cert: Falsification Tests for LLM Semantic Judges**

Across:

- MC311;
- Banking77;
- CLINC150;
- ChaosNLI;
- Civil Comments;
- perhaps EURLEX57K.

Ask whether conventional measures such as judge-human agreement predict whether a judge passes hard discriminant controls.

If not, that is an excellent finding.

---

## Rank 3 — Make acceptable-set repair into a real learning framework

Your +0.055 transfer gain is potentially more important than the benchmark itself.

But call it what it is carefully.

### What not to claim

Do not claim that “learning from multiple candidate labels” is novel.

Superset/partial-label learning has existed for years; Liu & Dietterich formalized learnability of superset-label learning at ICML 2014. 

Moreover, classical partial-label learning usually assumes:

> exactly one candidate is the hidden true label.

Your setting is different:

> **multiple labels can genuinely be acceptable simultaneously.**

That distinction matters.

### Proposed formulation

For each \(x_i\), CASE provides:

\[
A_i\subseteq\mathcal{Y}.
\]

Use the **outside-mass loss**

\[
L_{\text{set}}(x_i)
=
-\log\sum_{y\in A_i}p_\theta(y\mid x_i).
\]

It says:

> probability mass placed anywhere inside the defensible set is acceptable.

It does **not** incorrectly force a uniform distribution over acceptable labels.

This is important because:

\[
A=\{\text{Trees},\text{Waste}\}
\]

does not imply

\[
p(\text{Trees})=p(\text{Waste})=0.5.
\]

Acceptability is not probability.

### Then add the policy objective

Train jointly:

\[
L =
L_{\text{semantic-set}}
+
\lambda L_{\text{administrative}}.
\]

Semantic head:

\[
p(A\mid X)
\]

Administrative head:

\[
p(Y^{admin}\mid X,j).
\]

This directly solves the failure you are observing: naive judge repair destroys useful service-policy information.

### Exact ablations

Compare:

1. original hard administrative labels;
2. judge-replaced hard labels;
3. uniform soft labels over CASE set;
4. confidence-weighted judge targets;
5. outside-mass set loss;
6. multitask CASE-set + administrative loss;
7. multitask + city/policy adapter.

The current negative result for naive repair becomes useful:

> **semantic label repair helps transfer until it erases institutionally stable policy information.**

That motivates the model rather than undermining it.

### Strong win condition

You want all three:

- improved unseen-city semantic defensibility;
- improved or preserved administrative F1;
- elimination of the Waste/Trees negative-transfer failure.

Without all three, this is probably an ACL Findings-level contribution rather than a top-tier main-paper result.

---

## Rank 4 — Turn the 0.75 ambiguity correlation into **predictive ambiguity modeling**

Spearman 0.75 is scientifically interesting, but as currently stated it remains an observation.

The stronger question is:

> **Is ambiguity an intrinsic property of examples that generalizes across humans, LLM judges, classifier families, and domains?**

ChaosNLI already showed that NLI examples with high human disagreement account for a disproportionate share of model errors, so “humans disagree where models fail” by itself is not novel. ChaosNLI collected 100 annotations per item and showed strong model degradation on low-agreement items. 

You need to go beyond correlation.

### New task

Train an ambiguity estimator

\[
a(x)
\]

using CASE information from one set of domains/models.

Test whether \(a(x)\) predicts errors for:

- unseen classifier architectures;
- unseen foundation models;
- unseen cities;
- ideally unseen datasets.

Compare against:

- predictive entropy;
- max-softmax confidence;
- margin;
- ensemble variance;
- text length;
- class rarity;
- embedding distance to training set.

### Strong result

CASE ambiguity should predict future errors **even after conditioning on model confidence**.

For example:

\[
P(error\mid a(x),H_{model}(x))
\]

with ambiguity remaining strongly significant.

Then it is reasonable to argue that the ambiguity originates partly in **task geometry**, not merely classifier uncertainty.

### Practical payoff: selective prediction

Produce:

1. single label when ambiguity is low;
2. prediction set when several labels are defensible;
3. abstention when neither is sufficiently reliable.

Conformal prediction is a natural tool, but conformal prediction under ambiguous ground truth is already established, including a TMLR 2023 framework explicitly addressing ambiguous labels. 

Therefore the novel claim would need to be something narrower:

> conformal/selective prediction using **judge-certified defensibility sets under policy shift**, not conformal ambiguity itself.

---

# 2. Best datasets/tasks beyond 311 and CFPB

## Rank 1 — MIMIC-IV-ED: chief complaint → administrative diagnosis

### Why it is unusually well matched

MIMIC-IV-ED contains a free-text `chiefcomplaint` recorded at triage and subsequent ICD diagnoses assigned by trained coders after the ED visit for billing. 

This is almost a textbook construct-mismatch dataset:

\[
\text{what patient says}
\neq
\text{clinical interpretation}
\neq
\text{billing label}.
\]

Do **not** frame a mismatch as “incorrect medical coding.” Instead ask:

> How much of apparent text-to-code error reflects the fact that the downstream administrative label contains information that was not recoverable from the input text?

### Concrete experiment

Input:

- `chiefcomplaint` only.

Target:

- top-level ICD grouping or top 50–100 common primary ED diagnoses.

CASE asks:

> Is diagnosis \(d\) defensible from the chief complaint alone?

This is particularly powerful because CASE can explicitly answer “not inferable from this text.”

Compare:

\[
R_{\text{ICD}}
\]

versus

\[
R_{\text{text-defensible}}.
\]

### Why high impact

If you reproduce the MC311 phenomenon in medicine, the general thesis becomes very hard to dismiss as a municipal-data artifact.

### Why only medium feasibility

Medical acceptability requires real human clinical validation. A random-label discriminant floor alone is nowhere near enough.

---

## Rank 2 — Banking77 + CLINC150 + HWU64 as a **taxonomy-transfer suite**

BANKING77 contains fine-grained banking customer-service intents; CLINC150 has 150 in-scope intents plus out-of-scope queries; HWU64 provides another fine-grained intent taxonomy. CLINC150, for example, explicitly contains 150 classes with 100 train, 20 validation, and 30 test examples per in-scope intent. 

These are not as strong as MC311 for *administrative policy*, but they are excellent for testing whether:

- class boundaries are semantically unique;
- multiple intent labels are defensible;
- hard discriminant certification works;
- set-valued training helps fine-grained intent classification.

### Exact task

For every utterance \(x\):

\[
A(x)=\{y: y\text{ is semantically defensible}\}.
\]

Then compare:

- original exact-match accuracy;
- CASE-set accuracy;
- training with original labels;
- set-valued supervision.

BANKING77 is particularly interesting because many classes are operationally close:

- card payment vs cash withdrawal;
- transfer pending vs transfer failed;
- card vs cash vs transfer issues.

### Best use

External methodological validation, not your strongest domain-generalization claim.

---

## Rank 3 — Civil Comments / Jigsaw Specialized Rater Pools

This is the best dataset for validating your claim that CASE ambiguity captures **real human disagreement**.

Civil Comments exposes fractional toxicity labels corresponding to annotator vote fractions. 

The Jigsaw Specialized Rater Pools release is even more useful: it contains 25,500 comments and 382,500 disaggregated annotations across rater groups. 

### Experiment

Hide human distributions.

Run CASE.

Then test whether:

\[
H_{\text{CASE}}(x)
\]

predicts:

\[
H_{\text{human}}(x).
\]

Even better, ask whether judge-certified acceptable sets recover minority-but-substantial human interpretations better than majority-vote gold.

### Strong result

If CASE ambiguity predicts human disagreement on a completely unrelated subjective domain, it substantially strengthens your “shared ambiguity structure” story.

---

## Rank 4 — ChaosNLI

This is the cleanest controlled validation dataset.

ChaosNLI has approximately 4,645 items drawn from SNLI/MNLI/αNLI with **100 human annotations per item**. 

### Use it to validate three things

1. Does CASE identify the same examples humans find ambiguous?
2. Does the discriminant floor identify judges that better reconstruct human label distributions?
3. Does CASE ambiguity predict model errors beyond model confidence?

This is inexpensive and should almost certainly be included.

The disadvantage is novelty: ChaosNLI already established the connection between disagreement and model difficulty. Therefore use it as a **validation domain**, not as the main contribution.

---

## Rank 5 — MASSIVE

MASSIVE contains more than one million utterances, 60 intents, 55 slot types, and 52 languages. 

This gives you a particularly clean test of whether judge certification travels across languages.

### Key question

A judge might pass the discriminant floor in English but become permissive or conservative in another language.

Define:

\[
D_{J,\ell}
\]

for judge \(J\) and language \(\ell\).

Then test whether judge certification is stable across languages.

That could be a strong supporting result for the judge-certification paper.

---

## Rank 6 — EURLEX57K

EURLEX57K contains about 57k EU legal documents tagged with roughly 4.3k EUROVOC concepts and is inherently multi-label. 

Useful for:

- hierarchical acceptable sets;
- missing-label detection;
- distinguishing “another defensible concept” from “annotation error”;
- evaluating CASE when valid labels genuinely coexist.

Less useful for policy shift because the task is already explicitly multi-label.

---

# 3. Theoretical framing that would materially elevate the work

## Rank 1 — Measurement theory / construct validity

This is by far the strongest theoretical frame.

Define three things explicitly:

### Construct 1: semantic defensibility

\[
S(x,y)=1
\]

if label \(y\) is entailed or reasonably supported by text \(x\).

CASE estimates:

\[
\hat S(x,y).
\]

### Construct 2: administrative decision

\[
Y^{admin}_j.
\]

This is observed, but it is generated by an institutional process.

### Construct 3: benchmark score

Conventional evaluation computes:

\[
R_{\text{admin}}(f)
=
P[f(X)\neq Y^{admin}].
\]

But if the scientific claim is “the model understood the request,” the appropriate loss is closer to:

\[
R_{\text{semantic}}(f)
=
P[f(X)\notin A(X)].
\]

Those losses measure different constructs.

### Your central theorem-like observation

Whenever

\[
|A(X)|>1
\]

or

\[
Y^{admin}\notin A(X),
\]

exact-match administrative risk is not an unbiased measurement of semantic classification quality.

You do not need a deep theorem. A precise measurement model and propositions describing these cases would already greatly improve the paper.

### Introduce a **measurement gap**

For a model \(f\):

\[
G(f)
=
R_{\text{admin}}(f)-R_{\text{semantic}}(f).
\]

Then measure:

\[
G_j(f)
\]

for each jurisdiction.

Your observation that defensibility adjustment erases much of the apparent transfer gap becomes:

> Cross-jurisdiction evaluation contains a substantial **measurement component** in addition to actual model generalization error.

That is a much more general statement.

---

## Rank 2 — Learning with disagreement, but distinguish three causes

Do not put all disagreement into one bucket.

Define:

### Epistemic annotation noise
Annotator/judge made a mistake.

### Semantic ambiguity
More than one interpretation is supported by the observable text.

### Policy multiplicity
A single interpretation can map to different operational decisions under different policies.

The third category is the one most disagreement literature does not naturally capture.

Perspectivist NLP explicitly argues that disagreement can be meaningful and surveys methods preserving disaggregated labels rather than collapsing them into one gold standard. 

Your extension is:

\[
\text{disagreement}
=
\text{semantic uncertainty}
+
\text{perspective}
+
\text{institutional policy}.
\]

That is worth making a first-class taxonomy.

---

## Rank 3 — Selective prediction / conformal prediction

Use this mainly to turn ambiguity into an actionable task.

Let the model output:

\[
C_\alpha(x)\subseteq\mathcal{Y}.
\]

For semantic defensibility you could evaluate:

\[
P[C_\alpha(X)\cap A(X)\neq\emptyset]\ge 1-\alpha.
\]

But beware: this can be gamed with huge sets, so report:

- coverage;
- average set size;
- singleton rate;
- set precision.

An even more interesting extension is **dual coverage**:

\[
C^{sem}_\alpha(x)
\]

for defensible semantic labels, versus

\[
C^{admin}_\alpha(x,j)
\]

for the operational label.

That is conceptually clean and matches your service-vs-content decomposition.

Again, ambiguous-ground-truth conformal prediction already exists.  Your novelty would come from the dual semantic/administrative formulation and certified label sets.

---

## Rank 4 — Information-theoretic ambiguity

Useful, but supporting rather than central.

Construct:

\[
H_{\text{human}}(x),
\quad
H_{\text{CASE}}(x),
\quad
H_{\text{model}}(x).
\]

Then test whether the three uncertainty channels share information.

More interesting than simple Spearman:

\[
I(H_{\text{CASE}};E_{\text{model}}\mid H_{\text{model}},Y,\text{city})
\]

where \(E_{\text{model}}\) is the error indicator.

The empirical claim becomes:

> Certified semantic ambiguity contains information about model failure not already captured by the model's own uncertainty.

That would elevate the 0.75 correlation into a model-independent result.

---

# 4. What specifically turns this into a top-tier ACL/NeurIPS contribution

## Necessary change #1 — Replace the “MC311 benchmark” thesis with a **general learning problem**

Current resource-paper thesis:

> Here is a useful multi-city dataset; labels are surprisingly ambiguous.

Top-tier thesis:

> **Supervised text classification often treats institutionally generated decisions as semantic ground truth. We formalize the resulting construct mismatch, develop a falsifiable method for estimating text-defensible label sets, and show that separating semantic validity from institutional policy changes both cross-domain evaluation and learning.**

That is the paper.

MC311 becomes the unusually strong experimental substrate rather than the contribution itself.

---

## Necessary change #2 — Turn discriminant floor from one clever sanity check into a statistical certification protocol

Minimum credible protocol:

| Test | Negative |
|---|---|
| Random floor | random label |
| Prior control | common but unrelated label |
| Hard floor | closest wrong/sibling label |
| Confusion floor | common model-confusion label |
| Counterfactual | label valid before textual intervention, invalid after |
| Human calibration | expert acceptable sets |

For every judge report confidence intervals.

Instead of saying:

> judge accepts random labels 8%.

say something like:

\[
\Pr(a_{\text{random}}<0.10)>0.95
\]

or provide a bootstrap/binomial upper confidence bound.

Then certification actually means something statistically.

**Most important:** explicitly state that random-floor certification establishes non-vacuity, not semantic correctness.

That candor will help rather than hurt the paper.

---

## Necessary change #3 — Add at least **two external validation regimes**

I would use:

### External validation A: ChaosNLI

Purpose:

- judge-vs-human ambiguity;
- judge certification;
- ambiguity/error relationship.

### External validation B: MIMIC-IV-ED or Civil Comments

If feasibility permits, MIMIC-IV-ED is much stronger scientifically because it reproduces the distinction between observable text and downstream administrative label.

If medical validation is impractical, Civil Comments is the much easier alternative because extensive human disagreement information already exists.

Without external validation, reviewers can still reasonably say:

> this is a peculiarity of 311 ontology engineering.

---

## Necessary change #4 — Produce an algorithm that solves the failure you already discovered

Your naive label-repair failure is extremely useful.

Do **not** hide it.

Structure the paper:

1. hard administrative supervision transfers poorly;
2. CASE reveals semantic ambiguity/misalignment;
3. naive semantic repair improves overall transfer;
4. **but naive repair breaks stable operational mappings**;
5. therefore neither administrative labels nor semantic labels alone are sufficient;
6. introduce the two-level model;
7. semantic + policy learning fixes both.

That is a much better scientific narrative than reporting monotonically positive ablations.

### Final model

\[
h=Encoder(x)
\]

\[
p(z\mid x)=SemanticHead(h)
\]

\[
p(y^{admin}\mid x,j)
=
PolicyHead(h,z,e_j)
\]

where \(e_j\) is a jurisdiction/policy representation.

Train with:

\[
L =
L_{\text{CASE-set}}
+
\lambda L_{\text{admin}}.
\]

For a new city, estimate \(e_j\) from a tiny calibration set.

---

## Necessary change #5 — Beat the right baselines

At minimum:

### Label-quality baselines

- original labels;
- majority LLM judge;
- single strongest judge;
- simple ensemble judge;
- confidence filtering;
- Cleanlab/confident-learning-style label auditing;
- random relabel control, which you already have.

### Ambiguous-label baselines

- label smoothing;
- uniform soft labels;
- standard partial-label/superset learning;
- pseudo-label replacement;
- KL against judge distribution;
- your outside-set-mass objective.

### Domain-transfer baselines

- pooled ERM;
- per-city model;
- city-ID conditioning;
- leave-one-city-out ERM;
- domain-adversarial/domain-generalization baseline;
- mixture-of-experts or adapters.

### Critical baseline

**City token + ordinary classifier.**

If your sophisticated policy decomposition cannot beat this under few-shot policy adaptation, reviewers will correctly ask why the decomposition is necessary.

---

## Necessary change #6 — Make four preregisterable predictions

These would make the paper much sharper.

### P1 — Construct mismatch

\[
F1_{\text{CASE}}-F1_{\text{admin}}>0
\]

and the difference should increase on examples where the administrative label is unsupported.

### P2 — Policy-shift explanation

Cities with greater disagreement between semantic and administrative mappings should show larger apparent transfer gaps.

### P3 — Judge certification

Judges with stronger hard-discriminant scores should better predict human acceptable sets than judges selected merely by aggregate agreement.

This is especially important. It would show that certification has **predictive validity**.

### P4 — Two-level learning

Semantic-set + policy-conditioned training should dominate both:

- raw administrative training;
- naive judge repair.

Especially on the service/content conflict subset.

---

# What I would make the actual paper

## Main ACL/NeurIPS paper

### Central claim

**Cross-domain text classification can appear to fail because benchmarks conflate two targets: what an input semantically supports and what an institution chooses to do with it.**

### Contributions

1. **Formalization:** distinguish text-defensible semantic labels from policy-dependent administrative labels.
2. **Measurement:** CASE estimates acceptable label sets using an LLM ensemble that must pass explicit discriminant controls.
3. **Empirical discovery:** a substantial fraction of conventional cross-city transfer loss is measurement/policy disagreement rather than semantic failure.
4. **Learning:** judge-certified set supervision improves transfer, but naive repair fails when administrative policies are stable; a semantic + policy model resolves this.
5. **External validity:** reproduce judge/ambiguity behavior on ChaosNLI plus at least one unrelated operational or human-disagreement dataset.

### Paper-level graphical model

```text
                       jurisdiction policy Π_j
                                │
                                ▼
text X ──► semantic state Z ──► administrative label Y_admin
  │              │
  │              ▼
  └────────► acceptable set A(X)
```

CASE observes only:

\[
X,\mathcal Y
\]

and estimates \(A(X)\).

The ordinary dataset observes:

\[
X,Y^{admin}
\]

and silently assumes:

\[
Y^{admin}=Z.
\]

Your paper shows why that assumption fails.

---

# Final prioritization

## #1: Do this

**Make “semantic construct vs administrative policy” the central paper.**

It turns your most awkward current result — judge repair hurting Waste/Trees — into the result that motivates the entire method.

## #2: Build the certification ladder

The current random-label discriminant floor is genuinely interesting, but too weak alone. Add hard negatives, counterfactual tests, confidence bounds, and human calibration. Then you have something that can plausibly be claimed as **certified non-vacuous semantic judging**, rather than simply another LLM ensemble.

## #3: Replace hard repaired labels with the dual-objective set-valued model

Do not merely switch to “soft labels.” Preserve the distinction:

- acceptable-set supervision answers **what the text supports**;
- administrative supervision answers **what this institution does**.

That distinction is the scientific contribution.

## #4: Add ChaosNLI plus one genuinely different domain

ChaosNLI is nearly mandatory because it gives you 100-human-label ground truth for ambiguity. 

For the second domain:

**MIMIC-IV-ED > Civil Comments > Banking77/MASSIVE**

in scientific impact, although the ordering reverses somewhat in implementation difficulty.

---

# Bottom-line venue assessment

**As currently described:** strong resource/methodology paper, plausibly ACL/EMNLP main depending on execution, but the contribution bundle is somewhat diffuse.

**With more benchmark cities but no conceptual change:** not substantially stronger.

**With soft-label repair only:** better empirical paper, but probably still incremental because disagreement-aware and partial-label learning are established areas.

**With the policy/construct decomposition + certified judging + external validation + two-level learning:** credible **ACL main / EMNLP main** contribution and potentially **NeurIPS** if the formal task, judge-certification methodology, and multi-domain results are sufficiently general.

The single strongest paper-level sentence is:

> **The apparent ground truth in operational text classification is often the output of an institution, not a property of the text; treating these as identical creates both spurious domain-shift measurements and harmful label repair.**

That is substantially more general and defensible than “MC311 shows that 311 labels are ambiguous.”
```