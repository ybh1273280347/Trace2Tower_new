# Trace2Skill: Distill Trajectory-Local Lessons into Transferable Agent Skills

Jingwei Ni<sup>§,*2,3</sup>, Yihao Liu<sup>§,*4</sup>, Xinpeng Liu<sup>§,*4</sup>, Yutao Sun<sup>§,*5</sup>, Mengyu
Zhou<sup>†1</sup>, Pengyu Cheng<sup>1</sup>, Dexin Wang<sup>1</sup>, Erchao Zhao<sup>1</sup>, Xiaoxi Jiang<sup>1</sup>
and Guanjun Jiang<sup>1</sup>

<sup>1</sup>Qwen Large Model Application Team, Alibaba, <sup>2</sup>ETH Zürich, <sup>3</sup>University of Zurich, <sup>
4</sup>Peking University, <sup>5</sup>Zhejiang University

Work done during an internship at Alibaba. <sup>†</sup>Corresponding author. <sup>§</sup>Core Contributors.

Large Language Model (LLM) agents increasingly rely on domain-specific skills, yet manually authoring such skills does
not scale, and skills generated purely from parametric knowledge often miss critical operational pitfalls. We introduce
Trace2Skill, a framework that consolidates broad execution trajectories in parallel into a unified skill directory
through inductive reasoning over agent experience. Trace2Skill supports both deepening existing human-written skills and
creating useful skills from weak LLM-generated drafts. Experiments demonstrate the efectiveness of Trace2Skill across
diverse domains, including ofice workflows, math reasoning, and vision QA. Importantly, the evolved skills are not
merely memorized artifacts of the trajectories used to create them: they often transfer across model scales, across
model families, and to out-of-distribution settings. For example, skills evolved from Qwen3.5-35B trajectories improve a
Qwen3.5-122B agent by up to 57<sup>.</sup>65 percentage points on WikiTableQuestions. Further analyses show that
Trace2Skill outperforms sequential skill editing and ReasoningBank-style retrieval memories, compresses recurring
failures and workarounds into standard operating procedures (SoPs), and yields portable skills that can be reused
without parameter updates or test-time retrieval.<sup>a</sup>

<sup>a</sup>Code: https://github.com/Qwen-Applications/Trace2Skill

## 1. Introduction

LLM-based agents increasingly rely on skills: reusable documents that encode task procedures, domain knowledge, and
operational guidelines (Anthropic, 2026d; Zhou et al., 2026b). As agents move into specialized file- and tool-use
workflows, the bottleneck is no longer only model capability, but also the ability to create and maintain high-quality
skills for each domain (Han et al., 2026; Li et al., 2026a; Anthropic, 2026b; Liang et al., 2026; Zhou et al., 2026b).
Human-written skills can help, but they are not uniformly beneficial across agents: in Table 1, the oficial xlsx skill
improves a 122B spreadsheet agent while hurting a 35B agent on the same benchmark. Generating skills purely from
parametric knowledge is also brittle, because such drafts often lack the concrete failure modes, workarounds,

![](images/692686aacd00041f4913a77d8b7b44d8a4b07e424c45b947210b85a4b9539334.jpg)

Figure 1 | Left: sequential online skill evolution edits the skill after each incoming trace. Right: Trace2Skill
analyzes many traces in parallel and hierarchically consolidates recurring lessons into one portable skill.

and operational details exposed by actual execution traces (Li et al., 2026b; Jiang et al., 2026; Zhou et al., 2026b).

Recent systems therefore use agent experience to evolve skills or memories online (Yang et al., 2026; Xia et al., 2026a;
Alzubi et al., 2026; Zhou et al., 2026a; Jiang et al., 2026; Zhou et al., 2026b). This direction is promising, but many
existing approaches either store trajectory-local lessons for retrieval or edit skills sequentially as new traces
arrive (Fig. 1, left). Such designs can fragment reusable knowledge across a large memory or skill collection, and
sequential editing can make later skills depend on the order of earlier updates (Li, 2026). Human experts usually work
diferently: they inspect broad traces, abstract recurring patterns, and write compact procedures that are reusable
across cases.

![](images/a87d139be3e38ee4a099ec8fc5685d4a1300beadb27313e24ae4f0be97c90102.jpg)

Figure 2 | Overview of Trace2Skill’s three-stage pipeline: (1) roll out a frozen agent to collect labeled success and
failure trajectories, (2) propose trajectory-level skill patches in parallel with separate error and success analysts,
and (3) hierarchically merge all patches into one portable skill directory.

We introduce Trace2Skill, a framework for turning execution traces into portable skills. Rather than retrieving
per-episode memories at test time or absorbing traces through order-dependent sequential edits, Trace2Skill analyzes
many traces jointly and consolidates recurring lessons into a single skill directory (Fig. 1, right). The same mechanism
supports two common use cases: deepening an existing human-written skill and creating a useful skill from a weak
LLM-generated draft. This many-to-one consolidation acts as an inductive reasoning step over agent experience (Xiong et
al., 2025; Li et al., 2025; Lin et al., 2025), while preserving the portability of ordinary skill files.

Experiments show that trajectory-grounded skills improve performance while remaining portable across models, benchmarks,
and task domains. In spreadsheet workflows, Trace2Skill both strengthens Anthropic’s oficial xlsx skill and creates
useful skills from scratch. The resulting skills transfer across model scales and families: for example, a skill evolved
with Gemma-4-31B-it (Google, 2026) improves Qwen3.5-122B (Team, 2026). They also generalize to out-of-distribution (OOD)
data and tasks, such as transferring from spreadsheet editing to table QA. The benefits extend beyond spreadsheets:
Trace2Skill improves math reasoning and DocVQA, and further strengthens Anthropic’s oficial PDF, DOCX, and PPTX workflow
skills. Further analyses show that: parallel consolidation is much faster and generally stronger than order-dependent
skill editing; a single distilled skill outperforms ReasoningBank-style episodic retrieval (Ouyang et al., 2026); and
agentic analysts produce better patches by inspecting artifacts and validating fixes. Qualitatively, the learned patches
are not trajectory-specific tips: they coalesce into reusable SoPs, while patch-selection studies show that their value
is often combinatorial, making holistic consolidation more reliable than greedy local selection.

We contribute: (1) Trace2Skill, a framework for automatic skill creation and deepening. It mirrors human skill writing:
building broad prior knowledge through extensive trajectory analysis before drafting skills (§ 2). (2) Broad empirical
evidence that trajectory-grounded evolution yields skills that transfer efectively across LLM scales, families, and OOD
tasks (§ 3). (3) Comprehensive analysis showing why consolidation works: parallel merging improves over sequential
updates, one consolidated skill beats retrieval-based reasoning memories, agentic diagnosis improves patch quality, and
patch value is often combinatorial (§ 4).

## 2. Trace2Skill

Fig. 2 shows the three-stage Trace2Skill pipeline: collect trajectories, propose trajectory-level patches in parallel,
and consolidate them into one portable skill. We first define the skill-evolution objective, then describe each stage.

## 2.1. Skill and Problem Formalization

A skill is a human-readable directory $\boldsymbol { S } = ( M , \mathcal { R } )$ , where <sup>??</sup> is the root
SKILL.md and $\mathcal { R }$ contains auxiliary references, scripts, or assets. The root document stores broadly
applicable procedural knowledge, while auxiliary files provide deterministic tools or lower-frequency details.
Let $\pi _ { \theta }$ be a fixed LLM agent using skill S at inference time, and
let $\mathcal { P } ( S ; \pi _ { \boldsymbol { \theta } } , \mathcal { D } )$ denote the pass rate of that agent on
task set ${ \mathcal { D } } .$ Given an evolving set $\mathcal { D } _ { \mathrm { e v o l v e } }$ and a disjoint test
set $\mathcal { D } _ { \mathrm { t e s t } }$ , skill evolution constructs a new skill from evolving-set trajectories
without updating <sup>??</sup>:

$$
\mathcal {S} ^ {*} = \mathcal {E} (\mathcal {S} _ {0}, \mathcal {D} _ {\mathrm{evolve}}; \pi_ {\theta}), \qquad \mathcal {P} (\mathcal {S} ^ {*}; \pi_ {\theta}, \mathcal {D} _ {\mathrm{test}}) > \mathcal {P} (\mathcal {S} _ {0}; \pi_ {\theta}, \mathcal {D} _ {\mathrm{test}}).\tag{1}
$$

We evaluate two initializations for $\textstyle s _ { 0 } !$ : a human-expert skill and an LLM-generated draft from
parametric knowledge alone.

## 2.2. Stage 1: Trajectory Generation

We use a ReAct-style harness (Yao et al., 2023). For each evolving-set task, the fixed agent runs with $S _ { 0 }$ and
produces a trajectory $\tau _ { i }$ containing the query, reasoning/tool-use history, final output, and a binary
correctness outcome. The resulting corpus $\mathcal { T }$ is split into failures $\mathcal { T } ^ { - }$ and
successes ${ { \mathcal { T } } ^ { + } }$ . Trajectory generation is parallel across evolving-set problems, so the
trace corpus can be collected independently before patch proposal. See prompt templates in § I.1.

## 2.3. Stage 2: Parallel Patch Proposal

A group of analyst sub-agents independently proposes skill patches from individual trajectories. Failures are sent to an
error analyst $\mathcal { A }$ <sup>−</sup>, successes to a success analyst $\mathcal { A } ^ { + }$ , and the analysts
read $\scriptstyle { S _ { 0 } }$ and propose a patch for $\operatorname { i t } ,$ leading to a patch
pool $\mathcal { P } = \mathcal { P } ^ { - } \cup \mathcal { P } ^ { + }$

The analyst roles are intentionally asymmetric. $\mathcal { A } ^ { + }$ uses a single-pass workflow to identify
reusable behavior patterns from successful trajectories. $\mathcal { A } ^ { - }$ uses a ReAct-style loop that can
inspect traces and artifacts, compare outputs against ground truth, and validate candidate fixes before proposing a
patch. Failures that cannot be causally explained are excluded, ensuring $\mathcal { P } ^ { - }$ is grounded in
verified failure mechanisms rather than log-only guesses (Ouyang et al., 2026). Both roles are prompted to write
concise, actionable patches following skill-writing guidance (Anthropic, 2026b). Analyst prompt templates and
representative example patches are provided in § I.2.

## 2.4. Stage 3: Patch Consolidation

Stage 3 consolidates the patch pool into one coherent update $p ^ { * }$ and applies it to $S _ { 0 }$ . Patches are
merged hierarchically for $L = \lceil \log _ { B _ { \mathrm { m e r g e } } } | \mathcal { P } | \rceil$ levels; at
each level $\ell ,$ up to $B _ { \mathrm { m e r g e } }$ patches are synthesized into one patch:

$$
p ^ {(\ell + 1)} = \mathcal {M} \Big (\pi_ {\theta}, \mathcal {S} _ {0}, \{p _ {1} ^ {(\ell)}, \ldots , p _ {B _ {\mathrm{merge}}} ^ {(\ell)} \} \Big), \qquad \ell = 0, \ldots , L - 1,\tag{2}
$$

where M deduplicates, resolves conflicts, and preserves non-overlapping insights. $\pi _ { \theta }$ itself serves as
trajectory generator, analyst, and merge operator, so no external evolution/teacher model is required. The
final $p ^ { * }$ is translated into dif-style edits and applied with deterministic guardrails: reject edits to missing
files, withhold line-range conflicts, and validate the updated skill format.

The merge also performs inductive generalization. Because each patch comes from one trajectory, recurring edits across
independent patches are evidence of systematic task properties rather than one-of fixes. M is therefore instructed to
prefer prevalent patterns and discard idiosyncratic edits, producing a compact skill that replaces $S _ { 0 }$ and is
used directly at inference without retrieval. The merge operator prompt template and an example consolidated
patch $p ^ { * }$ are given in § I.3.

Trace2Skill supports two modes that cover scenarios with and without expert-written skills: Skill deepening starts with
a human-written skill, and Skill creation starts from a skill generated from LLM parametric knowledge. Both improve the
target skill using patches induced from trace analysis.

Table 1 | Main spreadsheet results across skill-author (evolves the skill) and skill-user (runs it at inference) models.
SpreadsheetBench reports Vrf (verified), Soft (problem pass rate), and Hard (task pass rate); WikiTQ and HiTab are
out-of-distribution (OOD) table-QA transfer,
and $\mathrm { A v } \mathbf { g } { = } ( \mathrm { I D } _ { \mathrm { a v g } } { + } \mathrm { O O D } _ { \mathrm { a v g } } ) / 2$
equally weights indistribution (ID, SpreadsheetBench) and OOD. Reference rows are absolute scores; evolved rows are
signed deltas from the Human-Written (Deepening) or Parametric (Creation) baseline. Bold marks the largest absolute
score per column. Per-seed standard deviations are in Table 7.


<table><tr><td rowspan="3">Condition</td><td colspan="5">Skill User: Qwen3.5-122B-A10B</td><td colspan="5">Skill User: Qwen3.5-35B-A3B</td><td rowspan="3">Avg↑</td></tr><tr><td colspan="3">SpreadsheetBench</td><td colspan="2">OOD</td><td colspan="3">SpreadsheetBench</td><td colspan="2">OOD</td></tr><tr><td>Vrf↑</td><td>Soft↑</td><td>Hard↑</td><td>WikiTQ↑</td><td>HiTab↑</td><td>Vrf↑</td><td>Soft↑</td><td>Hard↑</td><td>WikiTQ↑</td><td>HiTab↑</td></tr><tr><td colspan="12">Baseline (absolute scores)</td></tr><tr><td>No Skill</td><td>27.67</td><td>28.90</td><td>17.57</td><td>21.50</td><td>14.42</td><td>19.00</td><td>18.00</td><td>4.60</td><td>13.33</td><td>19.12</td><td>18.19</td></tr><tr><td>Human-Written</td><td>48.33</td><td>36.30</td><td>17.03</td><td>74.68</td><td>41.31</td><td>9.67</td><td>13.03</td><td>3.37</td><td>9.02</td><td>5.30</td><td>26.93</td></tr><tr><td>Parametric</td><td>26.17</td><td>36.60</td><td>17.50</td><td>23.73</td><td>17.36</td><td>20.17</td><td>13.70</td><td>3.87</td><td>20.14</td><td>13.94</td><td>19.23</td></tr><tr><td colspan="12">Skill Author: Qwen3.5-122B-A10B</td></tr><tr><td colspan="12">Deepening (Delta from init: Human-Written)</td></tr><tr><td>+Error</td><td>+17.50</td><td>+10.30</td><td>+10.40</td><td>+1.62</td><td>+2.14</td><td>+27.00</td><td>+9.44</td><td>+2.86</td><td>+9.26</td><td>+9.55</td><td>+9.28</td></tr><tr><td>+Success</td><td>+18.00</td><td>+8.60</td><td>+8.70</td><td>-10.35</td><td>-0.95</td><td>+19.66</td><td>+6.84</td><td>+1.33</td><td>+12.09</td><td>+22.33</td><td>+8.15</td></tr><tr><td>+Combined</td><td>+21.50</td><td>+10.87</td><td>+12.50</td><td>+4.56</td><td>+2.97</td><td>+21.16</td><td>+8.84</td><td>+1.80</td><td>+6.64</td><td>+11.94</td><td>+9.65</td></tr><tr><td colspan="12">Creation (Delta from init: Parametric)</td></tr><tr><td>+Error</td><td>+22.83</td><td>+3.77</td><td>+5.87</td><td>+7.89</td><td>-1.70</td><td>+8.66</td><td>+9.53</td><td>+4.00</td><td>+2.06</td><td>+9.64</td><td>+6.79</td></tr><tr><td>+Success</td><td>+15.33</td><td>-0.93</td><td>+4.33</td><td>+23.70</td><td>+19.89</td><td>+12.83</td><td>+11.57</td><td>+6.13</td><td>+30.36</td><td>+28.90</td><td>+16.96</td></tr><tr><td>+Combined</td><td>+14.00</td><td>-0.63</td><td>+3.53</td><td>+32.32</td><td>+23.11</td><td>+15.50</td><td>+14.50</td><td>+7.23</td><td>+29.70</td><td>+27.71</td><td>+18.62</td></tr><tr><td colspan="12">Skill Author: Qwen3.5-35B-A3B</td></tr><tr><td colspan="12">Deepening (Delta from init: Human-Written)</td></tr><tr><td>+Error</td><td>+16.67</td><td>+8.50</td><td>+8.14</td><td>-6.36</td><td>-2.38</td><td>+17.33</td><td>+9.17</td><td>+4.83</td><td>+2.71</td><td>+8.08</td><td>+5.64</td></tr><tr><td>+Success</td><td>+2.17</td><td>+2.73</td><td>+3.30</td><td>+1.46</td><td>+1.70</td><td>+12.33</td><td>+5.87</td><td>+1.23</td><td>+43.23</td><td>+34.70</td><td>+12.44</td></tr><tr><td>+Combined</td><td>+6.67</td><td>+3.87</td><td>+4.17</td><td>+2.65</td><td>+2.44</td><td>+20.00</td><td>+5.77</td><td>+2.36</td><td>+42.20</td><td>+36.46</td><td>+14.04</td></tr><tr><td colspan="12">Creation (Delta from init: Parametric)</td></tr><tr><td>+Error</td><td>+1.00</td><td>-7.70</td><td>+1.03</td><td>+57.65</td><td>+28.25</td><td>+3.83</td><td>+7.30</td><td>+2.66</td><td>+12.66</td><td>+17.81</td><td>+15.22</td></tr><tr><td>+Success</td><td>+5.33</td><td>-4.57</td><td>+2.43</td><td>+9.09</td><td>+2.34</td><td>+5.66</td><td>+5.80</td><td>+2.63</td><td>+3.31</td><td>+18.94</td><td>+5.65</td></tr><tr><td>+Combined</td><td>+8.33</td><td>-5.83</td><td>+2.00</td><td>+30.82</td><td>+16.93</td><td>+4.33</td><td>+9.73</td><td>+4.73</td><td>+18.00</td><td>+25.22</td><td>+13.31</td></tr></table>

## 3. Experiments

## 3.1. Experimental Setup

Spreadsheet setup. Our main experiments use SpreadsheetBench-Verified (Ma et al., 2024), where agents manipulate xlsx
files through tool use. We split its 400 samples into 200 evolution problems and 200 held-out test problems, and also
report Soft/Hard scores on full SpreadsheetBench (excluding all evolving-set problems) plus OOD transfer to
WikiTableQuestions (Pasupat & Liang, 2015) and HiTab (Cheng et al., 2022) converted into spreadsheet format. All
spreadsheet results are averaged over three random seeds using each benchmark’s oficial evaluation criteria.

Skill settings. We compare No Skill, the Anthropic xlsx skill (Human-Written), LLM-generated skills using Parametric
knowledge, and three Trace2Skill variants: +Error, +Success, and +Combined, which consolidate patches from failed
trajectories only, from successful trajectories only, and from all trajectories, respectively. Skill Deepening starts
from Human-Written, while Skill Creation starts from Parametric. We evaluate Qwen3.5- 122B-A10B and Qwen3.5-35B-A3B as
both skill authors and skill users. We do 100% self-evolution: the same model generates trajectories, proposes patches,
and edits skills. Details of dataset construction, scoring, and model serving are in § A; the external skill-creator
baseline is in § H.

## 3.2. Main Results

Table 1 reports all spreadsheet results across skill conditions, author models, user models, and transfer tasks.
Baseline rows give absolute scores; evolved rows give signed deltas from the relevant baseline, with Deepening compared
to Human-Written and Creation compared to Parametric. We use Avg as the primary summary metric because it equally
weights ID and OOD performance across both user models, rewarding skills that generalize across models and tasks.

![](images/d64ecf2144a187d2899fd911bed2b5f5f28806d9e99c17c73309154f9cbba1d1.jpg)

![](images/e5422fd8b17809e414e0b0f2b07c03e696a7c432976ef823c7047d2e929706ea.jpg)

No trace Qwen traces Gemma-4 traces GPT-5.5 trace

Figure 3 | Cross-family generalization on SpreadsheetBench. Gemma-4-31B-it and GPT-5.5-high each (i) evolve a skill from
their own traces and (ii) run Qwen3.5-authored skills.

Baselines reveal why both settings matter. Human-Written is a useful handcrafted prior, but it is not reliably portable
across model scales; Parametric remains close to No Skill, confirming that parametric knowledge alone does not yield
actionable spreadsheet skills (Han et al., 2026). These baselines motivate the two evaluation regimes: Deepening tests
whether a strong manual skill can be made more transferable, while Creation tests whether trajectory-grounded
distillation can build a useful skill from a weak seed.

Both Deepening and Creation produce generalizable skills. Starting from Human-Written, evolved skills consistently
strengthen in-distribution spreadsheet performance and often transfer to other model scales and OOD table tasks;
starting from Parametric, Creation can match or exceed Human-Written quality in favorable settings. The gains are broad:
they appear across author models, user models, and task families, not only on the model that produced the traces.
Notably, 35B Deepening +Combined attains the best absolute Avg, showing that skills authored by a small LLM can also
generalize. Across analysts, both +Error and +Success help spreadsheet tasks, and +Combined usually gives the largest
Avg improvement.

## 3.3. Model-Family Generalization

We test generalization across model families with Gemma-4-31B-it and GPT-5.5-high in two settings: each model (i)
deepens the human-written xlsx skill from its own traces via Trace2Skill, and (ii) runs the Qwen3.5- 122B/35B +Combined
deepened skills. Fig. 3 shows that both models self-improve from their own traces and also benefit from Qwen3.5-authored
skills, so Trace2Skill generalizes across families. § B.4 reports additional results and implementation details.

## 3.4. Math Reasoning

We apply Trace2Skill to math domain to test its domain-agnosticism. As in the spreadsheet setting (§ 3.1), the math
agent runs in a ReAct loop with a command-line Python interpreter to write and execute code for each question. We create
skills from scratch on 400 DAPO questions (Yu et al., 2025) and evaluate on 100 disjoint held-out DAPO questions (ID)
and AIME 2026 (OOD competition mathematics; avg@8 over 30 problems), following the cross-model protocol of § 3.2. Table
2 reports deltas from No Skill. Trace2Skill works for math domain: the distilled skills improve both held-out DAPO and
OOD AIME rather than overfitting the source distribution. +Error is the most stable signal, transferring cleanly between
the 122B and 35B users.

## 3.5. Visual Question Answering

To test multimodal generalization, we apply Trace2Skill to DocVQA (Mathew et al., 2020), where agents answer questions
over document images such as forms, tables, invoices, letters, and reports. Again, each agent runs in a ReAct loop with
a command-line + Python environment. We use 50 DocVQA examples only for skill evolution and remove them from evaluation;
the remaining 5,299 examples form the held-out test set. We report the

Table 2 | Math transfer to held-out DAPO and AIME. D-Test is DAPO-Math-Test pass rate; AIME is avg@8.


<table><tr><td rowspan="2">Condition</td><td colspan="2">Skill User: 122B</td><td colspan="2">Skill User: 35B</td></tr><tr><td>D-Test↑</td><td>AIME↑</td><td>D-Test↑</td><td>AIME↑</td></tr><tr><td colspan="5">Reference (absolute scores)</td></tr><tr><td>No Skill</td><td>91.0</td><td>90.4</td><td>89.0</td><td>83.3</td></tr><tr><td colspan="5">Skill Author: Qwen3.5-122B-A10B</td></tr><tr><td colspan="5">Creation (init: No Skill)</td></tr><tr><td>+Error</td><td>+4.0</td><td>+2.9</td><td>+5.0</td><td>+5.0</td></tr><tr><td>+Success</td><td>+2.0</td><td>+0.4</td><td>+2.0</td><td>+4.6</td></tr><tr><td>+Combined</td><td>+3.0</td><td>-1.2</td><td>+6.0</td><td>+5.5</td></tr><tr><td colspan="5">Skill Author: Qwen3.5-35B-A3B</td></tr><tr><td colspan="5">Creation (init: No Skill)</td></tr><tr><td>+Error</td><td>+3.0</td><td>+1.3</td><td>+4.0</td><td>+0.5</td></tr><tr><td>+Success</td><td>-1.0</td><td>-1.2</td><td>+0.0</td><td>-1.2</td></tr><tr><td>+Combined</td><td>+1.0</td><td>-0.4</td><td>+1.0</td><td>+0.0</td></tr></table>


Table 3 | DocVQA transfer from trajectory-distilled visual-reasoning skills. ANLS is similarity; Acc is ANLS ≥ 0.5.


<table><tr><td rowspan="2">Condition</td><td colspan="2">Skill User: 122B</td><td colspan="2">Skill User: 35B</td></tr><tr><td>ANLS↑</td><td>Acc↑</td><td>ANLS↑</td><td>Acc↑</td></tr><tr><td colspan="5">Reference (absolute scores)</td></tr><tr><td>No Skill</td><td>0.6300</td><td>70.27</td><td>0.6582</td><td>73.17</td></tr><tr><td colspan="5">Skill Author: Qwen3.5-122B-A10B</td></tr><tr><td colspan="5">Creation (init: No Skill)</td></tr><tr><td>+Error</td><td>+0.1949</td><td>+17.69</td><td>+0.1974</td><td>+16.64</td></tr><tr><td>+Success</td><td>+0.1639</td><td>+14.89</td><td>+0.1668</td><td>+14.23</td></tr><tr><td>+Combined</td><td>+0.2534</td><td>+22.25</td><td>+0.2049</td><td>+17.22</td></tr><tr><td colspan="5">Skill Author: Qwen3.5-35B-A3B</td></tr><tr><td colspan="5">Creation (init: No Skill)</td></tr><tr><td>+Error</td><td>+0.0884</td><td>+6.79</td><td>+0.1267</td><td>+10.15</td></tr><tr><td>+Success</td><td>+0.1572</td><td>+14.53</td><td>+0.2132</td><td>+18.27</td></tr><tr><td>+Combined</td><td>+0.0958</td><td>+7.73</td><td>+0.2158</td><td>+18.83</td></tr></table>


oficial ANLS and Accuracy $\left( \mathrm { A N L S } \geq 0 . 5 , \% \right)$ for both same-model use and cross-model
transfer between 122Band 35B-authored skills. Results in Table 3 show that all evolved DocVQA skills improve performance
over No Skill. +Combined is the strongest setting, giving the clearest same-model gains and positive cross-model
transfer.

## 4. Analysis

This section analyzes why Trace2Skill works and its broader application. We first isolate its three core design choices
under a shared trace pool and execution harness (§ 4.1), then characterize the standard operating procedures it
distills (§ 4.2), study how patch value composes (§ 4.3), and test transfer in broader real-world tasks (§ 4.4).

## 4.1. Core Design Comparisons

Trace2Skill rests on three design choices, which we isolate here by comparing each against its natural alternative under
the same trace pool and execution harness: parallel many-to-one consolidation versus sequential editing (speed without
quality loss), a single consolidated skill versus experience memory retrieval (reuse without test-time retrieval), and
agentic versus single-call error analysis (more accurate root-cause patches).

Table 4 | Parallel consolidation versus sequential editing on SpreadsheetBench (same trace pool, +Error). Parallel
outperforms Seq in efectiveness and eficiency. Table 8 and Table 10 show similar trend in math/VQA.


<table><tr><td rowspan="2">Condition</td><td colspan="3">Skill User: 122B</td><td colspan="3">Skill User: 35B</td><td rowspan="2">Time↓</td></tr><tr><td>Vrf↑</td><td>Soft↑</td><td>Hard↑</td><td>Vrf↑</td><td>Soft↑</td><td>Hard↑</td></tr><tr><td>Seq-B=4</td><td>59.00</td><td>40.63</td><td>20.63</td><td>26.17</td><td>22.37</td><td>7.47</td><td>~15 min</td></tr><tr><td>Seq-B=1</td><td>61.83</td><td>44.40</td><td>25.40</td><td>26.00</td><td>23.83</td><td>10.57</td><td>~60 min</td></tr><tr><td>Parallel (ours)</td><td>65.83</td><td>46.60</td><td>27.43</td><td>27.00</td><td>22.20</td><td>8.20</td><td>~3 min</td></tr></table>

Parallel Consolidation. Online skill evolution typically edits the skill as trajectory batches arrive, so later analyses
depend on earlier edits. We isolate the efect of our parallel many-to-one consolidation by comparing against two
sequential baselines: Seq-<sup>??</sup>=1, which updates after every trajectory, and Seq-<sup>??</sup>=4, which updates
after every four trajectories. All conditions use error analysts only and initialize from the Human-Written skill. Table
4 reports the resulting quality and wall-clock tradeof.

![](images/5867e64e2cd93308341656aa1e0a5696fa35f7dfd1d606581ac23fcbcfb0f3bc.jpg)

Figure 4 | Agentic +Error versus single-call +Error LLM on Avg, the balance of ID and OOD performance (same as Avg in
Table 1). The agentic analyst inspects artifacts and validates fixes, whereas +Error LLM reads only the execution log.

![](images/afe6792d4ad9b0d20a17964b946d9ed322dff6c19510baef54b73d6a2b7b0a9f.jpg)

Figure 5 | Greedy selective patch aggregation on SpreadsheetBench-Vrf (pass rate, %). Each curve adds one
validation-selected patch per iteration (<sup>??</sup>-axis) from the Error, Success, or Combined pool; the flat line is
full Trace2Skill aggregation of all patches.

Parallel consolidation is better on all 122B SpreadsheetBench metrics and on 35B Vrf; the exception is that
Seq-<sup>??</sup>=1 is modestly higher on 35B Soft and Hard. This small quality tradeof comes with a large eficiency
gap: parallel produces the skill in about 3 min, compared with 15 min for Seq-<sup>??</sup>=4 and 60 min for
Seq-<sup>??</sup>=1. Math reasoning and DocVQA show the same broad trend that parallel consolidation preserves or
improves quality while being much faster; § B gives the latency analysis and similar comparisons on math/VQA.

Holistic Skill vs. Retrieval. Past-experience retrieval is a common paradigm for reusing agent trajectories:
experience-memory systems store reflections, memories, or procedural lessons from previous executions and retrieve
relevant items at inference time (Shinn et al., 2023; Wang et al., 2023a; Ouyang et al., 2026; Fang et al., 2026; Wang
et al., 2024b; Liu et al., 2025).

We instantiate this paradigm with Reasoning-Bank (Ouyang et al., 2026): following its original setting, we store lessons
from both success and failure trajectories and retrieve the top-1 memory with Qwen3-Embedding-8B. We compare it against
+Combined, which uses the same trajectory pool. Results on same-model Deepening are shown in Table 5. +Combined is
consistently better than ReasoningBank. This supports the advantage of consolidating trajectory evidence into a compact
skill rather than retrieving isolated memories at test time. We exclude OOD evaluations, as retrieval is not applica-

Table 5 | Trace2Skill outperforms ReasoningBank-style retrieval on SpreadsheetBench (same trajectory pool). Table 9 and
Table 11 show similar trend in math/VQA.


<table><tr><td>Setting</td><td>User</td><td>Vrf↑</td><td>Soft↑</td><td>Hard↑</td></tr><tr><td rowspan="2">ReasoningBank (Ouyang et al., 2026)</td><td>122B</td><td>56.00</td><td>40.10</td><td>21.30</td></tr><tr><td>35B</td><td>20.50</td><td>17.30</td><td>4.97</td></tr><tr><td rowspan="2">Human-Written + Combined (ours)</td><td>122B</td><td>69.83</td><td>47.17</td><td>29.53</td></tr><tr><td>35B</td><td>29.67</td><td>18.80</td><td>5.73</td></tr></table>

ble for WikiTQ and HiTab whose queries are too semantically distant from the queries of SpreadsheetBench.

Agentic Error Analysis. Related work derives transferable lessons or skills from error trajectories via a single
non-interactive LLM call (Ouyang et al., 2026; Xia et al., 2026a; Yang et al., 2026; Jiang et al., 2026). We ablate this
design with +Error LLM, where one LLM call reads each failed trajectory and proposes a patch without inspecting
artifacts, querying ground truth, or validating fixes; Fig. 4 compares it with our agentic error analyst. Agentic +Error
achieves higher Avg than +Error LLM in most settings, showing that interactive diagnosis is more useful than log-only
patch generation; § D reports and discusses the full per-dataset results in Table 12. The qualitative analysis in § C.1
shows the mechanism: artifact access and fix validation help the agentic analyst identify root causes more precisely.

Table 6 | BO-selected +Error patches versus applying all of them (No Selection), in the 122B Deepening spreadsheet
setting. Evolve is the accuracy on evolving-set.


<table><tr><td>Skill</td><td>Evolve</td><td>Vrf</td><td>Soft</td><td>Hard</td><td>WikiTQ</td><td>HiTab</td></tr><tr><td>No Selection (+Error)</td><td>68.00</td><td>65.83</td><td>46.60</td><td>27.43</td><td>76.30</td><td>43.45</td></tr><tr><td>BO Selection (+Error)</td><td>72.83</td><td>69.83</td><td>47.27</td><td>28.90</td><td>77.51</td><td>43.38</td></tr></table>

Apples-to-apples vs. head-to-head comparison. The above three comparisons are apples-to-apples, isolating each core
design without the data, model, harness, and engineering confounders that complicate whole-system evaluation. As a
complementary view, § G reports a direct head-to-head against three concurrent systems (XSkill (Jiang et al., 2026),
EvoSkill (Alzubi et al., 2026), and SkillGen (Ma et al., 2026)) under a shared open model and benchmark, where
Trace2Skill still shows its advantage.

## 4.2. SoPs Learned

Many-to-one merging keeps the operations that recur across trajectories and turns them into standard operating
procedures (SoPs), rather than collecting one-of tricks. Across the 323 map patches from the 122B Deepening +Combined
run, four learned SoPs dominate, each cited by between 16% and 55% of patches (a single patch can cite more than one
SoP), including: recalculating and reading back formulas after writes, verifying target cells before submitting, and
deleting rows in a corruption-safe order. Task-specific quirks are routed to on-demand references/ files, keeping the
SKILL.md focused on broadly reusable procedures (Details in § C.3).

## 4.3. Selective Patch Aggregation

Trace2Skill aggregates all learned patches into the target skill, avoiding the cost of per-patch validation. If patches
vary in quality, can selecting a subset do better? We test two selectors against full aggregation: greedy top-1, which
at each iteration adds the locally best patch according to validation on evolving set, and Bayesian optimization (BO),
which searches binary patch-inclusion vectors and reuses validation scores from previously tried subsets to bias later
proposals toward promising combinations. Both use the 200-task evolving set (§ 3.2) and a 32-task validation set from
evolving set; § E gives the full algorithmic details.

Greedy rises quickly but plateaus below full aggregation. Fig. 5 shows greedy +Error and +Combined rising for a few
iterations and then plateauing, with +Success dipping and only partially recovering; every greedy curve stays below full
Trace2Skill aggregation. Two mechanisms drive the plateau (detailed in § C.2). (1) patch-irrelevant regression: the
patches flip some target tasks from wrong to correct, but their side efects also flip previously-correct tasks from
correct to wrong, so net accuracy stalls. (2) semantic overlap: recurring errors lead the pipeline to propose patches
that repeatedly target the same behaviors (e.g., formula recalculation, validation checklists), so a new patch largely
restates existing guidance and adds little marginal gain. Patch value is therefore combinatorial rather than a sum of
singletons, which makes optimizing the combined efec of patches necessary.

BO is useful but computationally heavy. BO searches patch subsets directly, so it can model complementarity and
interference that greedy misses. Table 6 shows that, compared with applying all +Error patches, BO improves most
selected metrics, especially the SpreadsheetBench columns. The cost is validation: each candidate subset must be
materialized as a skill and run on the validation set before it can inform the posterior, and larger patch universes
increase this cost quickly. We therefore view BO as useful when the target distribution and validation budget are known,
rather than as a replacement for the default full aggregation in Trace2Skill.

## 4.4. Broader Application

We test broader application of Trace2Skill in PDF extraction, PPTX editing, and DOCX editing. These settings use
Anthropic’s oficial pdf, pptx, and docx skills as starting points (Anthropic, 2026a) and retain the same execution style
as the spreadsheet experiments: agents operate over files with command-line tools and are scored by task-specific local
verifiers.

Across the three domains, evolution on source traces improve performance on separate held-out targets. For PDF, VRDU
traces transfer to VAREX, raising pass rate from 76.9% to 85.3% (Wang et al., 2023b; Barzelay et al., 2026). For PPTX,
TSBench traces transfer to a deck-disjoint TSBench OOD split, improving 72.5% to 88.8% (Jung et al., 2026). For DOCX,
evolving on synthetic tasks that cover generic DOCX operations transfers to the OficeBench DOCX subset, improving 79.7%
to 87.5% (Wang et al., 2024a). § F provides setup and implementation details for each domain.

## 5. Related Work

Agent skills. Agent skills package task procedures, domain knowledge, and operational guardrails into loadable
artifacts (Anthropic, 2026d), but ecosystems and benchmarks show the abstraction is delicate: focused, wellmatched
skills improve performance, while broad, stale, or mismatched ones can distract or even hurt the agent (Zhou et al.,
2026b; Li et al., 2026b; Han et al., 2026; Li, 2026; Li et al., 2026a; Liang et al., 2026). Trace2Skill keeps this view
of skills as portable SoPs, but asks a narrower question: how to compress broad execution evidence into guidance that
stays useful across models and task distributions.

Experience memory for agent self-evolution. Another line improves agents by storing execution experience for reuse, from
verbal reflection and accumulated behaviors to retrieval, procedural, workflow, or replay memories queried at test
time (Shinn et al., 2023; Wang et al., 2023a; Ouyang et al., 2026; Fang et al., 2026; Wang et al., 2024b; Qian et al.,
2024; Nottingham et al., 2024; Liu et al., 2025). These methods share our premise that experience contains reusable
structure but retain an external memory or retrieval module, whereas Trace2Skill distills many local observations into
one static skill directory. It favors inductive compression over nearest-neighbor reuse and avoiding test-time
dependence on retrieval quality.

Skill and policy evolution. The closest concurrent line evolves skills from agent interaction or guided refinement (
Zheng et al., 2025; Yang et al., 2026; Jiang et al., 2026; Alzubi et al., 2026; Zhou et al., 2026a; Ma et al., 2026;
Anthropic, 2026b), while a broader family co-evolves policies, skills, or agent loops online through skill-augmented
reinforcement learning, intrinsic skill evolution, or meta-learning (Xia et al., 2026a; Li et al., 2026c; Xia et al.,
2026b). Trace2Skill asks a complementary question: whether traces collected with one model can be compressed into a
static skill directory that directly benefits other models and tasks. Instead of maintaining a test-time memory,
retrieval index, or updated policy, it consolidates trajectory-local evidence into reusable SoPs that can be loaded
unchanged. Thus, our focus is not adaptive reuse within a particular agent loop, but portability of the learned artifact
across deployment settings.

## 6. Conclusion

We introduced Trace2Skill, a framework that distills agent execution traces into a portable skill: parallel analyst
sub-agents propose targeted patches from disjoint trajectory batches, and a consolidation step merges them at once into
one declarative skill directory. Skills distilled from a single model’s traces transfer across model scales, families,
and OOD tasks. Trace2Skill also exhibits strong applicability in various domains.

## Limitations

Trace2Skill applies all consolidated patches by default; selecting a higher-quality subset with Bayesian optimization
can help (§ 4.3), but it is costly. A reliable selection signal needs a large, representative validation set and BO must
materialize and score a new skill for every candidate subset, so cost grows quickly with both the validation set and the
patch universe. Ideally, using the whole evolving set for validation would better reflect patch quality than our sampled
validation with only 32 questions, but it would be much more computationally expensive. We therefore leave a more
thorough exploration of combinatorial patch selection to future work.

## Ethics Statement

We use publicly available datasets, which have no data privacy issues. All artifacts we use are under licenses allowing
research usage. Human annotation in qualitative analyses were conducted by the authors of this paper.

We do not identify any other ethical risks associated with this study. AI coding tools (Codex/Claude Code) are used to
assist with coding. We confirm that all such coding is under careful human supervision. Unit tests are implemented to
avoid hacking or unaligned behavior. We will also open-source the code for full reproducibility. We also leverage
ChatGPT for grammar check and fix, fully supervised by human authors.

## References

Salaheddin Alzubi, Noah Provenzano, Jaydon Bingham, Weiyuan Chen, and Tu Vu. Evoskill: Automated skill discovery for
multi-agent systems, 2026. URL https://arxiv.org/abs/2603.02766.

Anthropic. PDF, DOCX, and PPTX document skills. GitHub repository, 2026a. URL https://github.com/
anthropics/skills/tree/main/skills. Accessed: 2026-05-01; document-skill subfolders: pdf, docx, and pptx.

Anthropic. How to create a skill with claude through conversation. Claude Tutorials, 2026b.
URL https://claude.com/resources/tutorials/ how-to-create-a-skill-with-claude-through-conversation.

Anthropic. skill-creator. GitHub repository, 2026c. URL https://github.com/anthropics/skills/
tree/main/skills/skill-creator.

Anthropic. What are skills? Claude Help Center, 2026d. URL https://support.claude.com/en/
articles/12512176-what-are-skills. Access Date: 2026-03-22.

Udi Barzelay, Ophir Azulai, Inbar Shapira, Idan Friedman, Foad Abo Dahood, Madison Lee, and Abraham Daniels. VAREX: A
benchmark for multi-modal structured extraction from documents, 2026. URL https: //arxiv.org/abs/2603.15118.

Zhoujun Cheng, Haoyu Dong, Zhiruo Wang, Ran Jia, Jiaqi Guo, Yan Gao, Shi Han, Jian-Guang Lou, and Dongmei Zhang. HiTab:
A hierarchical table dataset for question answering and natural language generation. In Proceedings of the 60th Annual
Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 1094–1110, Dublin, Ireland, May

2022. Association for Computational Linguistics. doi: 10. 18653/v1/2022.acl-long.78.
      URL https://aclanthology.org/2022.acl-long.78/.

Runnan Fang, Yuan Liang, Xiaobin Wang, Jialong Wu, Shuofei Qiao, Pengjun Xie, Fei Huang, Huajun Chen, and Ningyu Zhang.
Memp: Exploring agent procedural memory, 2026. URL https://arxiv.org/abs/ 2508.06433.

Google. Gemma 4: Byte for byte, the most capable open models. https://blog.google/
innovation-and-ai/technology/developers-tools/gemma-4/, 2026. Accessed: 2026-05-18.

Tingxu Han, Yi Zhang, Wei Song, Chunrong Fang, Zhenyu Chen, Youcheng Sun, and Lijie Hu. Swe-skills-bench: Do agent
skills actually help in real-world software engineering?, 2026. URL https://arxiv.org/abs/ 2603.15401.

Guanyu Jiang, Zhaochen Su, Xiaoye Qu, and Yi R. Fung. Xskill: Continual learning from experience and skills in
multimodal agents, 2026. URL https://arxiv.org/abs/2603.12056.

Kyudan Jung, Hojun Cho, Jooyeol Yun, Soyoung Yang, Jaehyeok Jang, and Jaegul Choo. Talk to your slides: High-eficiency
slide editing via language-driven structured data manipulation, 2026. URL https: //arxiv.org/abs/2505.11604.

Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E. Gonzalez, Hao Zhang, and Ion
Stoica. Eficient memory management for large language model serving with pagedattention, 2023.
URL https://arxiv.org/abs/2309.06180.

Hao Li, Chunjiang Mu, Jianhao Chen, Siyue Ren, Zhiyao Cui, Yiqun Zhang, Lei Bai, and Shuyue Hu. Organizing,
orchestrating, and benchmarking agent skills at ecosystem scale, 2026a. URL https://arxiv.org/abs 2603.02176.

Jiachun Li, Pengfei Cao, Zhuoran Jin, Yubo Chen, Kang Liu, and Jun Zhao. Mirage: Evaluating and explaining inductive
reasoning process in language models, 2025. URL https://arxiv.org/abs/2410.09542.

Xiangyi Li, Wenbo Chen, Yimin Liu, Shenghan Zheng, Xiaokun Chen, Yifeng He, Yubo Li, Bingran You, Haotian Shen, Jiankai
Sun, Shuyi Wang, Binxu Li, Qunhong Zeng, Di Wang, Xuandong Zhao, Yuanli Wang, Roey Ben Chaim, Zonglin Di, Yipeng Gao,
Junwei He, Yizhuo He, Liqiang Jing, Luyang Kong, Xin Lan, Jiachen Li, Songlin Li, Yijiang Li, Yueqian Lin, Xinyi Liu,
Xuanqing Liu, Haoran Lyu, Ze Ma, Bowei Wang, Runhui Wang, Tianyu Wang, Wengao Ye, Yue Zhang, Hanwen Xing, Yiqi Xue,
Steven Dillmann, and Han chung Lee. Skillsbench: Benchmarking how well agent skills work across diverse tasks, 2026b.
URL https://arxiv.org/abs/2602.12670.

Xiaoxiao Li. When single-agent with skills replace multi-agent systems and when they fail, 2026.
URL https://arxiv.org/abs/2601.04748.

Yu Li, Rui Miao, Zhengling Qi, and Tian Lan. Arise: Agent reasoning with intrinsic skill evolution in hierarchical
reinforcement learning, 2026c. URL https://arxiv.org/abs/2603.16060.

Yuan Liang, Ruobin Zhong, Haoming Xu, Chen Jiang, Yi Zhong, Runnan Fang, Jia-Chen Gu, Shumin Deng, Yunzhi Yao, Mengru
Wang, Shuofei Qiao, Xin Xu, Tongtong Wu, Kun Wang, Yang Liu, Zhen Bi, Jungang Lou, Yuchen Eleanor Jiang, Hangcheng Zhu,
Gang Yu, Haiwen Hong, Longtao Huang, Hui Xue, Chenxi Wang, Yijun Wang, Zifei Shan, Xi Chen, Zhaopeng Tu, Feiyu Xiong,
Xin Xie, Peng Zhang, Zhengke Gui, Lei Liang, Jun Zhou, Chiyu Wu, Jin Shang, Yu Gong, Junyu Lin, Changliang Xu, Hongjie
Deng, Wen Zhang, Keyan Ding, Qiang Zhang, Fei Huang, Ningyu Zhang, Jef Z. Pan, Guilin Qi, Haofen Wang, and Huajun Chen.
Skillnet: Create, evaluate, and connect ai skills, 2026. URL https://arxiv.org/abs/2603.04448.

Brian S. Lin, Jiaxin Yuan, Zihan Zhou, Shouli Wang, Shuo Wang, Cunliang Kong, Qi Shi, Yuxuan Li, Liner Yang, Zhiyuan
Liu, and Maosong Sun. On llm-based scientific inductive reasoning beyond equations, 2025.
URL https://arxiv.org/abs/2509.16226.

Yitao Liu, Chenglei Si, Karthik Narasimhan, and Shunyu Yao. Contextual experience replay for self-improvement of
language agents, 2025. URL https://arxiv.org/abs/2506.06698.

Yuchen Ma, Yue Huang, Han Bao, Haomin Zhuang, Swadheen Shukla, Michel Galley, Xiangliang Zhang, and Stefan Feuerriegel.
Skillgen: Verified inference-time agent skill synthesis, 2026. URL https://arxiv. org/abs/2605.10999.

Zeyao Ma, Bohan Zhang, Jing Zhang, Jifan Yu, Xiaokang Zhang, Xiaohan Zhang, Sijia Luo, Xi Wang, and Jie Tang.
Spreadsheetbench: Towards challenging real world spreadsheet manipulation, 2024. URL https://arxiv.org/abs/2406.14991.

Minesh Mathew, Dimosthenis Karatzas, R Manmatha, and CV Jawahar. Docvqa: A dataset for vqa on document images. corr
abs/2007.00398 (2020). arXiv preprint arXiv:2007.00398, 2020.

Kolby Nottingham, Bodhisattwa Prasad Majumder, Bhavana Dalvi Mishra, Sameer Singh, Peter Clark, and Roy Fox. Skill set
optimization: Reinforcing language model behavior via transferable skills, 2024. URL https://arxiv.org/abs/2402.03244.

Siru Ouyang, Jun Yan, I-Hung Hsu, Yanfei Chen, Ke Jiang, Zifeng Wang, Rujun Han, Long Le, Samira Daruki, Xiangru Tang,
Vishy Tirumalashetty, George Lee, Mahsan Rofouei, Hangfei Lin, Jiawei Han, Chen-Yu Lee, and Tomas Pfister.
Reasoningbank: Scaling agent self-evolving with reasoning memory. In The Fourteenth International Conference on Learning
Representations, 2026. URL https://openreview.net/forum? id=jL7fwchScm.

Panupong Pasupat and Percy Liang. Compositional semantic parsing on semi-structured tables, 2015.
URL https://arxiv.org/abs/1508.00305.

Cheng Qian, Shihao Liang, Yujia Qin, Yining Ye, Xin Cong, Yankai Lin, Yesai Wu, Zhiyuan Liu, and Maosong Sun.
Investigate-consolidate-exploit: A general strategy for inter-task agent self-evolution, 2024.
URL https://arxiv.org/abs/2401.13996.

Noah Shinn, Federico Cassano, Edward Berman, Ashwin Gopinath, Karthik Narasimhan, and Shunyu Yao. Reflexion: Language
agents with verbal reinforcement learning, 2023. URL https://arxiv.org/abs/ 2303.11366.

Qwen Team. Qwen3.5: Accelerating productivity with native multimodal agents, February 2026.
URL https://qwen.ai/blog?id=qwen3.5.

Guanzhi Wang, Yuqi Xie, Yunfan Jiang, Ajay Mandlekar, Chaowei Xiao, Yuke Zhu, Linxi Fan, and Anima Anandkumar. Voyager:
An open-ended embodied agent with large language models, 2023a. URL https: //arxiv.org/abs/2305.16291.

Zilong Wang, Yichao Zhou, Wei Wei, Chen-Yu Lee, and Sandeep Tata. VRDU: A benchmark for visually-rich document
understanding. In Proceedings of the 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, 2023b. doi:
10.1145/3580305.3599929. URL https://arxiv.org/abs/2211.15421.

Zilong Wang, Yuedong Cui, Li Zhong, Zimin Zhang, Da Yin, Bill Yuchen Lin, and Jingbo Shang. OficeBench: Benchmarking
language agents across multiple applications for ofice automation, 2024a. URL https: //arxiv.org/abs/2407.19056.

Zora Zhiruo Wang, Jiayuan Mao, Daniel Fried, and Graham Neubig. Agent workflow memory, 2024b.
URL https://arxiv.org/abs/2409.07429.

Peng Xia, Jianwen Chen, Hanyang Wang, Jiaqi Liu, Kaide Zeng, Yu Wang, Siwei Han, Yiyang Zhou, Xujiang Zhao, Haifeng
Chen, Zeyu Zheng, Cihang Xie, and Huaxiu Yao. Skillrl: Evolving agents via recursive skill-augmented reinforcement
learning, 2026a. URL https://arxiv.org/abs/2602.08234.

Peng Xia, Jianwen Chen, Xinyu Yang, Haoqin Tu, Jiaqi Liu, Kaiwen Xiong, Siwei Han, Shi Qiu, Haonian Ji, Yuyin Zhou, Zeyu
Zheng, Cihang Xie, and Huaxiu Yao. Metaclaw: Just talk – an agent that meta-learns and evolves in the wild, 2026b.
URL https://arxiv.org/abs/2603.17187.

Chenfei Xiong, Jingwei Ni, Yu Fan, Vilém Zouhar, Donya Rooein, Lorena Calvo-Bartolomé, Alexander Hoyle, Zhijing Jin,
Mrinmaya Sachan, Markus Leippold, Dirk Hovy, Mennatallah El-Assady, and Elliott Ash. Co-DETECT: Collaborative discovery
of edge cases in text classification. In Ivan Habernal, Peter Schulam, and Jörg Tiedemann (eds.), Proceedings of the
2025 Conference on Empirical Methods in Natural Language Processing: System Demonstrations, pp. 354–364, Suzhou, China,
November 2025. Association for Computational Linguistics. ISBN 979-8-89176-334-0. doi: 10.18653/v1/2025.emnlp-demos.25.
URL https://aclanthology.org/2025.emnlp-demos.25/.

Yutao Yang, Junsong Li, Qianjun Pan, Bihao Zhan, Yuxuan Cai, Lin Du, Jie Zhou, Kai Chen, Qin Chen, Xin Li, Bo Zhang, and
Liang He. Autoskill: Experience-driven lifelong learning via skill self-evolution, 2026.
URL https://arxiv.org/abs/2603.01145.

Shunyu Yao, Jefrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, and Yuan Cao. React: Synergizing reasoning
and acting in language models, 2023. URL https://arxiv.org/abs/2210. 03629.

Oiving Yu. Zheng Zhang, Ruofei Zhu, Yufeng Yuan, Xiaochen Zuo, Yu Yue, Weinan Dai, Tiantian Fan, Gaohong Liu, Lingjun
Liu, Xin Liu, Haibin Lin, Zhiqi Lin, Bole Ma, Guangming Sheng, Yuxuan Tong, Chi Zhang, Mofan Zhang, Wang Zhang, Hang
Zhu, Jinhua Zhu, Jiaze Chen, Jiangjie Chen, Chengyi Wang, Hongli Yu, Yuxuan Song, Xiangpeng Wei, Hao Zhou, Jingjing Liu,
Wei-Ying Ma, Ya-Qin Zhang, Lin Yan, Mu Qiao, Yonghui Wu, and Mingxuan Wang. Dapo: An open-source llm reinforcement
learning system at scale, 2025. URL https://arxiv.org/abs/2503.14476.

Boyuan Zheng, Michael Y. Fatemi, Xiaolong Jin, Zora Zhiruo Wang, Apurva Gandhi, Yueqi Song, Yu Gu, Jayanth Srinivasa,
Gaowen Liu, Graham Neubig, and Yu Su. Skillweaver: Web agents can self-improve by discovering and honing skills, 2025.
URL https://arxiv.org/abs/2504.07079.

Huichi Zhou, Siyuan Guo, Anjie Liu, Zhongwei Yu, Ziqin Gong, Bowen Zhao, Zhixun Chen, Menglong Zhang, Yihang Chen,
Jinsong Li, Runyu Yang, Qiangbin Liu, Xinlei Yu, Jianmin Zhou, Na Wang, Chunyang Sun, and Jun Wang. Memento-skills: Let
agents design agents, 2026a. URL https://arxiv.org/abs/2603. 18743.

Table 7 | Standard deviations across seeds 41, 42, and 43 for the main spreadsheet results in Table 1. Reference rows
are standard deviations of absolute scores; evolved rows are standard deviations of the paired deltas defined in Table

1. The summary metric Avg is stable (standard deviation ≤ 3<sup>.</sup>4 in every row), and the mean gains in Table 1
   stay large relative to the per-cell spread, so the comparisons are robust. The higher variance in individual
   columns (
   Vrf and the OOD WikiTQ/HiTab transfer) is expected rather than a sign of instability: each score aggregates long
   agentic
   rollouts that often exceed 30 turns of environment interaction, so single-benchmark run-to-run stochasticity
   accumulates
   while averaging out in Avg. For conciseness, we omit additional standard-deviation tables for the paper’s other
   results,
   whose deviations are similarly small: std ranges for spreadsheet tasks are similar to this table. Std ranges of
   DAPO-Test, AIME’26, and DocVQA ANLS are 1.1-2.4, 1.8-3.4, and 0.034-0.139 correspondingly.

<table><tr><td rowspan="3">Condition</td><td colspan="5">Skill User: Qwen3.5-122B-A10B</td><td colspan="5">Skill User: Qwen3.5-35B-A3B</td><td rowspan="3">Avg↑</td></tr><tr><td colspan="3">SpreadsheetBench</td><td colspan="2">OOD</td><td colspan="3">SpreadsheetBench</td><td colspan="2">OOD</td></tr><tr><td>Vrf↑</td><td>Soft↑</td><td>Hard↑</td><td>WikiTQ↑</td><td>HiTab↑</td><td>Vrf↑</td><td>Soft↑</td><td>Hard↑</td><td>WikiTQ↑</td><td>HiTab↑</td></tr><tr><td colspan="12">Baseline (absolute scores)</td></tr><tr><td>No Skill</td><td>3.75</td><td>0.90</td><td>0.70</td><td>0.81</td><td>0.67</td><td>1.80</td><td>2.77</td><td>1.25</td><td>1.27</td><td>5.28</td><td>0.42</td></tr><tr><td>Human-Written</td><td>8.10</td><td>1.39</td><td>0.40</td><td>0.65</td><td>0.71</td><td>2.93</td><td>0.57</td><td>0.40</td><td>0.36</td><td>2.68</td><td>0.46</td></tr><tr><td>Parametric</td><td>0.58</td><td>1.04</td><td>0.75</td><td>6.52</td><td>8.93</td><td>1.44</td><td>1.56</td><td>0.50</td><td>13.32</td><td>18.31</td><td>3.28</td></tr><tr><td colspan="12">Skill Author: Qwen3.5-122B-A10B</td></tr><tr><td colspan="12">Deepening (Delta from init: Human-Written)</td></tr><tr><td>+Error</td><td>4.82</td><td>1.10</td><td>0.78</td><td>0.78</td><td>0.60</td><td>6.76</td><td>3.36</td><td>2.06</td><td>2.62</td><td>4.16</td><td>1.02</td></tr><tr><td>+Success</td><td>12.53</td><td>0.75</td><td>1.35</td><td>1.68</td><td>0.75</td><td>3.69</td><td>3.78</td><td>2.23</td><td>1.43</td><td>4.01</td><td>1.43</td></tr><tr><td>+Combined</td><td>9.53</td><td>1.68</td><td>0.61</td><td>1.22</td><td>0.85</td><td>5.92</td><td>2.92</td><td>1.73</td><td>2.53</td><td>2.74</td><td>1.57</td></tr><tr><td colspan="12">Creation (Delta from init: Parametric)</td></tr><tr><td>+Error</td><td>1.53</td><td>1.10</td><td>0.55</td><td>7.07</td><td>6.98</td><td>3.62</td><td>1.63</td><td>1.22</td><td>13.30</td><td>18.93</td><td>3.24</td></tr><tr><td>+Success</td><td>1.89</td><td>0.91</td><td>1.45</td><td>5.25</td><td>9.60</td><td>4.86</td><td>2.39</td><td>1.76</td><td>14.66</td><td>19.38</td><td>3.00</td></tr><tr><td>+Combined</td><td>3.04</td><td>0.90</td><td>1.10</td><td>6.07</td><td>9.69</td><td>3.04</td><td>1.48</td><td>1.13</td><td>13.52</td><td>19.72</td><td>2.99</td></tr><tr><td colspan="12">Skill Author: Qwen3.5-35B-A3B</td></tr><tr><td colspan="12">Deepening (init: Human-Written)</td></tr><tr><td>+Error</td><td>11.50</td><td>1.31</td><td>2.31</td><td>1.40</td><td>0.63</td><td>5.11</td><td>3.00</td><td>1.12</td><td>0.67</td><td>1.46</td><td>1.10</td></tr><tr><td>+Success</td><td>15.81</td><td>4.63</td><td>2.42</td><td>1.69</td><td>0.84</td><td>0.76</td><td>2.47</td><td>1.16</td><td>1.18</td><td>2.90</td><td>1.32</td></tr><tr><td>+Combined</td><td>9.00</td><td>2.14</td><td>1.33</td><td>1.94</td><td>0.85</td><td>0.50</td><td>3.62</td><td>2.54</td><td>3.27</td><td>2.71</td><td>0.88</td></tr><tr><td colspan="12">Creation (init: Parametric)</td></tr><tr><td>+Error</td><td>2.18</td><td>1.04</td><td>0.58</td><td>6.96</td><td>8.84</td><td>1.76</td><td>1.77</td><td>1.08</td><td>14.60</td><td>17.83</td><td>3.36</td></tr><tr><td>+Success</td><td>1.61</td><td>1.00</td><td>0.90</td><td>6.94</td><td>8.64</td><td>4.51</td><td>1.10</td><td>1.19</td><td>14.24</td><td>19.02</td><td>3.21</td></tr><tr><td>+Combined</td><td>2.75</td><td>1.02</td><td>1.28</td><td>6.37</td><td>8.19</td><td>5.51</td><td>1.62</td><td>1.91</td><td>10.98</td><td>19.64</td><td>3.16</td></tr></table>


Yingli Zhou, Shu Wang, Yaodong Su, Wenchuan Du, Yixiang Fang, and Xuemin Lin. A comprehensive survey on agent skills:
Taxonomy, techniques, and applications, 2026b. URL https://arxiv.org/abs/2605. 07358.

## A. Experimental Details

Random seeds and compute. Unless otherwise noted, all reported results are averaged over three random seeds: 41, 42, and

43. Experiments are run on nodes with 8 NVIDIA A100 GPUs. We spent roughly a total of 20,000 GPU hours for all
    experiments and exploration.

Spreadsheet data and scoring. SpreadsheetBench Verified is split into 200 evolution problems and 200 held-out test
problems; no held-out sample is used during skill evolution. We additionally report Soft, the sub-problem pass rate, and
Hard, where all sub-problems must pass, on full SpreadsheetBench, from which we remove every evolving-set problem so
that no problem seen during skill evolution is scored at test time. For OOD transfer, WikiTableQuestions is converted
into spreadsheet tasks over compositional Wikipedia tables, and HiTab is converted into spreadsheet tasks that require
hierarchical indexing plus implicit calculation and semantic reasoning. The evaluation sets contain 200
SpreadsheetBench-Verified held-out examples, 2,529 full SpreadsheetBench examples for Soft/Hard (all evolving-set
problems removed), 2,810 WikiTableQuestions examples, and 1,585 HiTab examples.

Table 8 | Parallel consolidation versus sequential editing on math reasoning (same trace budget). D-Test/AIME are as in
Table 2 (122B and 35B users).


<table><tr><td rowspan="2">Condition</td><td colspan="2">Runner: 122B</td><td colspan="2">Runner: 35B</td><td rowspan="2">Time↓</td></tr><tr><td>D-Test↑</td><td>AIME↑</td><td>D-Test↑</td><td>AIME↑</td></tr><tr><td>No Skill</td><td>92.0</td><td>90.4</td><td>89.0</td><td>83.3</td><td>0</td></tr><tr><td>Seq-B=4</td><td>93.0</td><td>91.7</td><td>89.0</td><td>68.3</td><td>3.8 min</td></tr><tr><td>Seq-B=1</td><td>94.0</td><td>85.8</td><td>90.0</td><td>81.7</td><td>25.9 min</td></tr><tr><td>Parallel</td><td>95.0</td><td>91.7</td><td>94.0</td><td>88.3</td><td>2.0 min</td></tr></table>


Table 9 | Trace2Skill outperforms ReasoningBank retrieval on math reasoning. D-Test/AIME are as in Table 2.


<table><tr><td rowspan="2">Condition</td><td colspan="2">Runner: 122B</td><td colspan="2">Runner: 35B</td></tr><tr><td>D-Test↑</td><td>AIME↑</td><td>D-Test↑</td><td>AIME↑</td></tr><tr><td>No Skill</td><td>92.0</td><td>90.4</td><td>89.0</td><td>83.3</td></tr><tr><td>ReasoningBank (error-only)</td><td>94.0</td><td>90.8</td><td>91.0</td><td>80.8</td></tr><tr><td>ReasoningBank (combined)</td><td>94.0</td><td>90.4</td><td>91.0</td><td>80.4</td></tr><tr><td>Trace2Skill +Error (ours)</td><td>95.0</td><td>91.7</td><td>94.0</td><td>88.3</td></tr></table>

Baseline skills. Human-Written is Anthropic’s oficial xlsx skill, and Parametric is an xlsx-basic seed generated by
prompting Qwen3.5-122B-A10B from parametric knowledge alone, with no trajectory grounding.

Model serving and evolution protocol. The two main author/user models are Qwen3.5-122B-A10B and Qwen3.5-35B-A3B. We use
the Hugging Face checkpoints Qwen/Qwen3.5-122B-A10B and Qwen/Qwen3.5-35B-A3B. Both are instruct/think hybrid MoE models:
we use instruct mode for multi-turn ReAct-style agents and thinking mode for single-call steps such as hierarchical
merging, success analysis, and patch conversion. Models are served with vLLM (Kwon et al., 2023); for reasoning-mode
calls, we follow Qwen’s oficial recommended decoding settings. Stage 1 generates one trajectory per problem; Stage 2
runs 128 sub-agents in parallel with merge batch size 32. We do not impose a separate tool-call budget.

## B. Additional Analysis Results

## B.1. Latency Analysis

With <sup>??</sup>=128 workers and <sup>??</sup>≈70 error lessons, all analysts execute in a single parallel round. With
merge batch size $B _ { \mathrm { m e r g e } } { = } 3 2$ , the hierarchical merge adds
only $\lceil \log _ { B _ { \mathrm { { m e r g e } } } } N \rceil { \approx } 2$ further sequential rounds, one per
merge layer, yielding ≈3 sequential LLM-call rounds in total. The sequential baselines require <sup>??</sup>
and ⌈<sup>??</sup>/<sup>??</sup>⌉ rounds respectively, since each skill edit depends on the preceding one. In practice
this translates to 3 min for parallel consolidation, 60 min for Seq-<sup>??</sup>=1 (20× slower), and 15 min for
Seq-<sup>??</sup>=4 (5× slower), with the gap scaling linearly in <sup>??</sup>. All times are wall-clock
skill-generation times measured on the same 8-GPU A100 node configuration.

## B.2. Math Reasoning

Tables 8 and 9 extend the main analysis ablations to math reasoning. We use the +Error setting for these math ablations
because it is the strongest math setting in Table 2. Parallel consolidation is best or tied for best on all reported
math metrics while also having the lowest skill-generation wall time.

## B.3. DocVQA

Tables 10 and 11 extend the main DocVQA results to sequential and retrieval-memory comparisons. We use the +Combined
setting for these DocVQA ablations because it is the strongest DocVQA setting in Table 3. Table 10 isolates the 122B
combined-protocol batch ablation, where all non-baseline rows use the same 25 failure and 25 success trajectories and
difer only in sequential batch size versus parallel consolidation. These 50 DocVQA examples are used only for evolution
and are excluded from evaluation; all DocVQA scores are computed on the remaining 5,299 held-out examples. The eficiency
gap between parallel and sequential paradigms will grow with the number of traces as analyzed in § B.1.

Table 10 | Parallel consolidation versus sequential editing on DocVQA using the +Combined trace pool. ANL-S/Acc are as
in Table 3.


<table><tr><td rowspan="2">Setting</td><td colspan="2">Runner: 122B</td><td rowspan="2">Time↓</td></tr><tr><td>ANLS↑</td><td>Acc↑</td></tr><tr><td>No Skill</td><td>0.6300</td><td>70.27</td><td>0</td></tr><tr><td>Seq-B=4</td><td>0.8674</td><td>90.80</td><td>5.2 min</td></tr><tr><td>Seq-B=1</td><td>0.8694</td><td>91.05</td><td>35.9 min</td></tr><tr><td>Parallel</td><td>0.8833</td><td>92.52</td><td>4.8 min</td></tr></table>


Table 11 | Trace2Skill outperforms ReasoningBank retrieval on DocVQA. ANLS/Acc are as in Table 3.


<table><tr><td rowspan="2">Setting</td><td colspan="2">Runner: 122B</td><td colspan="2">Runner: 35B</td></tr><tr><td>ANLS↑</td><td>Acc↑</td><td>ANLS↑</td><td>Acc↑</td></tr><tr><td>No Skill</td><td>0.6300</td><td>70.27</td><td>0.6582</td><td>73.17</td></tr><tr><td>ReasoningBank (combined)</td><td>0.8668</td><td>90.90</td><td>0.8568</td><td>89.62</td></tr><tr><td>Trace2Skill +Combined (ours)</td><td>0.8833</td><td>92.52</td><td>0.8740</td><td>92.00</td></tr></table>

## B.4. Cross-Model Trace Induction

Fig. 6 visualizes this cross-model trace-induction setting. Using Qwen traces, Gemma-4-31B-it and GPT-5.5-high-authored
Deepening skills improve the Qwen3.5-122B user over both No Skill (27.67% Vrf) and Human-Written (48.33% Vrf). The
strongest setting is GPT-5.5-high-authored Deepening +Combined at 68.00% Vrf, showing that cross-model trace evidence
can still induce meaningful portable skills.

Implementation details. We serve Gemma-4-31Bit with vLLM (Kwon et al., 2023) using the oficial recommended generation
configuration from its Hugging Face model card (google/gemma-4-31Bit); reasoning is disabled for the ReAct agent runs.
GPT-5.5-high is accessed through the oficial OpenAI API with only the reasoning efort set to high and all other settings
left at their defaults.

![](images/c43acb203142eeaecbdfdb51cc9184a589180d1651dc8c2c594a4787b782fe9d.jpg)

Figure 6 | Cross-model trace induction on SpreadsheetBench-Vrf (pass rate, %). Gemma-4 and GPT-5.5-high author skills
from Qwen traces, which are then run by the Qwen3.5-122B user.

## C. Qualitative Analyses

## C.1. Agentic vs. LLM Error Analysis

We qualitatively audit 33 shared error cases analyzed by both the agentic error analyst A<sup>−</sup> and the
single-call +Error LLM baseline. The two pipelines reach strong agreement on only 4 cases (12.1%), while 18 cases (
54.5%) show clear disagreement about the root cause or the appropriate skill patch. The main diference is access to
evidence: A<sup>−</sup> can inspect input/output artifacts, compare the submitted answer with ground truth, and validate
candidate fixes, whereas +Error LLM must infer the failure from the execution log alone.

This limitation makes the LLM-only analyzer prone to log-level false positives. Among cases where parseerror messages
appear, +Error LLM attributes the parse error as the primary root cause in 57% of cases, compared with 14% for the
agentic analyst. In one representative trajectory, +Error LLM hallucinated three distinct failure causes even though
artifact evaluation showed the output was already correct. By contrast, the agentic loop can reject such explanations
after checking the generated file and rerunning the relevant validation steps.

These qualitative diferences explain why the agentic patches transfer more reliably in § 4.1. Rather than encoding
surface symptoms from logs, A<sup>−</sup> anchors patches to verified failure mechanisms: wrong target ranges, stale
formula values, corrupted workbook structure, type conversion, or missing read-back verification. The resulting edits
are more likely to become domain-general guardrails and less likely to degrade ID, cross-model,

or OOD settings when applied as a portable skill.

## C.2. Patch Selection Qualitative Analysis

Patch-irrelevant regression. Later iterative patches do change agent behavior on some tasks, flipping them from wrong to
correct as the patch intends. However, the same edits also change behavior that is irrelevant to the targeted failure,
flipping other, previously-correct tasks from correct to wrong. The flips are largely of-source: across the
ten-iteration greedy +Combined path (seed 41), the selected patches’ source tasks have zero overlap with the evaluation
split, so the previously-correct tasks a step breaks are ones the selected patch never targeted: side efects rather than
targeted regressions. Accordingly, every materialized step records substantial flips in both directions (for example, 57
fail-to-pass against 21 pass-to-fail at one step, and 26 against 46 at another), so each addition simultaneously fixes
and breaks tasks. We closely inspect those flips, finding that the majority of fail-to-pass flips are patch-relevant
while pass-to-fail flips are usually irrelevant Because each greedy step both fixes and breaks tasks, the net accuracy
plateaus instead of rising. A locally best single patch therefore does not compose into a global gain: patch value
depends on how additions interact, so optimizing the combinatorial efect of patches rather than their isolated marginal
contributions is necessary.

Semantic overlap. The same failure modes recur across iterations, so the pipeline repeatedly proposes semantically
overlapping patches that target the same spreadsheet behaviors—formula recalculation, workbookstructure preservation,
reference consolidation, and validation checklists. The greedy path makes this concrete: across its ten iterations the
selector keeps returning to a few themes—recalculation and verification (iterations 1 and 6), structure/header/sheet
handling (iterations 2 and 9), and formatting/type/date handling (iterations 4, 5, 7, and 8). A newly selected patch
from an already-covered theme then largely restates guidance the skill already encodes and brings little marginal gain.

## C.3. Learned SoP Details

We inspect the 323 map patches produced by the 122B Deepening +Combined run. The four most prevalent SoPs are cited by
55<sup>.</sup>1%, 54<sup>.</sup>8%, 42<sup>.</sup>7%, and 16<sup>.</sup>4% of the 323 patches, respectively<sup>1</sup>;
these shares sum above 100% because a single patch can cite multiple themes.

Formula recalculation and write-back verification (55.1% of patches). Run recalc.py after every formula write and reopen
with data_only=True to confirm evaluation; skipping this step leaves cells stale and is the single most common error
mode in the run.

Tool selection: openpyxl over pandas.to_excel() (54.8% of patches). Use pandas for read/transform logic and openpyxl for
write-back; copy the input file to the output path first to preserve all structural anchors. pandas.to_excel() silently
destroys formula relationships and named ranges.

Explicit read-back verification (42.7% of patches). After writing, reopen the output file and confirm every target cell
holds the expected value before submitting; error trajectories that fail characteristically omit this check.

Structural-edit safety (16.4% of patches). Delete rows in descending order to prevent index-shift corruption; copy the
input workbook before editing to preserve formatting and formulas. Error trajectories document both failure modes;
success trajectories confirm the protective workflow.

Niche quirks are routed to references/. Low-support observations are not discarded but routed into 13 supplementary
reference files rather than the main SKILL.md. For example, cell color extraction and FIFO vs. LIFO mismatch under
special business logic are placed in on-demand references. This mirrors established skilldesign practice: procedural
guidance flows from general to case-specific, with the main document encoding universal workflow rules and references/
serving as an on-demand look-up layer for infrequent edge cases. Trace2Skill recovers this hierarchy automatically from
trajectory evidence rather than requiring manual curation.

Additional moderate-support SoPs. The following SoPs appear in the same run with moderate support (about

Table 12 | Agentic error analysis (+Error) versus single-call +Error LLM, across Deepening and Creation and all 122B/35B
author–user pairs. Vrf/Soft/Hard, the OOD splits (WikiTQ/HiTab), and Avg are as in Table 1. Bold marks the better of the
two methods per column.


<table><tr><td rowspan="3">Condition</td><td colspan="5">Skill User: Qwen3.5-122B-A10B</td><td colspan="5">Skill User: Qwen3.5-35B-A3B</td><td rowspan="3">Avg</td></tr><tr><td colspan="3">SpreadsheetBench</td><td colspan="2">OOD</td><td colspan="3">SpreadsheetBench</td><td colspan="2">OOD</td></tr><tr><td>Vrf</td><td>Soft</td><td>Hard</td><td>WikiTQ</td><td>HiTab</td><td>Vrf</td><td>Soft</td><td>Hard</td><td>WikiTQ</td><td>HiTab</td></tr><tr><td colspan="12">Skill Author: Qwen3.5-122B-A10B</td></tr><tr><td colspan="12">Deepening</td></tr><tr><td>+Error (ours)</td><td>65.83</td><td>46.60</td><td>27.43</td><td>76.30</td><td>43.45</td><td>36.67</td><td>22.47</td><td>6.23</td><td>18.28</td><td>14.85</td><td>36.21</td></tr><tr><td>+Error LLM</td><td>67.00</td><td>43.93</td><td>25.23</td><td>39.81</td><td>36.68</td><td>25.00</td><td>22.43</td><td>6.23</td><td>11.24</td><td>9.74</td><td>28.00</td></tr><tr><td colspan="12">Creation</td></tr><tr><td>+Error (ours)</td><td>49.00</td><td>40.37</td><td>23.37</td><td>31.62</td><td>15.66</td><td>28.83</td><td>23.23</td><td>7.87</td><td>22.20</td><td>23.58</td><td>26.02</td></tr><tr><td>+Error LLM</td><td>27.17</td><td>27.73</td><td>16.20</td><td>47.26</td><td>30.26</td><td>19.83</td><td>17.60</td><td>4.70</td><td>23.30</td><td>36.53</td><td>26.60</td></tr><tr><td colspan="12">Skill Author: Qwen3.5-35B-A3B</td></tr><tr><td colspan="12">Deepening</td></tr><tr><td>+Error (ours)</td><td>65.00</td><td>44.80</td><td>25.17</td><td>68.32</td><td>38.93</td><td>27.00</td><td>22.20</td><td>8.20</td><td>11.73</td><td>13.38</td><td>32.58</td></tr><tr><td>+Error LLM</td><td>37.83</td><td>22.93</td><td>12.83</td><td>77.05</td><td>42.05</td><td>30.50</td><td>20.17</td><td>8.73</td><td>9.95</td><td>12.73</td><td>28.80</td></tr><tr><td colspan="12">Creation</td></tr><tr><td>+Error (ours)</td><td>27.17</td><td>28.90</td><td>18.53</td><td>81.38</td><td>45.61</td><td>24.00</td><td>21.00</td><td>6.53</td><td>32.80</td><td>31.75</td><td>34.45</td></tr><tr><td>+Error LLM</td><td>22.00</td><td>27.67</td><td>16.60</td><td>54.61</td><td>37.93</td><td>23.50</td><td>16.87</td><td>4.93</td><td>11.24</td><td>33.75</td><td>26.49</td></tr></table>

3.1%–4.6% of patches) and are also encoded in the evolved skill.

Target-range and answer-position validation (4.6% of patches). Before writing, verify the exact target sheet name, cell
range, and answer_position field from the task metadata. Misreading these fields — writing to the wrong sheet or an
of-by-one range — causes silent failures that produce no error message but score zero.

Datatype and datetime preservation (4.6% of patches). Write dates and numeric values as native Python types, not
strings. Both pandas date parsing and openpyxl cell assignment can silently stringify datetime values; inspect each
column’s dtype before writing and use openpyxl’s native datetime assignment.

Workbook structure exploration before editing (success-dominant, ∼4.0% of patches). List all sheets, inspect row/column
layout, and verify header positions before any write. This pre-edit exploration prevents wrong-sheet and wrong-range
failures and accounts for a substantial share of the 151 success-leaning patches in the run.

## D. Full Agentic Error Analysis Results

Table 12 expands the main-text Avg comparison into per-dataset results for both skill authors, both evolution modes, and
both skill-user models. The main pattern is that agentic error analysis is more consistent: +Error (ours) has the higher
Avg in three of the four author–mode blocks and wins most SpreadsheetBench columns. The exception is 122B-authored
Creation, where +Error LLM is slightly higher on Avg because it does better on the OOD table benchmarks; even there, the
agentic analyst remains stronger on the in-distribution SpreadsheetBench metrics. Overall, the full table supports the
main claim that artifact inspection and fix validation produce error patches that transfer more reliably than log-only
analysis.

## E. Selective Patch Aggregation Details

The main experiments apply all learned patches after hierarchical consolidation; selective aggregation instead asks
whether some subset of patches yields a better skill. We reuse the formalization of § 2.1: S is a skill, <sup>??</sup>??
the fixed agent, and $\mathcal { P }$ the pool of trajectory-level patches from Stage 2 (§ 2.3), with $c ( p )$ the
number of source trajectories supporting patch $p \in \mathcal { P }$ . We write $s \oplus \mathcal { P } ^ { \prime }$
for the skill obtained by applying a patch subset $\mathcal { P } ^ { \prime } \subseteq \mathcal { P }$ to S through
the Stage 3 dif application $( \ S ~ 2 . 4 )$ ,
and $\operatorname { a c c } _ { V } ( S ) \triangleq { \mathcal { P } } ( S ; \pi _ { \theta } , V )$ for its
validation pass rate on a 32-task validation set <sup>??</sup>. Both selectors below search for a subset whose
materialized skill maximizes acc??.

Greedy top-1 selection. At iteration <sup>??</sup>, we evaluate the current skill ${ \cal { S } } _ { t } ,$ , collect
the newly proposed patches $\mathcal { P } _ { t } ,$ and form a candidate set $C _ { t }$ of the five highest-coverage
patches with $c ( p ) \geq 5$ . For each $p \in C _ { t }$ , we estimate its contribution by an add-one and remove-one
intervention:

$$
\begin{array}{c} \Delta^ {+} (p) = \mathsf {a c c} _ {V} (\mathcal {S} _ {t} \oplus p) - \mathsf {a c c} _ {V} (\mathcal {S} _ {t}), \\ \Delta^ {-} (p) = \mathsf {a c c} _ {V} (\mathcal {S} _ {t} \oplus \mathcal {P} _ {t}) \\ \qquad - \mathsf {a c c} _ {V} \bigl (\mathcal {S} _ {t} \oplus (\mathcal {P} _ {t} \setminus \{p \}) \bigr). \end{array}
$$

Patches with $\Delta ^ { + } ( p ) < 0$ or $\Delta ^ { - } ( p ) < 0$ are discarded; the remainder are ranked
by $r ( p ) = \Delta ^ { + } ( p ) + \Delta ^ { - } ( p )$ , with coverage as the tie-breaker, and the next skill
is $S _ { t + 1 } = S _ { t } \oplus p _ { t } ^ { \star }$ for the top-ranked
patch $p _ { t } ^ { \star } . \ \ S \ 4 . 3$ reports the SpreadsheetBench-Vrf results.

Bayesian optimization over patch subsets. BO fixes a patch
universe $\mathcal { P } = \{ p _ { j } \} _ { j = 1 } ^ { m }$ from the Stage-2 pool $( \ S \ 2 . 3 )$ , keeping the
top $m \le 1 5$ patches by coverage. It searches binary inclusion vectors $x \in \{ 0 , 1 \} ^ { m }$ ,
writing $\mathcal { P } _ { x } = \{ p _ { j } : x _ { j } = 1 \}$ for the selected subset and starting from the initial
skill $\scriptstyle { S _ { 0 } }$ . The objective is

$$
f (x) = \operatorname{acc} _ {V} (\mathcal {S} _ {0} \oplus \mathcal {P} _ {x}) - \lambda \| x \| _ {0}, \quad \lambda = 0.
$$

The initial design evaluates the empty subset, all singletons, five random mixed subsets, and the full subset. After
each batch, evaluated subsets are split into a good set <sup>??</sup> containing the top $\gamma = 0 . 2$ fraction
by $f ( x )$ and a bad set <sup>??</sup> containing the rest. With Beta smoothing $\alpha = 1$ , the per-patch Bernoulli
inclusion rates are

$$
\mu_ {j} ^ {G} = \frac {\sum_ {x \in G} x _ {j} + \alpha}{| G | + 2 \alpha}, \qquad \mu_ {j} ^ {B} = \frac {\sum_ {x \in B} x _ {j} + \alpha}{| B | + 2 \alpha}.
$$

Candidate subsets are sampled from $\phi _ { j } = ( 1 - \rho ) \mu _ { j } ^ { G } + \rho / 2$ with $\rho = 0 . 1$ and
ranked by the TPE acquisition

$$
A (x) = \sum_ {j = 1} ^ {m} \left[ \log q (x _ {j}; \mu_ {j} ^ {G}) - \log q (x _ {j}; \mu_ {j} ^ {B}) \right], \qquad q (b; \mu) = \mu^ {b} (1 - \mu) ^ {1 - b}.
$$

We evaluate the top eight candidates per round from a pool of 500, stop after four rounds or three rounds without
improvement, and return the best observed subset. § 4.3 compares the BO-selected +Error skill against applying all
+Error patches.

## F. Broader Application Details

We extend Trace2Skill to three broader document-agent settings: PDF extraction, PPTX editing, and DOCX editing. All
settings use Anthropic’s oficial document skills as the underlying skill family (Anthropic, 2026a). Source traces are
used only for skill evolution, while held-out target tasks are used only for evaluation. Each evaluation uses the task’s
exact local verifier and reports the same task-level pass-rate metric as the main experiments. Table 13 summarizes the
transfer setting used for each modality in the main text; “Base” denotes the immediate skill before the listed source
traces are applied.

PDF extraction. The PDF study adapts VRDU and VAREX into local PDF-to-JSON tasks (Wang et al., 2023b; Barzelay et al.,
2026). We use VRDU Registration Forms as the source corpus because it provides a coherent real collection of visually
rich form-extraction traces with recurring field and layout conventions. We evaluate on VAREX Flat as a separate
held-out structured-extraction benchmark, testing whether those induced formextraction procedures transfer across
datasets rather than to more examples from the same corpus. The reported PDF score uses the strongest validated VRDU
Registration source split; other Registration splits are also positive but yield smaller gains.

<table><tr><td>Domain</td><td>Source traces</td><td>Held-out evaluation</td><td>Base</td><td>Trace2Skill</td><td>Gain</td></tr><tr><td>PDF</td><td>VRDU Registration Forms</td><td>VAREX Flat validation</td><td>76.9%</td><td>85.3%</td><td>+8.4 pp</td></tr><tr><td>PPTX</td><td>TSBench training traces</td><td>TSBench deck-disjoint OOD</td><td>72.5%</td><td>88.8%</td><td>+16.3 pp</td></tr><tr><td>DOCX</td><td>Generated document-operation tasks</td><td>OfficeBench DOCX holdout</td><td>79.7%</td><td>87.5%</td><td>+7.8 pp</td></tr></table>


Table 13 | Broader document-agent transfer. Each row evolves an oficial Anthropic document skill from source traces and
evaluates on a held-out target domain. Base and Trace2Skill are target pass rates (%); Gain is their diference in
percentage points (pp).

PPTX editing. The PPTX study uses TSBench presentation-editing tasks from Talk-to-Your-Slides (Jung et al., 2026). The
PPTX result evolves from TSBench training traces on one set of decks and evaluates on a deckdisjoint TSBench OOD split
of held-out decks, testing transfer across presentation files rather than more edits from the same decks.

DOCX editing. The DOCX study uses OficeBench as the held-out target because it contains realistic oficeautomation tasks
with verifiable Word/DOCX outputs (Wang et al., 2024a). We reserve all 64 convertible OficeBench DOCX subtasks only for
evaluation at max_turns=100. Because the remaining real, verifiable DOCX agent data is too small after reserving this
target set, we use generated ofice-document traces as the source distribution. These traces cover reusable operations
such as editing templates, preserving document structure, appending content across files, and producing required sidecar
files.

## G. Head-to-Head Comparison with Concurrent Skill-Evolution Systems

This appendix is the head-to-head complement to the apples-to-apples comparisons in § 4.1: it pits Trace2Skill against
three full concurrent skill-evolution systems—XSkill (Jiang et al., 2026), EvoSkill (Alzubi et al., 2026), and
SkillGen (Ma et al., 2026)—rather than isolating one design choice at a time. We keep it out of the main text for three
reasons. Attribution: a whole-system score conflates a design idea with its base model, harness, and engineering, so it
cannot say which factor drove a diference; § 4.1 instead varies a single design choice under a shared trace pool, model,
and harness, which is what licenses our causal claims. Scope: these systems pursue diferent goals and were built around
specific (often proprietary) base models, bespoke harnesses, and diferent task domains, whereas we deliberately study
open models that self-evolve; transplanting them onto one benchmark shows behavior in our setting, not a ceiling on
their design, so we read the numbers conservatively.

We reproduce these pipelines following their oficial codebase adapted to one shared protocol: the same open base model
Qwen3.5-122B-A10B served with vLLM, evolution on the SpreadsheetBench-Verified 0:200 slice, and evaluation on the
held-out 200:400 slice scored by oficial instance accuracy (our Vrf metric). Fig. 7 reports the result against the No
Skill floor, the Human-Written skill that several systems also start from, and Trace2Skill Deepening +Error/+Combined.
Trace2Skill attains the highest Vrf (65<sup>.</sup>83 +Error, 69<sup>.</sup>83 +Combined), above all three systems; the
per-system settings and the deviations from each native pipeline are detailed below.

XSkill. XSkill is a non-parametric memory method that accumulates task-level procedure skills and action-level
experiences, then retrieves, adapts, and injects them at inference; it was designed for multimodal tool-using agents. We
collect 200 trajectories on the 0:200 slice (same evolving set as Trace2Skill), merge their pertrajectory skills into
one 865-word global SKILL.md (batched LLM merge), and distill 20 curated action-level experiences. At test time, for
each instance we retrieve the top-3 experiences by lexical overlap, have the LLM adapt the global skill to the task,
inject it into the system prompt, and run the agent for up to 100 turns. This reproduction reaches 23<sup>.</sup>0 Vrf.
We replaced the lexical retriever with Qwen3-Embedding-0.6B. However, this achieves a slightly worse performance at
20<sup>.</sup>0.

EvoSkill. EvoSkill is a self-evolving loop—base agent, failure proposer, skill generator, validation evaluator, and a
bounded frontier—designed for the Claude Code harness with Opus 4.5. We mine failures on 0:160, validate candidates on
160:200, and evaluate the selected program on 200:400 (our SpreadsheetBench Vrf test subset), with a frontier of size 3;
each iteration proposes and generates a replacement xlsx/SKILL.md and scores it on the validation slice, with
Qwen3.5-122B-A10B served by vLLM. We run two harnesses: a React-style local harness matching our setting reaches
59<sup>.</sup>5 Vrf (5 iterations), while Claude Code accessed through an Anthropic-compatible proxy (closest to the
native harness) reaches 33<sup>.</sup>5 (8 workers, early-stopped near iteration 11).

![](images/d46247f023a33e7c4e9fe54ee6e54583dc98843f82ff54d38a91cf2454c21558.jpg)

Figure 7 | Head-to-head comparison on SpreadsheetBench-Verified (Vrf, pass rate %) with all systems using the same open
base model (Qwen3.5-122B-A10B) and the held-out 200:400 test slice. Bars cover the No Skill floor, the Human-Written
starting skill (dashed line), the three reproduced concurrent systems, and Trace2Skill Deepening. EvoSkill’s two bars
are the same faithful reproduction difering only in agent harness (Claude Code, following original paper, vs. React,
matching our setting).

SkillGen. SkillGen induces a compact skill from contrastive failure/success analysis behind a paired verification gate
before held-out evaluation. We collect baseline trajectories on 0:200, cluster failure and success summaries with
Qwen3-Embedding-0.6B (<sup>??</sup>-means) to synthesize contrastive observations, generate a compact SKILL.md, and
verify it on 0:200 behind a net-gain acceptance gate (we use min_net_gain_abs=1; the upstream default is ≥ 3) for up to
8 refinement rounds, evaluating the accepted skill on 200:400. From a 52<sup>.</sup>5 baseline on the training slice,
round 1 nets 0 (rejected), round 2 nets −4 (rejected), and round 3 nets +1 (accepted), giving 27<sup>.</sup>5 Vrf on the
held-out slice. This improvement on training set but regression on test set might be attributed to the distribution
shift: the training slice mixes 125 sheet-level and 75 cell-level tasks while the test slice is cell-only.

These head-to-head comparisons complement, and do not replace, the confounder-free apples-to-apples comparisons in §
4.1, on which our claims rest.

## H. Skill-Creator Baseline

For the external Vrf-only baseline, we used Anthropic’s oficial skill-creator skill for skill drafting and improvement (
Anthropic, 2026c) through Claude Code with Opus 4.6 medium. Table 14 reports SpreadsheetBench-

<table><tr><td rowspan="2">User/Trace source</td><td colspan="3">Deepening</td><td colspan="3">Creation</td></tr><tr><td>Base</td><td>skill-creator</td><td>Trace2Skill +Error</td><td>Base</td><td>skill-creator</td><td>Trace2Skill +Error</td></tr><tr><td>122B/122B</td><td>48.33</td><td>27.33</td><td>65.83</td><td>26.17</td><td>26.67</td><td>49.00</td></tr><tr><td>122B/35B</td><td>48.33</td><td>19.50</td><td>65.00</td><td>26.17</td><td>18.33</td><td>27.17</td></tr><tr><td>35B/35B</td><td>9.67</td><td>18.50</td><td>27.00</td><td>20.17</td><td>17.67</td><td>24.00</td></tr><tr><td>35B/122B</td><td>9.67</td><td>28.00</td><td>36.67</td><td>20.17</td><td>27.33</td><td>28.83</td></tr><tr><td>Average</td><td>29.00</td><td>23.33</td><td>48.63</td><td>23.17</td><td>22.50</td><td>32.25</td></tr></table>


Table 14 | SpreadsheetBench-Verified (Vrf) comparison with Anthropic’s skill-creator baseline (pass rate, %). For each
skill-user/trace-source pair, skill-creator is compared with the corresponding Base skill and Trace2Skill +Error under
Deepening and Creation; the final row averages each column.

Verified Vrf scores for the same skill-user/trace-source pairings used in the main spreadsheet evaluation, alongside the
corresponding Table 1 baselines and Trace2Skill +Error results. The resulting skills do not improve the corresponding
base skill on average in either Deepening or Creation, while Trace2Skill +Error remains consistently stronger.

The prompts used for this baseline were:

```txt
Skill Creator Baseline Prompt: Deepening

I have an agent running spreadsheet jobs using an xlsx skill. The agent's traces are provided in /path/to/ctraces/, where error and success traces are annotated with *_FAILURE.md and *_SUCCESS.md. Your job is to deepen the xlsx (/path/to/xlsx) to improve the agent's future performance when using the new skill. You should first induce the common and generalizable failure and success patterns from the traces and patch the xlsx skill using the skill-creator skill. 
```

```txt
Skill Creator Baseline Prompt: Creation
I have an agent running spreadsheet jobs. The agent's traces are provided in /path/to/traces/, where error and success traces are annotated with *_FAILURE.md and *_SUCCESS.md. Your job is to create a spreadsheet skill to improve the agent's future performance equipped with the skill. You should first induce the common and generalizable failure and success patterns from the traces and then create the skill using the skill-creator skill. 
```

## I. Prompt Templates and Intermediate Outputs

This appendix reproduces the key prompt templates used in each pipeline stage and illustrates representative
intermediate outputs to make the pipeline fully transparent and reproducible.

## I.1. Stage 1: Agent System Prompt Template

The agent <sup>??</sup>?? operates under the following system prompt during trajectory collection. The skill S is
prepended to the user context at inference time. Note that this difers from the standard skill-loading process, where
the agent initially has access only to skill descriptions. We simplify this by preloading the SKILL.md content into the
system prompt because Trace2Skill focuses on improving a fixed target skill that is known relative to the task.
Therefore, there is no need for the standard skill-selection step. Importantly, the Trace2Skill skill-using agent still
needs to procedurally discover resources referenced by the preloaded SKILL.md (e.g., resources and scripts), which are
not preloaded.

Stage 1 — Agent System Prompt (abbreviated)

Role: You are an expert role (e.g., spreadsheet analysis) agent.

Skill context: [Contents of $S_{0}$ inserted here]

Task: Describing tasks and input files.

Tools available: Describing tools and environment. E.g., bash (shell execution) for ReAct with file system access.

Format: ReAct-style interaction — alternate between reasoning traces and tool calls until the task is complete.

## I.2. Stage 2: Analyst Prompt Templates and Example Patches

In Stage 2, the patch proposing agents first draw error and success memory items similar to (Ouyang et al., 2026), which
are generalizable trajectory-level knowledge that might be helpful for future task executions. Next, the agents read the
original skill directory and then propose a patch to encode the memory items into the skill.

## I.2.1. Error Analyst Prompt (A<sup>−</sup>)

Error Analyst System Prompt (abbreviated)

Role: You are an expert failure-analysis agent for {domain} tasks.

Mission: Given an agent's execution artifacts (logs + produced files) and the ground-truth solution, diagnose why the
agent failed, identify causal failure reasons, and validate your diagnosis by implementing a minimal fix that makes the
agent output match the ground truth. Your analysis must be systematic, evidence-driven, and reproducible. Do not guess
when you can verify.

Required Workflow (MANDATORY):

1. Understand the task and failure surface — identify exactly what is wrong in the output.
2. Trace the failure to agent behavior — locate the decision or code step that produced the mismatch.
3. Validate the root cause with a minimal fix — write fixed output and re-evaluate against the ground truth.
4. Re-evaluate — if still failing, return to steps 1–3 and revise your diagnosis.

Output: Produce (1) Failure Cause Items — systematic, causal reasons grounded in observable agent behavior; (2) Failure
Memory Items ( $\leq$ 3) — generalizable insights the agent should remember to avoid similar failures.

## I.2.2. Success Analyst Prompt (A<sup>+</sup>)

```txt
Success Analyst System Prompt (abbreviated)

Role: You are an expert in success pattern analysis for AI agent systems.

Mission: Given a successful agent trajectory, identify generalizable behavior patterns that contributed to the correct answer.

Requirements:

Broad Coverage — every effective behavior in the trajectory must be captured by a pattern.

Frequency Awareness — patterns covering more instances should be listed first; rare behaviors should be absorbed into the nearest broader pattern.

Generalization — each pattern must describe a general mechanism, not a single task-specific detail.

Output: A compact set of Success Memory Items with title, description, and concrete examples of the effective behaviors observed. 
```

## I.2.3. Example Trajectory-Level Patch (Error Analyst Output)

Each analyst produces a structured patch for a single trajectory. The following example is from an error analyst (
A<sup>−</sup>) processing one failure trajectory where the agent deleted rows outside the specified range. The patch
proposes adding a new SKILL.md section on row-manipulation safety; this patch, along with 52 others citing related
structural-edit failures, is later consolidated into the “Structural-edit safety” SoP (§ 4.2).

```diff
Example Stage 2 Trajectory-Level Patch (error analyst, one trajectory)

Analyst reasoning: Single error record reveals a clear failure pattern: the agent deletes rows outside the specified range and fails to distinguish partially-empty data rows from completely empty padding rows. The skill lacks any guidance on row/column manipulation or range boundary enforcement.

--- a/SKILL.md
+++ b/SKILL.md
@@ -55,0 +55,26 @@ ## Formula Verification Checklist

+## Row and Column Operations
+
+### Range Boundary Enforcement
+- When a task specifies a cell range (e.g., F14:M83), operations
+ MUST stay within those exact boundaries
+- Parse range format: start_col + start_row : end_col + end_row
+- Never apply operations to rows/columns outside the range
+
+### Row Deletion Logic
+Before deleting any row, verify BOTH conditions:
+1. Target columns are empty (as specified by task)
+2. Row falls within the explicit deletion range
+
+**Do NOT delete** completely empty rows across ALL columns unless
+explicitly instructed -- these often serve as structural padding.
+
+### Post-Operation Validation
+After row/column operations:
+- [ ] Count remaining rows matches expected count
+- [ ] No header/footer/padding rows outside range were affected
+- [ ] Data integrity is maintained in adjacent columns
+- [ ] Range boundaries were respected (no off-by-one errors) 
```

## I.3. Stage 3: Merge Operator Prompt and Example Consolidated Patch

## I.3.1. Merge Operator Prompt (M)

![](images/de8a769235c13270bd6c3ef7292282d1a9993a56e25f2679ea81683aaec783d8.jpg)

## I.3.2. Example Final Consolidated Patch <sup>??∗</sup> (After Full Merge Hierarchy)

The following excerpt shows the reasoning and representative edits from the final consolidated patch <sup>??∗</sup>
produced after four levels of hierarchical merging over 323 individual trajectory patches on SpreadsheetBench-Verified.

```txt
Example Stage 3 Final Patch Output (excerpt)
{
    "reasoning": "Merged 3 patches addressing mixed failure/success evidence. Key consolidation decisions: (1) Synthesized recalc.py workflow from all patches using the most prominent CRITICAL WARNING placement, CSV fallback validation, and a verification loop;
    (2) Consolidated library selection guidance into a comprehensive decision tree; (3) Combined row deletion guidance emphasizing bottom-up/right-to-left deletion order from all patches;
    (4) Merged formula validation checklists without redundancy.",
    "edits": [
    {
    "file": "SKILL.md",
    "op": "insert_after",
    "target_section": "# Requirements for Outputs",
    "content": "## Important Automation Guidelines\n\n
    **Prefer Python over VBA for Automation**: When tasks request VBA macros or spreadsheet automation, implement the logic in Python using openpyxl/pandas instead. This provides better error handling, easier debugging, cross-platform compatibility, and avoids macro security issues."
    },
    {
    "file": "SKILL.md",
    "op": "insert_after",
    "target_section": "## Important Requirements",
    "content": "### When to Compute Directly in Python\n
    If LibreOffice/recalc.py cannot evaluate certain formula patterns (complex array formulas, advanced INDEX/MATCH combinations), compute the logic directly in Python and write results as values. Use this fallback when formula returns errors after recalculation despite correct syntax."
    }
    ],
    "changelog_entries": [
    "Added automation guidelines preferring Python over VBA",
    "Consolidated recalc.py warnings into a single CRITICAL WARNING"
    ]
} 
```

I.3.3. Example Translated Dif: Final Consolidated Patch <sup>??</sup>

The JSON patch <sup>??∗</sup> produced by Stage 3 is translated programmatically into a unified dif and applied directly
to the skill directory. The following excerpt shows two hunks from the final applied dif on SpreadsheetBench-Verified,
corresponding to the two most prevalent SoPs: formula recalculation enforcement (55.1% of patches) and tool-selection
guidance (54.8% of patches).

```diff
Example Stage 3 Applied Diff (excerpt from final p*)
--- a/SKILL.md
+++ b/SKILL.md
@@ -126,3 +261,18 @@ ## Common Workflow

## Common Workflow

+### CRITICAL WARNING: Formula Recalculation Is Mandatory
+
+**If you write ANY formulas to an Excel file using openpyxl,
+you MUST run recalc.py before considering the task complete.**
+
+Formulas written via openpyxl exist only as text strings until
+recalculated. Without running recalc.py:
+- Cells return None/empty when read with data_only=True
+- Evaluation fails even if formulas are syntactically correct
+- The output file is incomplete
+
+This is non-negotiable. Do not proceed to verification or
+delivery until recalc.py confirms success.
+
@@ -138,3 +285,9 @@ ### Working with openpyxl

+### Tool Selection Warning
+
+**CRITICAL**: When modifying spreadsheets that contain existing 
```

+formulas you need to preserve: +- Use openpyxl (load_workbook() then save()) -- formulas remain + as strings +- Avoid
pandas (to_excel()) -- silently converts formulas to + static values permanently 