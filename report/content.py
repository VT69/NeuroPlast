"""Content of the DSN4091 capstone report (rendered by report/build_report.py).

Sources: the paper (paper/main.tex), PROGRESS.md, docs/LITERATURE_CONTEXT.md, the deck and the dashboard. Every result
number goes through n("<macro>") (the paper's numbers pipeline); other numbers (hardware, software versions, the demo
accuracy matrices) are registered with other(...) and listed with their sources in report/NUMBERS_CHECK.md.

Markup inside text: **bold**, *italic*, `code`, [[bibkey]] (numbered citation), <<fig:label>> / <<tab:label>>.
"""
from __future__ import annotations

import importlib.metadata as md
import json
import platform
import re
import subprocess

from PIL import Image

OTHER: list[tuple[str, str, str]] = []
PF = "paper/figures"
RF = "report/figures"


def other(what, val, src):
    OTHER.append((what, val, src))
    return val


def fitw(path, maxw=6.25, maxh=7.6):
    w, h = Image.open(path).size
    return min(maxw, maxh * w / h)


# ---------------------------------------------------------------- abstract (about 250 words, for a non-specialist)
def ABSTRACT(n):
    return (
        "A learning system that is trained on one task after another tends to lose what it learned earlier: this is "
        "called catastrophic forgetting. Many proposed remedies borrow ideas from the brain, such as spiking neurons, "
        "local learning rules, homeostasis and a sleep-like offline phase, but they are usually added several at a "
        "time, so it is hard to tell which idea does the work. This project tests these ideas one at a time. We "
        "built a small learning agent for grid-world tasks in which each mechanism can be switched on or off, and "
        "compared each one with a matched control that is not brain-inspired but uses the same resources. The study "
        f"ran on a 4-core CPU without a GPU ({n('computeRuns')} training runs, about {n('computeCoreHours')} CPU "
        "hours). Each final comparison was specified in a version-controlled experiment log before its runs, with "
        f"{n('seedRange')} random seeds per arm and 95% confidence intervals. No mechanism produced a statistically "
        "reliable improvement in final accuracy or forgetting beyond its matched control; with these seed counts, "
        "this rules out large effects only where the intervals are narrow, and it never shows that two methods are "
        "equivalent. One narrow effect survived: homeostasis made a spiking network learn faster when its encoder "
        "learned without backpropagation (the output layers still used exact gradients). Two results are conditional: a shared network with a small memory "
        "buffer needed less memory than one network per task, but only against full-width per-task networks, and the "
        "spiking encoder needed fewer operations in an energy estimate while costing "
        f"{n('snnCpuSlowdown')} times more CPU time to train in our implementation. Whether Transformer working "
        "memory helps remains inconclusive. We also report seven earlier positive findings that shrank or disappeared "
        "once the controls were fair.")


# ---------------------------------------------------------------- helpers for tables of numbers
def mean(v):
    return v.split(" ±")[0]


def blocks(n, snippet):
    B = []
    p = lambda t, **k: B.append(("p", t, k) if k else ("p", t))  # noqa: E731
    ul = lambda items: B.append(("ul", items))  # noqa: E731
    ol = lambda items: B.append(("ol", items))  # noqa: E731
    fig = lambda lab, path, cap, w=6.2: B.append(("fig", lab, path, cap, w))  # noqa: E731
    tab = lambda lab, cap, head, rows, widths, **k: B.append(("tab", lab, cap, head, rows, widths, k))  # noqa: E731
    code = lambda cap, text, **k: B.append(("code", cap, text, k))  # noqa: E731
    h2 = lambda num, t: B.append(("h2", num, t))  # noqa: E731
    h3 = lambda num, t: B.append(("h3", num, t))  # noqa: E731

    # =========================================================================================== CHAPTER 1
    B.append(("chapter", 1, "Project Description and Outline"))
    h2("1.1", "Introduction")
    p("Reinforcement learning (RL) is a form of machine learning in which an *agent* learns by trial and error: it "
      "observes its environment, takes an action, receives a numerical *reward*, and gradually adjusts its *policy* "
      "(the rule that maps observations to actions) so that it collects more reward. In *continual* reinforcement "
      "learning the agent meets a sequence of tasks, one after another, and is expected to keep the skills it "
      "acquired on earlier tasks while it learns new ones [[khetarpal2022towards]].")
    p("With plain gradient-based training, this rarely works. The network's parameters are shared by all tasks, so "
      "the updates that teach the new task also overwrite what the earlier tasks needed. Performance on old tasks "
      "then collapses, often within a few thousand updates. This failure is called *catastrophic forgetting*, and it "
      "is usually described as a trade-off between stability (keeping old knowledge) and plasticity (acquiring new "
      "knowledge) [[khetarpal2022towards, kirkpatrick2017overcoming]]. The standard remedies fall into three "
      "families: regularisation, which penalises changes to parameters that mattered for earlier tasks "
      "[[kirkpatrick2017overcoming]]; replay, which mixes stored experience from earlier tasks into training "
      "[[rolnick2019experience]]; and capacity isolation, which gives each task its own parameters "
      "[[mallya2018packnet]].")
    p("A second family of remedies is inspired by the brain. Spiking neural networks (SNNs), in which artificial "
      "neurons communicate through discrete spikes instead of continuous values, promise event-driven, low-energy "
      "computation [[fang2023spikingjelly, zhou2026spikingformer]]. Local learning rules such as spike-timing-"
      "dependent plasticity (STDP), which strengthens or weakens a connection depending on the relative timing of "
      "the spikes on either side of it, are proposed as biologically plausible complements to backpropagation "
      "[[lee2025brain, huo2025research]]. A sleep-like offline phase has been reported to recover forgotten tasks "
      "[[tadros2022sleep]], and homeostatic plasticity, the brain's way of keeping firing rates in a working range, "
      "is often added to stabilise such systems [[turrigiano2004homeostatic]].")
    p("This report describes NeuroPlast, a capstone project that tests these brain-inspired mechanisms one at a "
      "time. We built a single agent for MiniGrid grid-world tasks [[chevalierboisvert2023minigrid]], trained with "
      "the PPO algorithm [[schulman2017proximal]], in which each mechanism can be switched on or off, and we "
      "compared each mechanism with a matched control that is not brain-inspired. The setting is *task-incremental*: "
      "the agent is told which task it is currently doing. The rest of this chapter explains the motivation, states "
      "the problem and lists the objectives. Chapter 2 reviews related work, Chapter 3 lists the requirements, "
      "Chapter 4 presents the design, Chapter 5 the implementation, Chapter 6 the results and Chapter 7 the "
      "conclusions and future work.")

    h2("1.2", "Motivation for the Work")
    p("Brain-inspired mechanisms for continual learning are popular, but they are usually introduced several at a "
      "time, in one system, and compared with an unmodified baseline. When such a system performs better, two "
      "questions remain open. First, which of the mechanisms did the work? Second, would a simpler non-biological "
      "control, given the same resources, have done just as well? Without answers to both, a reported improvement "
      "cannot be attributed to the brain-inspired idea itself.")
    p("Resources are a particular source of confusion. A sleep phase that replays stored experience may look better "
      "than ordinary replay simply because it replays more samples. A shared network may look more efficient than one "
      "network per task if only parameters are counted and the replay buffer it needs is ignored. A small number of "
      "random seeds can make a chance difference look significant. Each of these confounds affected our own early "
      "results, and correcting them changed the conclusions (Section 6.2.5).")
    p("The question also matters in practice. Agents that keep learning after deployment, such as robots or "
      "embedded devices, work under tight memory and compute budgets. If a brain-inspired mechanism helps, it should "
      "help per unit of memory and compute, not only in accuracy. Finally, negative results are useful: knowing which "
      "popular mechanisms do not help in a controlled setting saves effort for others, provided the test was fair.")

    h2("1.3", "Problem Statement")
    p("**In a controlled task-incremental continual RL setting, which brain-inspired mechanisms improve learning or "
      "retention when each is isolated and compared with a matched control?**")
    p("An improvement counts only if it exceeds the matched control by more than seed-to-seed variation, in a test "
      "whose comparison, statistic, seed count and stopping rule were fixed before the runs. The controls are chosen "
      "to rule out the most obvious alternative explanation: a random update of the same size for STDP, replay with "
      "an exactly matched number of replayed samples for sleep, and total-memory accounting (parameters plus replay "
      "buffer) for the comparison between a shared network and one network per task.")

    h2("1.4", "Objective of the Work")
    p("The objectives of the project were:")
    ol(["To test whether a **spiking encoder** reaches a better accuracy-versus-operations trade-off than a "
        "conventional convolutional encoder of the same topology that has been sparsified.",
        "To test whether reward-modulated **STDP** improves learning beyond backpropagation alone, beyond "
        "homeostasis alone and beyond a random update of the same size.",
        "To test whether **homeostasis** helps, with ordinary backpropagation and with direct feedback alignment.",
        "To test **direct feedback alignment (DFA)**, a form of backprop-free credit assignment for the encoder, "
        "with and without homeostasis.",
        "To test whether **sleep-like consolidation** reduces forgetting beyond experience replay and elastic weight "
        "consolidation, at a matched replay budget.",
        "To test whether a **shared network trunk with per-task heads** retains more performance per unit of total "
        "memory than one network per task (parameter isolation).",
        "To test whether **Transformer working memory** helps on a task that actually requires memory, compared with "
        "a memoryless agent and a stack of recent frames.",
        "To keep every comparison **fair**: the same frames, seeds and learning rate in every arm, tests specified "
        "before the runs, and multiple-comparison correction.",
        "To count **memory** as parameters plus everything stored, not parameters alone.",
        "To make the work **reproducible**: every number in the results must be regenerated by a script from the "
        "frozen run files, and the trained agents must be inspectable through an offline dashboard."])

    h2("1.5", "Summary")
    p("Continual RL suffers from catastrophic forgetting. Brain-inspired remedies are popular but are usually tested "
      "together and without matched controls, which leaves their individual contribution unknown. This project "
      "isolates six brain-inspired mechanisms (the spiking encoder, STDP, homeostasis, DFA, sleep-like consolidation "
      "and a shared trunk versus isolation) and attempts a seventh (Transformer working memory), each against a "
      "matched non-biological control, with resource budgets matched and tests specified in advance. The next "
      "chapter places this work in the context of existing approaches.")

    # =========================================================================================== CHAPTER 2
    B.append(("chapter", 2, "Related Work Investigation"))
    h2("2.1", "Existing Approaches/Methods")
    p("This section groups the existing approaches into six families. The first three are the standard strategies "
      "against forgetting; the last three are the brain-inspired ideas that NeuroPlast tests.")
    p("**Regularisation.** Elastic weight consolidation (EWC) estimates how important each parameter was for "
      "earlier tasks, using the Fisher information, and penalises moving important parameters while a new task is "
      "learned [[kirkpatrick2017overcoming]]. It protects old tasks at the cost of plasticity for new ones. Learning "
      "without Forgetting (LwF) regularises the network's *outputs* instead of its parameters: before a new task, it "
      "keeps a copy of the network and distils that copy's outputs on new-task inputs, so it needs no stored data "
      "[[li2017learning]].")
    p("**Replay.** CLEAR mixes new experience (for plasticity) with replayed experience from earlier tasks (for "
      "stability) and adds behavioural cloning on the replayed experience: a term that keeps the current policy "
      "close to the stored policy, and one that keeps the current value estimate close to the stored value "
      "[[rolnick2019experience]]. It is simple, needs no task identities and is a very strong baseline. Its cost is "
      "the replay buffer, which grows with the number of tasks.")
    p("**Capacity isolation.** PackNet trains a task, prunes the least important weights, freezes the rest and uses "
      "the freed capacity for the next task, storing a mask per task [[mallya2018packnet]]. The simplest form of "
      "isolation trains one separate network per task. Isolation cannot forget by construction, so the meaningful "
      "question is not whether a method forgets less than isolation, but whether it keeps high performance at lower "
      "total memory while exploiting transfer between tasks.")
    p("**Sleep-like consolidation.** Tadros et al. implement a sleep-like phase in feed-forward networks: offline, "
      "with noisy input and local unsupervised Hebbian plasticity, it recovers previously forgotten tasks in "
      "incremental classification [[tadros2022sleep]]. Related work on self-consolidation distils accumulated "
      "experience into an agent's parameters instead of retrieving raw trajectories [[yu2025selfconsolidation]], "
      "which motivates counting the memory that replay needs.")
    p("**Spiking neural networks and surrogate gradients.** A spike is a step function of the neuron's membrane "
      "potential and has no useful derivative, so SNNs are trained end to end with *surrogate gradients*: a smooth "
      "stand-in derivative used only in the backward pass [[neftci2019surrogate]]. The slope of that surrogate "
      "matters much more in RL than in supervised learning [[vandenberghe2025adaptive]]. Toolkits such as "
      "SpikingJelly and snnTorch make SNNs easy to build [[fang2023spikingjelly, eshraghian2023training]]. Spiking "
      "Transformer designs combine spiking computation with attention, for example for offline RL "
      "[[pandey2025spiking]]; Spikingformer warns that many “spiking” architectures hide non-spike "
      "floating-point computation [[zhou2026spikingformer]], which matters when operations are counted.")
    p("**Local learning rules.** STDP changes a synapse according to the timing of pre- and post-synaptic spikes "
      "[[bi1998synaptic]]. Plain STDP has no notion of reward, so RL uses *three-factor* rules, in which a global "
      "signal such as reward or advantage modulates the local STDP term [[fremaux2016neuromodulated, lee2025brain, "
      "huo2025research]]. Direct feedback alignment (DFA) sends the output error to each hidden layer through a fixed "
      "random matrix, so a layer can learn without the exact transposed weights that backpropagation needs "
      "[[nokland2016direct]]. Homeostatic plasticity adjusts each neuron's excitability to keep its firing rate near "
      "a set point [[turrigiano2004homeostatic]].")
    p("**Evaluation.** Continual World argues that forgetting alone is an incomplete metric and that forward transfer "
      "must also be measured [[wolczyk2021continual]]; forgetting itself is usually measured as in Chaudhry et al. "
      "[[chaudhry2018riemannian]]. Energy claims for SNNs usually rest on a proxy that charges an accumulate per "
      "spike-driven operation and a multiply–accumulate per dense operation, using 45 nm estimates "
      "[[horowitz2014computing]].")

    h2("2.2", "Pros and Cons of the Stated Approaches/Methods")
    p("<<tab:proscons>> summarises the strengths and weaknesses of each family as they bear on this project.")
    tab("proscons", "Pros and cons of existing approaches to forgetting",
        ["Approach", "Pros", "Cons"],
        [["Regularisation (EWC; LwF)", "No stored data (LwF) or a fixed-size penalty (EWC); simple to add",
          "Protects old tasks at a cost in plasticity; EWC's cost grows with its strength λ"],
         ["Replay (CLEAR)", "Very strong in practice; simple; no task identities needed",
          "Needs a replay buffer that grows with the number of tasks; its budget is rarely reported"],
         ["Capacity isolation (one network per task; PackNet)", "Cannot forget by construction",
          "Parameters grow with the number of tasks; no transfer between separate networks"],
         ["Sleep-like consolidation", "Offline, needs no environment interaction; biologically motivated",
          "Usually still relies on stored data; its replay budget is seldom matched to replay's"],
         ["Spiking networks", "Event-driven; fewer operations possible on neuromorphic hardware",
          "Slower to simulate on CPUs and GPUs; energy savings are estimates without such hardware"],
         ["Local rules (STDP, three-factor, DFA, homeostasis)", "Biologically plausible; DFA avoids weight transport",
          "Credit assignment is weak; unstabilised STDP can run away; often tested without matched controls"]],
        [1.7, 2.3, 2.3])
    p("**The gap.** The brain-inspired mechanisms are typically introduced together and compared against an "
      "unmodified baseline, without matching the resources that the strongest non-biological controls would use: "
      "the number of replayed samples, the total memory including any buffer, the number of seeds, and, for local "
      "rules, the size of the update. NeuroPlast addresses this gap by testing each mechanism alone against such a "
      "control, with the budgets stated and matched, and by reporting the results whether they are positive or "
      "negative.")

    # =========================================================================================== CHAPTER 3
    B.append(("chapter", 3, "Requirement Artifacts"))
    h2("3.1", "Introduction")
    p("This chapter lists what the project needed in order to answer its research question: the hardware and "
      "software, the data (which in RL is generated rather than downloaded), the functional requirements of the "
      "software, its performance and security requirements, and the look and feel of the figures and the "
      "dashboard. Because the question is empirical, the most important requirement is that every comparison be "
      "fair and every number be reproducible.")

    h2("3.2", "Hardware and Software Requirements")
    cpu = subprocess.run(["lscpu"], capture_output=True, text=True).stdout
    model = re.search(r"Model name:\s+(.+)", cpu).group(1).strip()
    cores = re.search(r"^CPU\(s\):\s+(\d+)", cpu, re.M).group(1)
    mem = subprocess.run(["free", "-g"], capture_output=True, text=True).stdout.split("\n")[1].split()[1]
    other("CPU model of the build container", model, "lscpu on the cloud container used to build the report")
    other("logical CPUs of the container", cores, "lscpu")
    other("RAM of the container (GB, as reported by free -g)", mem, "free -g")
    p("All experiments ran on CPUs; no GPU was used at any stage. The models are small, and most of the time in "
      "each run goes to stepping the MiniGrid environments, so the work was parallelised across CPU cores, one "
      "thread per run. <<tab:hw>> lists the hardware.")
    tab("hw", "Hardware used",
        ["Item", "Used in this project"],
        [["Main compute", f"Cloud container with {cores} logical CPUs ({model}), about {mem} GB of RAM, no GPU"],
         ["Development machine", "A team member's laptop (CPU only), used for development, the dashboard and the demo"],
         ["Parallelism", "One thread per run (OMP_NUM_THREADS=1); up to four runs at a time"],
         ["Total compute (frozen study)", f"About {n('computeCoreHours')} single-thread CPU hours for "
                                          f"{n('computeRuns')} PPO runs"]],
        [1.8, 4.5])
    pk = ["torch", "numpy", "gymnasium", "minigrid", "snntorch", "scipy", "pandas", "matplotlib", "PyYAML", "pytest"]
    req = {}
    for line in open("requirements.txt"):
        line = line.strip()
        if line and not line.startswith("#"):
            name = re.split(r"[<>=]", line)[0].strip()
            req[name.lower()] = line
    rows = [["Python", "–", other("Python version", platform.python_version(), "platform.python_version()")]]
    for k in pk:
        v = md.version(k)
        other(f"installed version of {k}", v, "importlib.metadata in the build environment")
        rows.append([k, req.get(k.lower(), req.get(k.lower().replace("pyyaml", "pyyaml"), "–")), v])
    rows.append(["LaTeX (paper), LibreOffice (report), python-pptx (deck), Playwright (screenshots)",
                 "not in requirements.txt", "tools for the deliverables only"])
    tab("sw", "Software stack: requirement in requirements.txt and the version installed in the build environment",
        ["Package", "requirements.txt", "Installed version"], rows, [2.6, 1.9, 1.8])
    p("The spiking neurons are implemented in the project itself rather than taken from a library, so that the "
      "per-timestep spikes needed by the STDP rule are directly available; snnTorch is used only as a cross-check in "
      "the unit tests [[eshraghian2023training]]. The environments come from Gymnasium and MiniGrid "
      "[[chevalierboisvert2023minigrid]], and PPO is a single-file implementation in the style of CleanRL "
      "[[huang2022cleanrl]].")

    h2("3.3", "Specific Project Requirements")
    h3("3.3.1", "Data Requirements")
    p("The project uses no fixed dataset. In RL the data are the agent's own experience, generated by interacting "
      "with an environment, so the data requirement is a set of environments with known properties. All are "
      "MiniGrid grid-worlds with the standard egocentric 7×7 view, encoded as 20 one-hot channels (object type, "
      "colour and state) [[chevalierboisvert2023minigrid]]. <<tab:envs>> lists them.")
    tab("envs", "Environments and task sequences",
        ["Environment", "What the agent must do", "Used for", "Frames"],
        [["fetch3", "A 6×6 room with three objects; task k rewards picking up object k; any other pick-up ends the "
                    "episode with reward 0", "Continual learning (3 tasks)",
          f"{n('framesPerTaskFlat')} per task ({n('framesPerTaskHybrid')} for the SNN+Transformer agent)"],
         ["fetch5", "As fetch3, with five objects", "Continual learning (5 tasks)",
          f"{n('framesPerTaskFlat')} per task ({n('framesPerTaskFiveFair')} for the fair 5-task SNN runs)"],
         ["DoorKey-6x6", "Pick up a key, open a door, reach the goal; a run either solves it or stays near zero",
          "Local plasticity, homeostasis and DFA", f"{n('framesDoorKey')}"],
         ["Memory S7, S11, S13", "See a cue object at one end of a corridor and choose the matching object at the "
                                 "other end", "The Transformer working-memory question",
          f"{n('sSevenFrames')} (S7 pilot); {n('probeFrames')}–{n('probeLongFrames')} (S11/S13 probes)"]],
        [1.2, 2.5, 1.5, 1.1])
    p("In fetch3 and fetch5 the observations are statistically identical across tasks, so the same state demands "
      "different actions in different tasks. This guarantees interference under naive training, while navigation "
      "and picking up objects are shared skills that allow transfer. The agent is told the task identity; without it "
      "the tasks would be unsolvable, because the goal is not visible. The other data the project produces are "
      "stored run files: for every run, its configuration, training curve, evaluation results and final checkpoint. "
      f"Replay and sleep keep a buffer of stored states, {n('bufferPerTask')} per task by default, at a measured "
      f"cost of {n('bytesPerState')} bytes per state ({n('bytesPerStateHybrid')} bytes for the SNN+Transformer "
      "agent's four-frame windows).")
    h3("3.3.2", "Functional Requirements")
    ul(["Train a PPO agent on a single task or on a sequence of tasks, with a CNN or SNN encoder and optional "
        "Transformer working memory.",
        "Switch each mechanism on or off independently through configuration: STDP and its random-update control, "
        "homeostasis, DFA, the sleep phase, and the baselines (naive fine-tuning, EWC, replay, isolation).",
        "Evaluate after every task on every task, producing the accuracy matrix R and the continual-learning metrics.",
        "Count operations per frame for the encoders, and memory as parameters plus stored states.",
        "Run the statistical tests that were specified for each comparison, and regenerate every number in the paper, "
        "the deck, the dashboard and this report from the frozen run files.",
        "Show the results and the trained agents on an offline dashboard, and run one episode of a trained agent on "
        "a CPU."])
    h3("3.3.3", "Performance and Security Requirements")
    ul(["**Resumable runs.** Experiments run from job lists through a queue that skips finished jobs and claims "
        "running ones with lock files, so an interrupted queue can simply be restarted.",
        "**Thread pinning.** Each run uses one CPU thread (OMP_NUM_THREADS=1), so parallel runs do not compete for "
        "cores and timings are comparable.",
        f"**CPU budget.** The frozen study had to fit on a 4-core machine: about {n('computeCoreHours')} "
        f"single-thread CPU hours for {n('computeRuns')} runs. Frame budgets were chosen so that each arm of a "
        "comparison received the same number of frames.",
        "**Checkpoints.** Every run stores its final checkpoint so that its agent can be re-evaluated or shown in "
        "the demo without retraining.",
        "**Security and privacy.** The project processes no personal data. The dashboard is a single offline HTML "
        "file that makes no network requests, and the demo runs locally on a CPU."])
    h3("3.3.4", "Look and Feel Requirements")
    ul(["Figures are drawn at their printed size with a consistent colour for each method across all figures, the "
        "deck and the dashboard (sleep blue, replay orange, isolation green, EWC yellow, naive pink).",
        "Every figure shows the spread across seeds (individual seeds or 95% confidence intervals), not only means.",
        "The dashboard works offline as one file, in light and dark themes and at phone width, offers a table view "
        "for every chart, and states every result in plain language next to its key number."])
    h2("3.4", "Summary")
    p("The project needs only commodity CPU hardware and an open-source Python stack. Its data are generated by "
      "MiniGrid environments chosen for specific properties: interference between tasks, bimodal single-task "
      "outcomes, and a need for memory. Its main non-functional requirements are fairness (matched budgets) and "
      "reproducibility (resumable runs and scripted numbers).")

    # =========================================================================================== CHAPTER 4
    B.append(("chapter", 4, "Design Methodology and its Novelty"))
    h2("4.1", "Methodology and Goal")
    p("The methodology is an *ablation study*: one agent with swappable parts, in which each brain-inspired mechanism "
      "is switched on alone and compared with a matched control that is not brain-inspired. The control is chosen to "
      "remove the most obvious alternative explanation for any improvement. <<tab:mech>> lists each mechanism, its "
      "implementation and its control.")
    tab("mech", "Mechanisms, implementations and controls (each row is switched on and off independently)",
        ["Mechanism", "Implementation", "Control / comparator"],
        [["Spiking encoder", f"LIF convolutional network, {n('snnT')} timesteps per frame, surrogate gradients; "
                             "membrane reset every environment step", "Same-topology CNN; operations at matched accuracy"],
         ["STDP", "Reward-modulated three-factor STDP on encoder synapses, added to the backprop update, stabilised by "
                  "mean-centring and homeostasis", "Random update of the same size, plus homeostasis; homeostasis alone"],
         ["Homeostasis", f"Per-neuron bias pushes the firing rate towards {n('homeoTarget')}", "Same agent without it"],
         ["DFA", "Backprop-free encoder credit assignment: encoder layers learn from the output error through fixed "
                 "random feedback (heads use exact gradients)", "DFA without homeostasis; backprop"],
         ["Sleep", "Offline phases: old-task distillation on stored states and current-task self-distillation",
          "CLEAR-style replay with the replayed samples matched exactly; EWC; naive fine-tuning"],
         ["Shared trunk", "One trunk shared by all tasks, with per-task heads and task embeddings",
          "One network per task (isolation), at equal total memory; after the freeze, isolation at quarter width"],
         ["Transformer memory (inconclusive)", "Transformer over the last K encoded frames",
          "Memoryless agent; a stack of the last K frames"]],
        [1.35, 2.75, 2.2])
    p("**Fair budgets.** Within a comparison, every arm used the same frames, learning rate and seeds; for the "
      "frozen runs the seeds were 1 to n in every arm, and for the post-freeze follow-up "
      f"{n('postSeeds')}. The one exception is the unmatched SNN+Transformer sleep arm (seeds 1–{n('nHybSleep')}), "
      f"shown next to replay on seeds 1–{n('nHybReplay')}; the budget-matched test uses seeds "
      f"1–{n('matchN')} in both arms.")
    p("**Prospective specification.** The work was done in four phases. The first two were exploratory. For the last "
      "two, the comparisons, test statistics, seed counts and stopping rules were recorded in the project's "
      "version-controlled experiment log before the runs. This is a project record, not a public registry. We call "
      "these tests *prospectively specified*.")
    p("**Statistics and labels.** Solve rates are compared with Fisher's exact test and the Newcombe interval for a "
      "difference of proportions [[newcombe1998interval]]; continuous metrics with Welch's t-test and its 95% "
      "confidence interval [[welch1947generalization]]; and where several contrasts form one planned family, Holm's "
      "correction is applied [[holm1979simple]]. With few seeds the intervals are wide, so a non-significant "
      "difference is read as “no large effect” only where the interval excludes large effects, and never as "
      "equivalence. Every result in Chapter 6 carries one of three labels: **[confirmatory]** for a prospectively "
      "specified test run as specified, **[exploratory]** for a comparison not specified in advance (its p-values "
      "are uncorrected and should be read as descriptive), and **[descriptive]** for numbers given without a test.")
    p("**Novelty.** The novelty of this project is not a new mechanism. Every mechanism tested comes from cited prior "
      "work (Chapter 2). The contribution is the controlled, resource-matched ablation itself and what it found: the "
      "mechanisms are isolated one at a time, each is given a control that matches the resource a sceptic would point "
      "to (update size, replayed samples, total memory, and later network width), the tests are fixed in advance, "
      "and every earlier positive finding that did not survive these controls is reported with its cause.")

    h2("4.2", "Functional Modules Design and Analysis")
    p("**Encoder.** The encoder turns the 7×7 view into a feature vector. The CNN version has three 2×2 "
      f"convolutions with {n('convChannels')} channels and a linear layer to {n('featDim')} features. The SNN version "
      "has the same topology with leaky integrate-and-fire (LIF) neurons: each neuron accumulates input into a "
      f"membrane potential that leaks (factor β, initially {n('lifBeta')}, learnable), emits a spike when it "
      f"crosses a threshold, and is reset. Each frame is presented for {n('snnT')} timesteps, and the membrane is "
      "reset at every environment step, so the SNN has no memory across frames. Training uses a fast-sigmoid "
      f"surrogate gradient with slope {n('surrogateSlope')} [[neftci2019surrogate]].")
    p("**Transformer working memory.** In the SNN+Transformer agent, each of the last K frames is encoded and a "
      "one-layer Transformer encoder [[vaswani2017attention]] attends over them, reading out the newest position; "
      f"padding before the start of an episode is masked. On the fetch tasks K = {n('hybridWindow')}.")
    p("**Shared trunk and actor–critic heads.** Every non-isolation agent is a shared trunk with per-task heads. "
      f"The trunk (the encoder) is shared by all tasks; for the CNN agent it has {n('paramsCnnTrunk')} parameters. "
      f"Each task adds an actor–critic head ({n('paramsHead')} parameters: a policy and a value network) and a "
      f"{n('paramsTaskEmb')}-dimensional task embedding, {n('paramsPerTaskShared')} parameters per task, so the "
      f"3-task agent has {n('paramsCnnShared')} parameters. Isolation instead adds a full network of "
      f"{n('paramsCnnPerTask')} parameters per task. The parameters that grow with the number of tasks are therefore "
      "the heads and embeddings and, for replay and sleep, the replay buffer.")
    p("**Plasticity rules.** After every PPO optimiser step, the spikes recorded in that minibatch produce a local "
      "STDP update for each encoder layer (eligibility traces with pre/post pairing), multiplied by a third factor, "
      "the normalised advantage of each sample. The update is normalised to unit root-mean-square (RMS) size and "
      f"scaled by α times the learning rate (α = {n('stdpAlpha')} in the specified comparison). The "
      "*stabilised* version mean-centres pre- and post-synaptic spikes and adds homeostasis. The *random* control "
      "applies an update with the same per-layer RMS but a random direction. Homeostasis nudges each neuron's bias "
      f"towards a target firing rate of {n('homeoTarget')} with rate {n('homeoLr')}. DFA replaces the backpropagated "
      "gradient in the encoder with the output error sent through fixed random matrices [[nokland2016direct]].")
    p(f"**Sleep phase and replay baseline.** Every {n('sleepPeriod')} PPO updates, and once at the end of each task, "
      f"the sleeping agent goes offline for {n('sleepSteps')} gradient steps on two losses: distillation towards the "
      "stored policy and value on states saved at the end of each earlier task, and self-distillation of the current "
      "task towards a frozen snapshot taken at sleep onset. The replay baseline adds the same distillation loss on one "
      "replayed minibatch to every PPO minibatch, as in CLEAR [[rolnick2019experience]]. Both keep CLEAR's cloning "
      "losses but not its off-policy RL loss, because PPO is on-policy. With the number of replayed states matched, "
      "the comparison therefore isolates *when* replay happens (offline versus interleaved), plus the "
      "self-distillation term.")
    p("**Baselines.** The non-biological comparators follow the three standard strategies: regularisation (an online "
      "EWC that keeps one accumulated Fisher estimate and one anchor copy of the parameters, with "
      "λ = 10, 100, 1000), replay (above), and capacity isolation (one fresh network per task), with naive "
      "fine-tuning as the lower bound. The post-freeze follow-up adds LwF [[li2017learning]] with a copy of the "
      f"network stored in 8-bit integers (int8) as the teacher, distilling {n('postLwfBatch')} current-task states per "
      "PPO minibatch; it stores no data between tasks.")
    p("**Evaluation and statistics pipeline.** After training on task i, the agent is evaluated on every task j for "
      f"{n('evalEpisodesCl')} episodes, giving the accuracy matrix entry R(i, j). Final accuracy (ACC) is the mean of "
      "the last row; forgetting (FORGET) is the average drop from each old task's best value to its final value "
      "[[chaudhry2018riemannian]]; forward transfer (FWT) compares the area under each task's training curve with a "
      "from-scratch network [[wolczyk2021continual]]. Single-task runs report the solve rate (final evaluation "
      f"return above {n('solveThreshold')} over {n('evalEpisodesSingle')} episodes) and the AUC, the mean training "
      "return over the run, which measures learning speed. A numbers pipeline reads every frozen run file and "
      "computes every number in the paper, the deck, the dashboard and this report.")

    h2("4.3", "Software Architectural Designs")
    p("<<fig:arch>> shows the agent's data flow: the environment's view goes through the encoder, the optional "
      "Transformer memory and the per-task actor–critic heads, which choose the action. <<fig:wake>> shows the "
      "continual training loop of the sleep variant: online learning (wake) and offline consolidation (sleep) "
      "alternate on the same shared trunk, linked by the replay buffer.")
    fig("arch", f"{RF}/arch_pipeline.png", "Architecture of the agent: environment, spiking encoder, Transformer "
        "working memory and actor–critic heads", 6.3)
    fig("wake", f"{RF}/arch_wake_sleep.png", "Continual training of the sleep variant: wake and sleep phases "
        "alternate on the same shared trunk, linked by the replay buffer", 6.3)
    p("The code is organised so that each mechanism is one module and each experiment is one configuration file. "
      "<<tab:layout>> gives the layout.")
    tab("layout", "Repository layout",
        ["Path", "Contents"],
        [["neuroplast/models/", "LIF neuron and surrogate gradient, CNN and SNN encoders with operation counting, "
                                "Transformer memory, actor–critic heads, the agent"],
         ["neuroplast/learning/", "Trace STDP and eligibility, the hybrid STDP rule with its random control and "
                                  "homeostasis, DFA"],
         ["neuroplast/sleep/, neuroplast/baselines/", "Sleep phase; naive, EWC, replay, replay buffer"],
         ["neuroplast/envs/, neuroplast/eval/", "MiniGrid wrappers and task suites; continual-learning metrics"],
         ["train.py, continual.py", "Single-file PPO; the task-sequence runner (accuracy matrix, forgetting, transfer)"],
         ["configs/, jobs/, scripts/", "One YAML file per experiment; job lists; the resumable queue, statistics and "
                                       "memory accounting"],
         ["runs/, results/", "Frozen run files; aggregated tables"],
         ["paper/, presentation/, demo/, report/", "Paper and numbers pipeline, deck, dashboard and demo, this report"],
         ["explore/", "Post-freeze follow-up: its experiment log, runner, runs and tests"]],
        [2.3, 4.0])

    h2("4.4", "User Interface Designs")
    p("The project's user interface is an offline results dashboard: one self-contained HTML file built from the "
      "frozen results by the same numbers pipeline as the paper. It opens with the headline answer and a tally of "
      "the four result groups (<<fig:dashhead>>), then shows the architecture, a mechanism scorecard, the trained "
      "agents, forgetting curves for each task sequence (<<fig:dashfg>>), the memory trade-off with the post-freeze "
      "follow-up, and the next steps. Its design choices follow from the look-and-feel requirements: each method "
      "has one colour everywhere, every chart has a hover tooltip and a table view, the layout adapts to phone "
      "width, and the page works in light and dark themes. More screenshots are in Appendix A.")
    fig("dashhead", f"{RF}/dash_header.png", "Dashboard: headline, plain-language summary and the tally of result "
        "groups (light theme)", fitw(f"{RF}/dash_header.png", 6.0, 4.0))
    fig("dashfg", f"{RF}/dash_forgetting.png", "Dashboard: forgetting curves with a sequence selector; shaded bands "
        "are 95% confidence intervals", fitw(f"{RF}/dash_forgetting.png", 6.0, 4.4))
    h2("4.5", "Summary")
    p("The design is one agent with swappable modules and a matched control for every mechanism, evaluated with a "
      "fixed metric set and statistics specified in advance. The architecture separates the encoder, memory, heads, "
      "plasticity rules, sleep phase and baselines, so that each ablation changes exactly one component. The novelty "
      "lies in the controlled, resource-matched comparison, not in any individual mechanism.")

    # =========================================================================================== CHAPTER 5
    B.append(("chapter", 5, "Technical Implementation and Analysis"))
    h2("5.1", "Outline")
    p("The implementation is written in Python with PyTorch and runs on CPUs. This chapter shows the key code "
      "solutions with short excerpts taken verbatim from the repository: the spiking neuron, the stabilised STDP rule "
      "and homeostasis, the matched random-update control, the sleep phase's replay budget, the memory accounting and "
      "the statistical tests. It then describes the experiment queue and the prototype (the dashboard and the demo "
      "script). Longer excerpts are in Appendix B. <<tab:protocol>> lists the training protocol shared by all arms.")
    tab("protocol", "Training protocol (identical in every arm of a comparison)",
        ["Setting", "Value"],
        [["PPO", f"{n('numEnvs')} parallel environments × {n('numSteps')} steps per rollout, {n('updateEpochs')} "
                 f"epochs, minibatch {n('minibatch')}, clip {n('clipCoef')}, γ = {n('gammaVal')}, GAE λ = "
                 f"{n('gaeLambda')}, early stop at approximate KL {n('targetKl')}"],
         ["Learning rate (Adam)", f"{n('lrFlat')} (CNN, SNN); {n('lrHybrid')} (SNN+Transformer, memory probes)"],
         ["Frames per task", f"{n('framesPerTaskFlat')} (CNN, SNN); {n('framesPerTaskHybrid')} (SNN+Transformer); "
                             f"{n('framesPerTaskFiveFair')} (5-task SNN, fair budget)"],
         ["SNN", f"{n('snnT')} timesteps, learnable leak (initial {n('lifBeta')}), threshold 1, surrogate slope "
                 f"{n('surrogateSlope')}, membrane reset per frame"],
         ["Plasticity", f"α = {n('stdpAlpha')} (specified comparison); homeostasis target {n('homeoTarget')}, "
                        f"rate {n('homeoLr')}"],
         ["Sleep", f"every {n('sleepPeriod')} updates and at the end of each task; {n('sleepSteps')} steps per phase; "
                   f"{n('bufferPerTask')} states per task"],
         ["Evaluation", f"{n('evalEpisodesCl')} episodes per task (continual); {n('evalEpisodesSingle')} (single "
                        "task); actions sampled from the policy"]],
        [1.6, 4.7])

    h2("5.2", "Technical Coding and Code Solutions")
    h3("5.2.1", "LIF Neuron and Surrogate Gradient")
    p("The spike is a step function, so its derivative is replaced in the backward pass by a fast-sigmoid surrogate. "
      "The neuron is stateless per call: the caller threads the membrane potential through the timesteps, which "
      "makes the per-timestep spikes available to the STDP rule. The reset term is detached from the gradient, as in "
      "snnTorch, for stable training.")
    code("Excerpt from neuroplast/models/snn/lif.py", snippet("neuroplast/models/snn/lif.py", r"^class FastSigmoidSpike",
                                                              r"return grad, None") +
         "\n" + snippet("neuroplast/models/snn/lif.py", r"def forward\(self, current", r"return s, v"))
    h3("5.2.2", "Stabilised STDP and Homeostasis")
    p("Unstabilised STDP ran away in early experiments: because the frame is static across the SNN's timesteps, the "
      "same pre/post pairings repeat, and a mostly potentiating rule feeds back on itself. The stabilised rule "
      "mean-centres each neuron's spikes, which turns STDP into a covariance rule with no expected drift under "
      "independent firing, and homeostasis nudges each neuron's bias towards a target firing rate.")
    code("Excerpt from neuroplast/learning/hybrid.py", snippet("neuroplast/learning/hybrid.py", r"^def _center",
                                                               r"return list\(x\)") +
         "\n" + snippet("neuroplast/learning/hybrid.py", r"def _homeostasis", r"self.homeo_lr \* \(self.homeo - rate\)"))
    h3("5.2.3", "The Matched Random-Update Control")
    p("The control that separates “STDP helps” from “any perturbation of this size helps” is "
      "implemented in the same loop as STDP. Whatever the update's direction (STDP or random), it is rescaled to "
      "the same per-layer RMS, α times the learning rate, before it is added to the weights.")
    code("Excerpt from neuroplast/learning/hybrid.py (HybridSTDP.after_step)",
         snippet("neuroplast/learning/hybrid.py", r"if self.mode == \"random\":", r"layer.weight \+= \(self.alpha"))
    h3("5.2.4", "Sleep Phase with a Matched Replay Budget")
    p("To compare sleep with replay at an exactly matched budget, a sleep run can be given a total replay budget "
      "equal to the measured number of replayed states of the replay run with the same seed. The budget is spread "
      "as evenly as possible over the sleep phases the schedule will run.")
    code("Excerpt from neuroplast/sleep/sleep.py (Sleep.plan_steps)",
         snippet("neuroplast/sleep/sleep.py", r"def plan_steps", r"return \[q \+ \(1 if i < r"))
    h3("5.2.5", "Memory Accounting")
    p(f"Total memory is fp32 parameters plus the replay buffer at its measured storage cost: {n('bytesPerState')} "
      "bytes per stored state for the CNN and SNN agents (a 147-byte uint8 observation, a mask, a task identifier, "
      f"the teacher's logits and value) and {n('bytesPerStateHybrid')} bytes for the four-frame windows of the "
      "SNN+Transformer agent.")
    code("Excerpt from scripts/memory_fair.py", snippet("scripts/memory_fair.py", r"^OBS_B, FIXED_B", r"return OBS_B \* window"))
    h3("5.2.6", "Statistical Tests")
    p("The numbers pipeline computes the difference of means with a Welch confidence interval, and the difference of "
      "solve rates with a Newcombe (Wilson score) interval and Fisher's exact test.")
    code("Excerpt from paper/compute_numbers.py", snippet("paper/compute_numbers.py", r"^def welch\(a, b\)",
                                                          r"return d, d - math.sqrt"))
    h3("5.2.7", "Experiment Queue and Reproducibility Pipeline")
    p("Experiments are listed in job files, one job per line with a “done” marker. The queue skips finished "
      "jobs and claims running ones with a lock file holding the worker's process id, so an interrupted queue resumes "
      "where it stopped, and stale locks left by dead workers are taken over. After the runs, one command per "
      "deliverable regenerates everything from the frozen files: `python paper/compute_numbers.py` writes every "
      "number with its source, `python paper/figures/make_figures.py` the figures, `python demo/build_dashboard.py` "
      "the dashboard, `python presentation/build_deck.py` the deck and `python report/build_report.py` this report.")
    code("Excerpt from scripts/run_queue.py", snippet("scripts/run_queue.py", r"^def claim\(marker\)", r"^    return None$"))

    h2("5.3", "Prototype Submission")
    p("The prototype consists of two parts that run on any CPU, with no GPU and no network connection.")
    p("**Results dashboard.** `python demo/build_dashboard.py` builds `demo/index.html`, a single offline HTML file "
      "that can be opened in any browser. It shows the scorecard, the two demo agents (naive fine-tuning and sleep) "
      "with recorded episodes and their real accuracy matrices, the forgetting curves and the memory trade-off "
      "(screenshots in Appendix A).")
    p("**Live episode.** `python demo/live_episode.py --agent sleep --task 0` runs one episode of a trained agent on "
      "the CPU and prints each step and the agent's accuracy matrix; `--agent naive` shows the naive agent, and "
      "`--window` opens a game window. The agents are the fetch3 CNN agents of seed 1 after the full three-task "
      "sequence, the same checkpoints as the dashboard. One episode is an anecdote; the measured score is the "
      f"accuracy matrix (final rows {n('demoNaiveRow')} for the naive agent and {n('demoSleepRow')} for the sleep "
      "agent).")
    p("**Paper and deck.** The paper is built with LaTeX from the numbers pipeline, in a preprint and an anonymous "
      "journal version, and the slide deck is generated by a script from the same numbers.")
    h2("5.4", "Summary")
    p("The implementation keeps each mechanism small and inspectable: a 40-line LIF neuron, an STDP rule that shares "
      "its scaling with its random control, a sleep phase whose replay budget can be matched exactly, a measured "
      "memory cost per stored state, and statistical tests in one script. The resumable queue and the numbers "
      "pipeline make the whole study reproducible from its frozen run files.")

    # =========================================================================================== CHAPTER 6
    B.append(("chapter", 6, "Project Outcome and Applicability"))
    h2("6.1", "Key Implementations Outline of the System")
    p(f"The system was used to run {n('computeRuns')} PPO runs, about {n('computeCoreHours')} single-thread CPU hours "
      f"in total, with {n('seedRange')} seeds per arm in the main comparisons. After the results were frozen, one "
      "further prospectively specified comparison was run (the post-freeze follow-up, Section 6.2.2). <<tab:score>> "
      "summarises the outcome for each mechanism, in the four groups used throughout this chapter: what survived "
      "the controls, what is conditional, what did not survive, and what remains unresolved.")
    tab("score", "Outcome by mechanism",
        ["Mechanism", "Group", "Key result (95% CI)", "Label"],
        [["Homeostasis with DFA (backprop-free encoder)", "Survived",
          f"Learning speed (AUC) {n('dfaAucCi')}, Holm p = {n('dfaAucHolm')}; solve rate not significant", "Confirmatory"],
         ["Shared trunk with per-task heads", "Conditional",
          f"200 states/task: similar mean ACC to full-width isolation in {n('smallSnnTwoHundredMemPct')}% of the "
          f"memory; quarter-width isolation beat it after the freeze ({n('postNarrowMean')} vs {n('postLwfMean')})",
          "Confirmatory"],
         ["Spiking encoder", "Conditional",
          f"Works at {n('rqThreeSnnTwoLowOps')} operations per frame where the CNN needs {n('rqThreeCnnFloorOps')}; "
          f"measured CPU cost {n('snnCpuSlowdown')}×", "Descriptive"],
         ["STDP (tested rule)", "Did not survive", f"vs same-size random update: AUC {n('rqOneContrBAucCi')}",
          "Confirmatory"],
         ["Homeostasis with backprop", "Did not survive", f"AUC {n('rqOneContrCAucCi')}, Holm p = "
                                                          f"{n('rqOneContrCAucHolm')}", "Confirmatory"],
         ["Sleep-like consolidation", "Did not survive", f"vs replay at a matched budget: ACC {n('matchAccCi')}",
          "Confirmatory"],
         ["Transformer working memory", "Unresolved", "Benchmark validity not established", "Confirmatory (decision rule)"]],
        [1.65, 0.95, 2.75, 0.95], sz=9.5)

    h2("6.2", "Significant Project Outcomes")
    p("**Headline.** No mechanism produced a statistically reliable improvement in final accuracy or forgetting "
      "beyond its matched control. Given the seed counts, these results rule out large effects rather than show "
      "equivalence, and they rule out large effects only where the intervals are narrow. The subsections below "
      "give the evidence for each group.")
    h3("6.2.1", "Survived: Homeostasis Speeds Up Backprop-Free Learning")
    p(f"[confirmatory] With DFA as backprop-free encoder credit assignment (heads use exact gradients; "
      f"{n('rqOneDfaN')} seeds per arm, the same seeds in both arms), homeostasis raised the AUC from "
      f"{n('rqOneDfaAuc')} to {n('rqOneDfaHomeoAuc')} (difference {n('dfaAucCi')}, {n('dfaAucRatio')}×; Welch "
      f"p = {n('dfaAucP')}, Holm p = {n('dfaAucHolm')}, significant). It also raised the solve rate from "
      f"{n('rqOneDfaSolved')} to {n('rqOneDfaHomeoSolved')} (difference {n('dfaSolveCi')}; Fisher p = "
      f"{n('dfaFisherP')}, Holm {n('dfaFisherHolm')}), which is not significant, and the remaining seeds still "
      "failed outright (<<fig:rq1>>). In our interpretation, homeostasis is a real but partial fix for DFA: it speeds up "
      "learning without weight transport in the encoder, but it does not make DFA reliable, and backprop with "
      f"homeostasis ({n('rqOneHomeoSolved')} solved) remains better.")
    fig("rq1", f"{PF}/fig_rq1.png", f"Local-plasticity arms on DoorKey-6x6 ({n('rqOneBpN')} seeds per arm). Dots are "
        "seeds' AUC (filled: solved); bars are means with 95% confidence intervals; numbers above the dots are seeds "
        "solved. Left: backprop as the learning signal; right: DFA with and without homeostasis", 6.2)
    h3("6.2.2", "Conditional: Shared Trunk and Spiking Encoder")
    p(f"**Shared trunk versus isolation.** [descriptive] Per parameter, the shared trunk with sleep is clearly more "
      f"efficient than isolation (ACC per parameter {n('perParamRatioRange')} times higher). But at the default "
      f"{n('bufferPerTask')} stored states per task, the buffer outweighs the extra networks: isolation uses less "
      f"total memory and gets {n('perMbRatioRange')} times more accuracy per MB on every sequence (<<fig:memory>>). "
      f"[confirmatory] With a small buffer the picture changes. On the SNN, sleep with 200 states per task reaches a "
      f"similar mean accuracy to isolation ({n('smallSnnTwoHundredAcc')} vs {n('snnIsoAcc')}; difference "
      f"{n('snnSleepTwoHundredVsIsoCi')}, p = {n('snnSleepTwoHundredVsIsoP')}, {n('smallSnnTwoHundredN')} seeds per arm) "
      f"in {n('smallSnnTwoHundredMb')} MB against {n('memSnnThreeIsoMb')} MB, {n('smallSnnTwoHundredMemPct')}% of the "
      "memory. With so few seeds the interval is wide: it rules out neither a large deficit nor a large advantage, "
      "so this is not evidence of equal accuracy.")
    fig("memory", f"{PF}/fig_memory.png", "Final accuracy against total memory (fp32 parameters plus replay buffer, "
        f"log scale). Numbers above the axis are replay states stored per task; the 5-task SNN runs use "
        f"{n('framesPerTaskFiveFair')} frames per task", 6.2)
    p("**Post-freeze follow-up: isolation's network width.** [confirmatory, post-freeze] The comparisons above use "
      f"isolated networks of the same width as the shared trunk ({n('paramsCnnPerTask')} parameters each). After the "
      "results were frozen, a follow-up specified in advance in its own experiment log tested narrower isolated "
      f"networks on the 5-task CNN sequence, with seeds {n('postSeeds')} in every arm (<<fig:c4>>, <<tab:c4>>). "
      f"Quarter-width isolation ({n('postNarrowPerTask')} parameters per task, {n('postNarrowMb')} MB) reached "
      f"{n('postNarrowAcc')}. A shared trunk trained with LwF and an int8 teacher ({n('postLwfMb')} MB) reached "
      f"{n('postLwfAcc')} (difference {n('postHEightCi')}, Holm p = {n('postHEightHolm')}), and the same trunk with "
      f"replay ({n('postReplayBuf')} states per task, {n('postReplayMb')} MB) reached {n('postReplayAcc')} (difference "
      f"{n('postHNineCi')}, Holm p = {n('postHNineHolm')}). Quarter width minus full width was {n('postHTenCi')}. In "
      "our interpretation, the frozen isolation baseline's networks were much larger than these tasks need, so the "
      "small-buffer advantage of the shared trunk holds only against full-width isolation. Two limits apply: the "
      "shared trunk was not shrunk in the same way, and the SNN case is untested, because the only narrow SNN "
      f"isolation runs were a two-seed pilot at {n('framesPerTaskFlat')} frames per task.")
    fig("c4", f"{RF}/c4_dots.png", f"Post-freeze follow-up (CNN, 5 tasks, {n('postN')} seeds per arm): mean final "
        "accuracy on a 0.85–1.00 scale, with memory", 6.2)
    tab("c4", f"Post-freeze follow-up: isolation at the shared trunk's memory (CNN, 5 tasks, seeds {n('postSeeds')})",
        ["Arm", "Final ACC (mean ± SD)", "Memory (MB)", "Difference vs quarter-width isolation [95% CI]"],
        [["Isolation, quarter width", n("postNarrowAcc"), n("postNarrowMb"), "–"],
         ["Shared trunk + LwF (int8 teacher)", n("postLwfAcc"), n("postLwfMb"), f"{n('postHEightCi')} (Holm p = "
                                                                                f"{n('postHEightHolm')})"],
         ["Shared trunk + replay", n("postReplayAcc"), n("postReplayMb"), f"{n('postHNineCi')} (Holm p = "
                                                                          f"{n('postHNineHolm')})"],
         ["Isolation, full width", n("postIsoAcc"), n("postIsoMb"), f"quarter − full: {n('postHTenCi')}"]],
        [2.0, 1.3, 0.9, 2.1], sz=10)
    p("**Spiking encoder.** [descriptive] Where both encoders work, they are equally accurate at matched operation "
      f"counts (<<fig:frontier>>). The sparsified CNN still works at {n('rqThreeCnnFloorOps')} operations per frame "
      f"(accuracy {n('rqThreeCnnFloorAcc')}), but a stronger penalty silences it entirely. The SNN keeps working "
      f"below that floor: with 2 timesteps it reaches {n('rqThreeSnnTwoLowOps')} operations at accuracy "
      f"{n('rqThreeSnnTwoLowAcc')}, about {n('rqThreeOpsRatio')} times fewer. Under the estimated arithmetic energy "
      f"proxy this is {n('rqThreeSnnTwoLowEnergy')} nJ per frame against {n('rqThreeCnnFloorEnergy')} nJ for the "
      "sparsest working CNN; these are estimates that assume event-driven hardware, not measured power. The cost we "
      f"did measure, in our CPU implementation, goes the other way: one PPO training step takes {n('msSnn')} ms for "
      f"the SNN against {n('msCnn')} ms for the CNN ({n('snnCpuSlowdown')}×; CPU process time on one thread for "
      "one forward pass, backward pass and Adam update on 256 observations), because the SNN simulates several "
      "timesteps per frame.")
    fig("frontier", f"{PF}/fig_frontier.png", f"Accuracy against operation count for the spiking and the conventional "
        f"encoder (behaviour cloning of a DoorKey-6x6 teacher, {n('rqThreeSeeds')} seeds per point). Left: held-out "
        "action accuracy; right: closed-loop return of the cloned policy", 6.2)
    h3("6.2.3", "Did Not Survive: STDP, Homeostasis with Backprop, Sleep")
    p(f"**STDP.** [confirmatory] With backprop as the learning signal, the four arms solved DoorKey-6x6 at similar "
      f"rates: backprop {n('rqOneBpSolved')}, with homeostasis {n('rqOneHomeoSolved')}, with stabilised STDP "
      f"{n('rqOneStdpSolved')}, and with the matched random update {n('rqOneRandSolved')}. All four specified "
      "contrasts are null after Holm correction (<<tab:rq1>>). The tested rule (stabilised reward-modulated STDP at "
      f"α = {n('stdpAlpha')}, added to the backprop update) was not distinguishable from a random update of the "
      f"same size (AUC difference {n('rqOneContrBAucCi')}, p = {n('rqOneContrBAucP')}). This interval is wide, so it "
      "does not exclude a large effect in either direction: it means no detectable effect of this rule in this "
      "setup, not that STDP cannot help. [exploratory] Unstabilised STDP hurt: in an earlier DoorKey-5x5 experiment "
      f"it solved {n('vanillaStdpSolved')} runs against {n('vanillaBpSolved')} for backprop, with runaway firing "
      f"(spike rate {n('vanillaStdpRate')} against {n('bpRate')}).")
    tab("rq1", f"Specified local-plasticity contrasts on DoorKey-6x6 ({n('rqOneBpN')} seeds per arm; first arm minus "
               "second; Holm-corrected p in brackets)",
        ["Contrast", "Δ solve rate [95% CI]", "Fisher p (Holm)", "Δ AUC [95% CI]", "Welch p (Holm)"],
        [["STDP vs homeostasis alone", n("rqOneContrASolveCi"), f"{n('rqOneContrAFisherP')} ({n('rqOneContrAFisherHolm')})",
          n("rqOneContrAAucCi"), f"{n('rqOneContrAAucP')} ({n('rqOneContrAAucHolm')})"],
         ["STDP vs matched random", n("rqOneContrBSolveCi"), f"{n('rqOneContrBFisherP')} ({n('rqOneContrBFisherHolm')})",
          n("rqOneContrBAucCi"), f"{n('rqOneContrBAucP')} ({n('rqOneContrBAucHolm')})"],
         ["Homeostasis vs backprop", n("rqOneContrCSolveCi"), f"{n('rqOneContrCFisherP')} ({n('rqOneContrCFisherHolm')})",
          n("rqOneContrCAucCi"), f"{n('rqOneContrCAucP')} ({n('rqOneContrCAucHolm')})"],
         ["Random vs homeostasis", n("rqOneContrDSolveCi"), f"{n('rqOneContrDFisherP')} ({n('rqOneContrDFisherHolm')})",
          n("rqOneContrDAucCi"), f"{n('rqOneContrDAucP')} ({n('rqOneContrDAucHolm')})"],
         ["DFA + homeostasis vs DFA", n("dfaSolveCi"), f"{n('dfaFisherP')} ({n('dfaFisherHolm')})", n("dfaAucCi"),
          f"{n('dfaAucP')} ({n('dfaAucHolm')})"]],
        [1.75, 1.25, 0.95, 1.4, 0.95], sz=9.5)
    p(f"**Homeostasis with backprop.** [confirmatory] Homeostasis raised the AUC by {n('rqOneContrCAucCi')} (raw "
      f"p = {n('rqOneContrCAucP')}), which does not survive Holm correction (p = {n('rqOneContrCAucHolm')}).")
    p("**Sleep-like consolidation.** [exploratory] Sleep and replay both remove most of naive fine-tuning's "
      "forgetting on every agent (<<fig:forgetting>>, <<tab:continual>>). EWC protects old tasks at a plasticity cost "
      "that grows with λ: the return on the last task right after training it falls from "
      f"{n('naiveLastTaskCnn')} for naive fine-tuning to {n('ewcLastTaskTen')}, {n('ewcLastTaskHundred')} and "
      f"{n('ewcLastTaskThousand')} for λ = 10, 100 and 1000. At unmatched replay budgets, sleep never differs "
      "significantly from replay.")
    fig("forgetting", f"{PF}/fig_forgetting.png", "Forgetting on the continual task sequences. Top: return on the "
        "first task after each task is trained; bottom: mean return over the tasks seen so far. Means over seeds "
        "with 95% confidence intervals (clipped to [0, 1])", 6.2)
    tab("continual", "Final ACC (mean ± SD over seeds) and FORGET on the continual sequences [exploratory; the "
                     "matched row belongs to the confirmatory test]",
        ["", "CNN, 3 tasks", "SNN, 3 tasks", "SNN+Transformer, 3 tasks", "CNN, 5 tasks"],
        [["**Final ACC**", "", "", "", ""],
         ["naive", n("accCnnNaive"), n("accSnnNaive"), n("accHybNaive"), n("accCnnFiveNaive")],
         ["EWC (λ = 100)", n("accCnnEwc"), n("accSnnEwc"), "–", "–"],
         ["replay", n("accCnnReplay"), n("accSnnReplay"), n("accHybReplay"), n("accCnnFiveReplay")],
         ["sleep", n("accCnnSleep"), n("accSnnSleep"), n("accHybSleep"), n("accCnnFiveSleep")],
         ["sleep, matched", "–", "–", n("accHybSleepMatched"), "–"],
         ["isolation", n("accCnnIso"), n("accSnnIso"), n("accHybIso"), n("accCnnFiveIso")],
         ["**FORGET (mean)**", "", "", "", ""],
         ["naive", n("forgetCnnNaive"), n("forgetSnnNaive"), n("forgetHybNaive"), n("forgetCnnFiveNaive")],
         ["replay", n("forgetCnnReplay"), n("forgetSnnReplay"), n("forgetHybReplay"), n("forgetCnnFiveReplay")],
         ["sleep", n("forgetCnnSleep"), n("forgetSnnSleep"), n("forgetHybSleep"), n("forgetCnnFiveSleep")],
         ["sleep, matched", "–", "–", n("forgetHybSleepMatched"), "–"],
         ["**sleep − replay, ACC [95% CI]**", n("sleepVsReplayCnnCi"), n("sleepVsReplaySnnCi"),
          n("sleepVsReplayHybCi"), n("sleepVsReplayCnnFiveCi")]],
        [1.45, 1.2, 1.2, 1.25, 1.2], sz=9.5, align=["left", "center", "center", "center", "center"])
    p("[confirmatory] On the SNN+Transformer agent, the original sleep runs had consumed "
      f"{n('unmatchedExtraPct')}% more replayed states than replay. With an exactly matched budget, sleep was not "
      f"distinguishable from replay on final accuracy ({n('matchAccSleep')} vs {n('matchAccReplay')}; difference "
      f"{n('matchAccCi')}, p = {n('matchAccP')}, {n('matchN')} seeds per arm, the same seeds in both arms), and unlike "
      "the STDP contrasts this interval is narrow (<<fig:budget>>). [descriptive] Forgetting was also close "
      f"({n('matchForgetSleep')} vs {n('matchForgetReplay')}). In our interpretation, sleep's small earlier edge came "
      "from its extra replay, not from consolidating offline: in our implementation sleep is a way of scheduling "
      "replay, and at an equal budget we found no advantage for it. This is one sleep schedule on one agent; it does "
      "not show that offline consolidation can never help.")
    fig("budget", f"{PF}/fig_budget.png", f"Sleep versus replay at an exactly matched replay budget (SNN+Transformer, "
        f"fetch3; {n('matchN')} seeds per arm). Grey lines connect runs with the same seed. Left: final accuracy; "
        "right: forgetting", 6.2)
    h3("6.2.4", "Unresolved: Transformer Working Memory")
    p("[exploratory] On the fetch tasks, which are fully determined by the current frame, the Transformer has nothing "
      f"to remember. On MiniGrid-Memory S7, a memoryless CNN already succeeded in {n('sSevenCnnOne')} of episodes, so "
      "S7 does not isolate memory. [confirmatory] On S11 and S13, where the cue is beyond the agent's view at the "
      f"junction, the memoryless CNN was at chance, as required ({n('probeElevenOne')} and {n('probeThirteenOne')} "
      f"success), but so was a CNN with a {n('memWindow')}-frame stack ({n('probeElevenTwelve')} and "
      f"{n('probeThirteenTwelve')}) (<<fig:probe>>). Neither map met the criterion written before the probes, so the "
      "Transformer comparison was not run. [exploratory] A longer run of the stacked CNN on S11 for "
      f"{n('probeLongFrames')} frames also stayed at chance ({n('probeElevenTwelveLong')}). No agent learned to go "
      "back and look at the cue; they walk to the junction and guess. The question is therefore inconclusive, not "
      "answered in either direction. The SNN version of the isolation-width comparison is likewise unresolved.")
    fig("probe", f"{PF}/fig_probe.png", f"Memory-benchmark probes on MiniGrid-Memory S11 and S13 (CNN, one seed each): "
        f"single frame versus a stack of the last {n('memWindow')} frames. A return of 0.5 is chance", 6.2)
    h3("6.2.5", "What Did Not Survive Controls")
    p("Several results looked positive before the controls were in place. <<tab:survive>> lists them with the control "
      "that removed them; every “evidence then” entry is exploratory. The causes recur: too few seeds, too "
      "small a training budget for fresh networks, an unmatched replay budget, memory counted without the buffer, a "
      "benchmark that did not need memory, and, found after the freeze, a fixed isolation width.")
    tab("survive", "Earlier positive findings that shrank or disappeared once a control was added",
        ["Earlier finding", "Evidence then", "After the control", "Cause", "Label"],
        [["Sleep beats replay (SNN+Transformer)", f"{n('hybSleepReplayNThreeSeeds')}: {n('hybSleepReplayNThreeCi')}, "
                                                  f"p = {n('hybSleepReplayNThreeP')}",
          f"seeds 1–{n('nHybSleep')}: p = {n('hybSleepReplayNSixP')}; budget-matched, {n('matchN')} seeds: "
          f"{n('matchAccCi')}, p = {n('matchAccP')}", "Seed count; replay budget", "Conf."],
         ["Shared trunk far more efficient than isolation", f"ACC per parameter: {n('perParamRatioRange')}×",
          f"Per total memory at the default buffer, isolation is {n('perMbRatioRange')}× more efficient; 200 "
          f"states/task (SNN): similar mean ACC in less than half the memory", "Memory accounting",
          "Desc.; conf. (SNN)"],
         ["Sleep far beats isolation on 5 tasks (SNN)", f"{n('framesPerTaskFlat')}/task, {n('fiveSnnShortN')} seeds: "
                                                        f"{n('fiveSnnShortSleep')} vs {n('fiveSnnShortIso')}",
          f"{n('framesPerTaskFiveFair')}/task, {n('fiveSnnFairN')} seeds: {n('fiveSnnFairCi')}, p = {n('fiveSnnFairP')}",
          "Training budget", "Conf."],
         ["Homeostasis helps backprop", f"{n('homeoEarlySeeds')}: {n('homeoEarlySolved')} solved, p = "
                                        f"{n('homeoEarlyP')}",
          f"{n('rqOneBpN')} seeds: {n('rqOneContrCAucCi')}, Holm p = {n('rqOneContrCAucHolm')}",
          "Seed count; multiple comparisons", "Conf."],
         ["Homeostasis makes DFA work", f"{n('dfaEarlySolvedHomeo')} vs {n('dfaEarlySolvedDfa')} solved",
          f"{n('rqOneDfaN')} seeds: {n('rqOneDfaHomeoSolved')} vs {n('rqOneDfaSolved')}, Fisher p = {n('dfaFisherP')} "
          "(AUC effect survives)", "Seed count", "Conf."],
         ["Transformer helps on a memory task", f"S7 pilot: {n('sSevenSnnTf')} vs {n('sSevenSnnEight')} (frame stack)",
          f"Memoryless CNN on S7: {n('sSevenCnnOne')}; on S11/S13 no memory agent learns",
          "Benchmark validity; learning-rate confound", "Conf. (rule)"],
         ["With a small buffer, a shared trunk is more memory-efficient than isolation",
          f"Against full-width isolation: CNN, 200 states/task {n('smallCnnTwoHundredPerMb')} vs "
          f"{n('memCnnThreeIsoPerMb')} ACC per MB",
          f"Quarter-width isolation, CNN 5 tasks, {n('postN')} seeds: {n('postNarrowMean')} at {n('postNarrowMb')} MB vs "
          f"LwF {n('postLwfMean')} at {n('postLwfMb')} MB ({n('postHEightCi')})", "Network width not matched",
          "Conf., post-freeze"]],
        [1.35, 1.45, 1.95, 0.9, 0.75], sz=9)

    h2("6.3", "Project Applicability on Real-World Applications")
    p("The results apply most directly to agents that must keep learning after deployment under tight memory and "
      "compute budgets: mobile robots that meet new tasks in the field, embedded and edge devices that adapt on the "
      "device instead of in a data centre, and other systems where storing past experience is expensive. We did not "
      "build or test such systems; the points below state where the findings are relevant, not what we "
      "demonstrated.")
    ul(["**Memory budgets decide the method.** Whether a shared network or separate per-task networks is cheaper "
        "depends on the replay buffer and on the size of each network. A deployed system should account for its "
        "total memory, buffer included, before choosing.",
        "**Energy claims need hardware.** Spiking encoders reach fewer operations per frame, but on ordinary CPUs "
        "they were slower to train. Any energy benefit depends on event-driven hardware and should be measured there.",
        "**Matched controls before adoption.** A mechanism that looks helpful should be compared with a simple control "
        "of the same cost (a random update of the same size, replay with the same number of samples) before it is "
        "adopted, as these controls removed most of our early effects.",
        "**Learning without backpropagation.** Homeostasis made backprop-free (DFA) training of a spiking encoder "
        "faster, which is relevant to on-chip learning where backpropagation is hard to implement, though it did not "
        "make it reliable."])
    h2("6.4", "Inference")
    p("Across six tested mechanisms, none improved final accuracy or forgetting reliably beyond a matched "
      "non-biological control. One narrow effect survived (homeostasis speeding up DFA), two are conditional (the "
      "shared trunk's memory advantage, which depends on the buffer size and on isolation's network width, and the "
      "spiking encoder's lower operation count, which depends on hardware we did not use), and one question is "
      "unresolved (Transformer working memory). The most consistent finding is methodological: most early positive "
      "effects disappeared once seeds, training budgets, replay budgets, memory accounting and network width were "
      "made fair.")

    # =========================================================================================== CHAPTER 7
    B.append(("chapter", 7, "Conclusions and Recommendation"))
    h2("7.1", "Outline")
    p("We isolated six brain-inspired mechanisms in a small task-incremental continual RL setting and gave each a "
      "matched control; a seventh, Transformer working memory, remained inconclusive because benchmark validity was "
      "not established. No mechanism produced a statistically reliable improvement in final accuracy or forgetting "
      "beyond its matched control; given the seed counts, these results rule out large effects rather than show "
      "equivalence, and only where the intervals are narrow.")
    ul(["**Survived:** homeostasis sped up learning of a spiking encoder with backprop-free encoder credit assignment "
        "(DFA; heads use exact gradients), without making it reliable.",
        "**Conditional:** with a small replay buffer, a shared trunk with per-task heads and sleep reached similar mean "
        "accuracy to full-width isolation in less than half the memory, but after the freeze quarter-width isolation "
        "beat shared-trunk methods at equal or lower memory on a 5-task CNN sequence; and the spiking encoder reached "
        "a lower operation-count regime whose energy value is an estimate and whose measured CPU cost in our "
        "implementation was higher.",
        "**Did not survive:** the tested stabilised STDP rule against a same-size random update, homeostasis with "
        "backprop, and sleep against replay at a matched replay budget, together with the earlier positive findings "
        "of Section 6.2.5.",
        "**Unresolved:** the Transformer question and the SNN version of the isolation-width comparison."])
    p("The broader lesson is methodological: in this setting, the earlier positive effects of biological mechanisms "
      "shrank or disappeared once seed counts, training budgets, replay budgets, memory accounting and network width "
      "were made fair. We recommend that claims of this kind be reported with those budgets stated and matched.")
    h2("7.2", "Limitations/Constraints of the System")
    p("**Task identity is given.** Our setting is task-incremental: the agent knows which task it is doing. Removing "
      "the task identity from fetch3 would make it unsolvable (the goal is not visible), so a task-agnostic study "
      "needs a different benchmark, and sleep's distillation, which relies on per-task heads, would need "
      "redesigning.")
    p(f"**Scale.** All agents are small ({n('paramsCnnPerTask')}–{n('paramsHybridShared')} parameters), the "
      "environments are MiniGrid grid-worlds, sequences have 3 or 5 tasks, and budgets are "
      f"{n('framesPerTaskFlat')}–{n('framesPerTaskFiveFair')} frames per task. Results may not transfer to larger "
      "agents, longer sequences or visually rich tasks.")
    p(f"**CPU only, few seeds.** Everything ran on a 4-core CPU. Many continual arms have only {n('nCnnSleep')} seeds, "
      "so their confidence intervals are wide and a tie means at most “no large difference” where the "
      f"interval is narrow, never equivalence. The prospectively specified comparisons used {n('matchN')}–"
      f"{n('rqOneBpN')} seeds per arm, and the post-freeze follow-up {n('postN')}.")
    p("**Energy is an estimated arithmetic energy proxy.** The energy numbers multiply operation counts by literature "
      "constants for 45 nm arithmetic and assume event-driven hardware. We measured no power. The only cost we "
      "measured is CPU time in our implementation, and there the SNN was slower.")
    p("**Single settings.** Hyperparameters (homeostasis target, α, EWC λ beyond three values, sleep "
      "schedule) were not tuned per arm. A different homeostasis target or sleep schedule could change the results.")
    p("**Baselines are simple.** In the frozen study, isolation uses one full-width network per task; the post-freeze "
      "follow-up shows that narrower isolated networks change the memory comparison (Section 6.2.2), and "
      "PackNet-style masks would be cheaper still. The shared trunk was never shrunk in the same way. Our sleep phase "
      "is replay plus distillation, not the sleep replay consolidation of Tadros et al. [[tadros2022sleep]].")
    p(f"**The Transformer question is inconclusive.** On S7 memory was unnecessary; on S11 and S13 no agent, "
      f"including a CNN with a {n('memWindow')}-frame stack, learned to use the cue, even at {n('probeLongFrames')} "
      "frames. Benchmark validity was therefore not established, and the comparison was not run.")
    p("**Prospective specification was informal.** The test plans were recorded in the project's version-controlled "
      "experiment log before the runs, not in a public registry. The first two phases were exploratory, and their "
      "positive findings are exactly the ones Section 6.2.5 revisits. The follow-up was planned after the frozen "
      "results were known, although each of its tests was specified before its own runs.")
    h2("7.3", "Future Enhancements")
    p("Everything in this section is a plan or an open question, not a result; we make no prediction about what "
      "these studies will find.")
    p("**Scaling up.** The main limitation is scale: every experiment ran on a 4-core CPU, and the bottleneck was "
      "CPU-bound environment stepping, not model size. We plan to port the agent, the mechanisms and the baselines to "
      "JAX and run them on GPUs with XLand-MiniGrid, a GPU-native implementation of MiniGrid that is reported to "
      "simulate millions of environment steps per second on one GPU and has a rule-and-goal system for generating "
      "many related tasks [[nikulin2024xland]].")
    p("**Replication first.** Before trusting any new result, we will replicate the headline results on "
      "XLand-MiniGrid's ports of the MiniGrid environments: sleep versus replay at a matched replay budget, STDP "
      "versus a random update of the same size, and the post-freeze isolation-width result.")
    p("**Questions for the scaled study.**")
    ul(["Longer sequences of 10–50 related tasks where transfer matters, with full accuracy-versus-memory curves "
        "in which every method shrinks along its own size setting (isolation width, shared-trunk width, buffer size).",
        "10–20 seeds per arm, with a power analysis and equivalence tests, so that “no large effect” "
        "results become informative.",
        "A memory benchmark that agents can actually learn, then Transformer versus GRU versus frame stack, and a "
        "stateful SNN.",
        f"The SNN version of the isolation-width comparison at an adequate budget ({n('framesPerTaskFiveFair')} frames "
        "per task), whether the conclusions change with network size, and a task-agnostic setting with observable "
        "goals."])
    p("**Direction for publication: TinyML and embedded systems.** Following our supervisor's direction, the next "
      "phase will target a TinyML / embedded-systems IEEE venue, where memory and compute are the binding "
      "constraints. This requires four additions to the study, all of them planned work:")
    ol(["**Full training-time memory accounting:** not only the memory kept after training (parameters and buffer) "
        "but also the peak memory during training (activations, optimiser state, snapshots and reservoirs).",
        "**int8 versions of every method:** parameters, buffers and teacher networks stored in 8-bit integers, as the "
        "LwF teacher already was in the follow-up, so that methods are compared at deployable precision.",
        "**Accuracy-versus-memory curves:** for every method, swept along its own size setting, instead of single "
        "operating points.",
        "**Measurements on real embedded hardware:** latency, memory and energy measured on a microcontroller or "
        "edge board, replacing the estimated arithmetic energy proxy."])
    p("**Smaller follow-ups.** A replay-budget sweep comparing sleep with interleaved replay (motivated by an "
      "exploratory result only), splitting sleep into offline replay alone versus replay plus self-distillation, "
      "plots of the STDP runaway mechanism, a stronger CNN sparsifier such as k-winners-take-all, compressed or "
      "generative replay to shrink the buffer [[shin2017continual]], a PackNet variant that can regrow capacity, and "
      "one standard benchmark such as Continual World [[wolczyk2021continual]] for comparability. The methodology "
      "will stay the same: prospectively specified tests, fresh seeds for confirmation, matched resources, and a log "
      "of every failure.")
    h2("7.4", "Inference")
    p("NeuroPlast set out to find which brain-inspired mechanisms help continual RL when each is isolated and "
      "matched against a fair control. In a small task-incremental setting, the answer is that none produced a "
      "statistically reliable improvement in final accuracy or forgetting, one narrow effect on learning speed "
      "survived, and two advantages hold only under stated conditions. The project's main contribution is the "
      "controlled, resource-matched method and the record of which effects did not survive it. Scaling the same "
      "method up, and moving it towards memory-constrained embedded hardware, is the natural next step.")

    # =========================================================================================== APPENDIX A
    B.append(("appendix", "A", "Screenshots"))
    p("The screenshots below show the offline results dashboard (`demo/index.html`, light theme), frames from the "
      "recorded episodes of the two demo agents, and their accuracy matrices.")
    fig("ascore", f"{RF}/dash_scorecard.png", "Dashboard: mechanism scorecard with result groups, key numbers and "
        "evidence labels", fitw(f"{RF}/dash_scorecard.png", 6.2, 7.6))
    fig("aagents", f"{RF}/dash_agents.png", "Dashboard: the naive and sleep agents after the full fetch3 sequence, "
        "with recorded episodes and accuracy matrices", fitw(f"{RF}/dash_agents.png", 6.2, 6.0))
    fig("amemory", f"{RF}/dash_memory.png", "Dashboard: memory trade-off, with the key result and the post-freeze "
        "follow-up", fitw(f"{RF}/dash_memory.png", 6.2, 7.6))
    fig("anext", f"{RF}/dash_next.png", "Dashboard: the “What's next” section (plans, not results)",
        fitw(f"{RF}/dash_next.png", 6.2, 5.2))
    fig("anaive", f"{RF}/gif_frames_naive.png", "Final frames of the three recorded episodes per task, naive "
        "fine-tuning agent after the full sequence (the episode outcome is printed above each frame)",
        fitw(f"{RF}/gif_frames_naive.png", 6.0, 7.2))
    fig("asleep", f"{RF}/gif_frames_sleep.png", "Final frames of the three recorded episodes per task, sleep agent "
        "after the full sequence", fitw(f"{RF}/gif_frames_sleep.png", 6.0, 7.2))
    for agent, lab in (("naive", "naive fine-tuning"), ("sleep", "sleep")):
        path = f"runs/demo_ckpt/fetch3_{agent}_s1/results.json"
        R = json.load(open(path))["R"]
        rows = []
        for i, row in enumerate(R):
            vals = [other(f"demo {agent} agent, R[{i + 1}][{j + 1}]", f"{v:.3f}", path) for j, v in enumerate(row)]
            rows.append([f"after task {i + 1}"] + vals)
        tab(f"m{agent}", f"Accuracy matrix of the {lab} demo agent (seed 1): mean return on each task (columns) after "
                         "training each task (rows), 100 evaluation episodes per cell",
            ["", "Task 1 (red ball)", "Task 2 (green key)", "Task 3 (blue box)"], rows, [1.5, 1.4, 1.4, 1.4],
            sz=10.5, align=["left", "center", "center", "center"])

    # =========================================================================================== APPENDIX B
    B.append(("appendix", "B", "Coding"))
    p("Key source excerpts only, copied verbatim from the repository; the complete code is in the project "
      "repository. Each listing names the file it comes from.")
    listings = [
        ("Listing B.1: neuroplast/models/snn/lif.py (LIF neuron with a fast-sigmoid surrogate gradient)",
         snippet("neuroplast/models/snn/lif.py", r"^class FastSigmoidSpike")),
        ("Listing B.2: neuroplast/learning/stdp.py (pair-based STDP with eligibility, dense layers)",
         snippet("neuroplast/learning/stdp.py", r"^def traces", r"return dW / B if mean else dW")),
        ("Listing B.3: neuroplast/learning/hybrid.py (hybrid rule: STDP or random update, homeostasis)",
         snippet("neuroplast/learning/hybrid.py", r"    @torch.no_grad\(\)", r"self.homeo_lr \* \(self.homeo - rate\)")),
        ("Listing B.4: neuroplast/baselines/replay.py (CLEAR-style replay and the distillation loss)",
         snippet("neuroplast/baselines/replay.py", r"^def distill_loss")),
        ("Listing B.5: neuroplast/sleep/sleep.py (sleep phase)",
         snippet("neuroplast/sleep/sleep.py", r"    def sleep_phase", r"return dict\(sleep_loss")),
        ("Listing B.6: neuroplast/eval/metrics.py (continual-learning metrics)",
         snippet("neuroplast/eval/metrics.py", r"^def cl_metrics")),
        ("Listing B.7: scripts/memory_fair.py (memory accounting)",
         snippet("scripts/memory_fair.py", r"^OBS_B, FIXED_B", r"^    return groups")),
        ("Listing B.8: paper/compute_numbers.py (confidence intervals and tests)",
         snippet("paper/compute_numbers.py", r"^def welch\(a, b\)", r"stats.fisher_exact")),
    ]
    for cap, text in listings:
        code(cap, text, sz=8)

    B.append(("refs",))
    return B
