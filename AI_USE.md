AI Usage Documentation

1. Overview

ChatGPT was used as a development assistant during the DSC 180A Embodied AI drone simulation project. Its primary roles included technical explanations, coding assistance, debugging, experimental planning, and documentation support.

Human team members remained responsible for choosing research objectives, conducting experiments, reviewing output, and making development decisions.

2. AI-Assisted Tasks

Simulation Setup and Debugging

AI assistance was used to investigate issues related to Isaac Sim, the IRIS quadrotor, simulation physics, object configuration, and execution of Python scripts.

Drone Flight and Wind Experiments

AI assistance was used to discuss flight behavior, environmental disturbances, trajectory logging, and experimental design.

Human-executed experiments included baseline flight, steady-wind conditions, and gust-wind conditions.

Reinforcement Learning

AI assistance was used to explore reinforcement learning approaches, including PPO and associated Python libraries. This assistance does not imply that a successfully trained and validated drone policy has been completed.

Data Analysis

AI was used to discuss methods for analyzing recorded trajectories, drift, and altitude stability. Human review of experimental output informed subsequent development decisions.

3. Prompt Documentation

The following categories summarize how AI was used. Exact prompts and relevant AI-generated code should be documented below or linked through chat exports.

Area	Purpose
Isaac Sim setup	Installation, environment configuration, and troubleshooting
IRIS simulation	Drone setup, motion control, and physics behavior
Wind experiments	Disturbance implementation and experimental comparisons
Data collection	Logging timestamps, positions, and simulation results
Reinforcement learning	PPO training concepts and troubleshooting
Analysis	Evaluating trajectory deviation and altitude stability

Exact prompt records: To be added from original AI chat history. The categories above are summaries, not verbatim prompts.

4. Testing and Validation

* Baseline, steady-wind, and gust-wind simulations were executed.
* Simulation behavior and recorded outputs were reviewed.
* Unexpected behavior and technical problems were investigated.
* AI suggestions were considered during troubleshooting and iterative development.

Testing status varies by component. Experiments that ran successfully should be distinguished from proposed features, incomplete implementations, and unvalidated code.

5. Limitations and Next Steps

The project combines independently developed components from multiple team members. Full integration and consistent cross-component evaluation are ongoing.

AI assistance does not replace independent execution, verification, or critical evaluation of simulation results.





1. Human and AI contributions

ChatGPT assisted with understanding the Isaac Sim and IRIS simulation stack, planning implementation steps, suggesting possible coding approaches, troubleshooting Python and library issues, discussing baseline/steady-wind/gust-wind experiments, identifying possible sensor and data-logging workflows, exploring reinforcement learning with PPO, interpreting potential evaluation metrics, and editing project-related writing.

The student chose which experiments to attempt, installed and ran software, worked with a local codebase, executed and inspected simulations, provided errors and results for troubleshooting, and made decisions about next steps. Successful simulation runs do not by themselves establish successful PPO training or validated closed-loop drone control. Other teammates developed parallel components; their AI contributions and validation records are not included here unless explicitly added.

2. Original prompts recovered from accessible history

The entries below are grouped chronologically by task. Dates are chat dates or dates supported by the available record, not always precise prompt timestamps. Exact quotes preserve spelling and punctuation where available. Some longer messages are represented by a faithful visible excerpt, which is explicitly marked; an excerpt is not an exhaustive prompt record.

A. Project onboarding and initial Isaac Sim installation (October 14–19, 2026)

P01 — October 14 — Catch-up plan — EXACT

> Help me with understnding the progress of the work so far. what should i have completed and if i have done nothing, whats the steps to take to catch up fast?

Context: Project onboarding and assignment planning; screenshots and a course PDF were also supplied. AI role: explanation and task planning. Verification: student subsequently installed Isaac Sim and attempted project work; no specific generated code is attributable to this prompt without transcript review.

P02 — October 18 — Project-aligned PA3 work — EXACT

> Can i maybe do a little bit more today so i have something more to write about?

P03 — October 18 — Project alignment — EXACT

> can i maybe do something that is more related to the porject?

P04 — October 18 — Instructor/teammate guidance — EXACT

> can you look at what the TS said on week 2 and mybe we can work from there? and also what does colin advise?

Context: Uploaded instructor/teammate PDFs. AI role: interpreting requirements and planning experiments. Verification: compare suggestions with course instructions and teammate discussions.

P05 — October 18 — Simulation and training roadmap — EXACT

> hummm… im a little confused.  
> so i need to do 1. do a done simulation and try ot make it fly and stuff?  
> 2. train something???  
> can you make this into a good overview and how what I’m gonna do will connect with th eproject? then give me a step by step guide of how to complete them

P06 — October 18 — Environment debugging — EXACT

> I fail to open the post instaill. a window would pop and disappear right after when i open it. but i did run the

P07 — October 18 — Installation next steps — EXACT

> I reached here! after all the striggles and stuff. i just clicked issac sim .bat and it took a while but we;re here. whats’ next?

P08 — October 18 — Simplified wind experiment — SUMMARY / NOT VERBATIM

Requested step-by-step instructions for a simplified drone wind-gust simulation, starting with Isaac Sim installation. Original full wording unavailable.

P09 — October 18–19 — PA3 reflection — EXACT, accessible excerpt

> One interesting result I have found doing my task this week is learning more on how Issac sim works and how physics interacts with objects in the simulator. I discovered that there is all kinds of physical interaction with many objects and environments to be tested out in the simulator. I am also looking at how to better use this in our project which involves environmental forces such as wind’s effect on the drones and such. I found the physics in the simulator very interesting. The simulator is giving me an opportunity to think about why certain objects act certain ways. I learned to try to predict and internally analyze the forces that act on certain objects, taking into consideration gravity, friction, and collision.

Context: User-provided writing for feedback, not necessarily code generation. Further paragraphs and other versions exist in conversation history.

B. Local development environment and Git (October 18–24, 2026)

P10 — October 18 — Python interpreter — EXACT

> I can;t find Python: Select Interpreter tho

Context: Troubleshooting VS Code after validating imports for gymnasium and stable_baselines3.

P11 — October 21 — Local laptop setup — EXACT

> okay… step by step. how to prep my laptop. starting with the setup

P12 — October 21 — Remote Isaac Sim access — EXACT

> can i get issac sim on my mac? or at least let it run here somehow?

P13 — October 21 — Remote-access outcome — EXACT

> done. i can acess my pc on my mac now

P14 — October 23 — Work outside Git repository — EXACT

> Can I do my drone experiament at DSC180A, just don’t touch the github file? and if ineed things illl copy there

P15 — October 24 — Simulator code and environments — EXACT

> Hummm… May I ask, hwo to “code” issac sim? and how do they “code it? and furter more, what are the enviroenmnt files?

P16 — October 24 — Build simulation from scratch — EXACT

> Okay, step by step, make a simple WBS for me setting up my own simulation and coding them myself

C. IRIS, drone motion, wind disturbances, and sensors (October 24–November 3, 2026)

P17 — October 24–25 — Choose IRIS and point-to-point flight — EXACT

> actually, let’s just use iris for now. let’s continue. can we first make it fly from point a to b?

P18 — October 25 — Timestamp and physical control questions — SUMMARY / NOT VERBATIM

Requested timestamps for per-trial output files; clarification about motor/force-based flight versus directly moving or setting drone position; sensor data collection including IMU/RGB; longer flight distance; and wind simulation. Original full prompts not recovered.

P19 — October 25–26 — Wind direction and PPO — SUMMARY / NOT VERBATIM

Asked whether wind direction changed or gust magnitude alone varied, and requested PPO implementation/training. Original full prompts not recovered.

P20 — October 26 — PPO Python environment failures — SUMMARY / NOT VERBATIM

Shared error outputs involving blocked DLL loading while importing Python scientific libraries, and later a PyTorch import incompatibility. Exact surrounding user prompts not recovered. Recover original error message and assistant-generated fixes before claiming which code was adopted.

P21 — November 3 — Sensors and simulation scenarios — EXACT, recovered excerpt only

> I want to add sensors and collect data … a simple rbg camera and IMU

Context from available history: requested separate files for sensor readings versus environment/ground-truth values, multiple scene presets (empty/warehouse/outdoor), initially pausing wind, and a straight approximately 100 m flight at 5 m altitude. Only the quoted portion is verbatim.

Related tasks lacking exact prompt transcripts: steady-wind and gust-wind runs; camera angle/lighting; drift correction; altitude stability; body/rotor path checks; trajectory plotting and output filenames; whether the drone was physically controlled or teleported. Do not imply these are fully documented yet.

D. Project administration, research, and reporting (October 21–November 6, 2026)

P22 — October 21 — Repo setup/log outputs — EXACT, excerpt

> claire@Socrates drone_final_project1 %  
> conda activate dsc180a

Context: Sent terminal installation logs for NumPy, pandas, matplotlib, and ipykernel. This is an excerpt of a log-containing message, not a complete request or a generated-code prompt.

P23 — October 26 — Methodology assignment — EXACT

> I need to complete this assignment tonight, break down the assignment and the to-dos for me and assist me complete it in a timely manner

P24 — October 26 — Assignment improvement — EXACT

> what do you think about my assignmnet now? how can i fix it or improve it?

P25 — October 26 — Grammar and Q1/Q2 — EXACT

> can you help me fix the main problems and grammer mistake and give me a few suggestions to how i can rephrase q1 and q2?

P26 — October 26 — Critical feedback — EXACT

> can you help me with q1, i dont really know which points to attack, i think his videos is oretty solid

P27 — October 26 — Grammar only — EXACT

> For the other 3, can you fix my grammer for me and give me a copy pate ready of my work but dont’ alter the text, just fix the grammer and make it fluent

P28 — November 1 — Participation — EXACT

> Help me with the participation questions for this week

P29 — November 2 — Partner B Git steps — EXACT

> I already got. apartnet, he is on sttep 3, i am partner B, what should i do now

P30 — November 4 — Research paper analysis — EXACT

> analyze this paper that my group mate sent, tell me the main points and give me key ideas that this paper is trying to convey

Context: A PDF about SAC-assisted hybrid control for quadrotor path following under wind disturbances was supplied. AI role: paper explanation, not proof of implementing SAC.

P31 — November 5 — Team branch inspection and checkpoint — EXACT, excerpt

> this projects each persoon works on their own section, so the things i gave you is only from me and po-chen (2 members) I will uplaod the branches of the other 3 members so you can fully evalutie our work. However most of our work is completely parallel, so we really have no idea much about one another, we only put some pieces together during meetings, but there is still some misconmuications

P32 — November 5 — Checkpoint major flaws — EXACT

> yeah, based on these finisng, is there anythign major flaws with our currecnt checkpoint submission?

P33 — November 6 — GitHub repository status — EXACT

> can you take a look at the github, and more over, if im the only branch, how can i see my branch? if i am the only branch, can i merge the breanch with main so that its more united

P34 — November 6 — Project progress — EXACT

> https://github.com/randonDsc/drone_final_project1/tree/main  
> can you read the repo and tell me where exactly are we in terms of progress right now? (my team mates told me to just look

Note: The accessible original message ends mid-sentence; recover the full transcript.

P35 — November 6 — PPO status — EXACT

> I am doing osme work on PPO right? I thought I also trained something on it?

P36 — November 6 — Branch and teammate documentation — EXACT

> here is my branch, can you inspect it for me? Also here is the four explaing documents that my teammate sent me, can you read it as well. i think he mentioesmt about 4 wind levels or smt, can you tell me abotu that

P37 — November 6 — AI disclosure instructions — EXACT

> Can you help me complete this part? The hard part for me is not the explaining but the problem Is I only have my Phone with me and I am outside now How do I update a repo now???

P38 — November 6 — Creating the document — EXACT

> How to get AI use md ?

P39 — November 6 — Recover prompts — EXACT

> Where can I get all the prompts I have?

P40 — November 6 — Compile prompt log — EXACT

> Please collect all my accessible original prompts related to DSC 180A, organize them chronologically by task, and prepare a copy-paste-ready AI_USE.md file. Preserve exact wording when available and mark any missing conversations or prompts.

3. AI-assisted work: testing and validation inventory

|Workstream            |Known student activity                                                                     |Evidence/verification still needed                                                                      |
|----------------------|-------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------|
|Isaac Sim setup       |Installed and launched Isaac Sim; explored physics and a test scene                        |Commands, version, console output, simulation screenshots                                               |
|IRIS flight           |Attempted or executed point-to-point quadrotor motion                                      |Code path, control type (motor/force-based vs kinematic), actual route and log                          |
|Wind experiments      |Baseline, steady-wind, and gust-wind trials were run/reviewed                              |Exact wind parameters, seeds, timestamps, data paths, plots, quantitative comparisons                   |
|Sensors               |Discussed IMU and RGB camera additions, plus ground-truth logging                          |Confirm that code was implemented and data were recorded, not merely planned                            |
|Reinforcement learning|Explored PPO, attempted environment setup and training                                     |Actual training logs, saved policy checkpoints, reward curves; do not claim trained policy without these|
|Research / writing    |Requested paper explanations, assignment feedback, grammar edits, and checkpoint assessment|Identify any AI-authored sentences adopted, cite original research papers independently                 |
|Team integration      |Separate teammate branches and simulators were inspected/discussed                         |Each teammate’s own AI prompts, generated code, edits and tests                                         |



