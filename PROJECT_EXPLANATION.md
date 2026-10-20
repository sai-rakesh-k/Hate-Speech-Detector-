# Project Explanation: Context-Aware Hate Speech Detection with Structured Evidence-Based Reasoning

This project is a full-stack research prototype for detecting hate speech in context and explaining the result in a structured, auditable way. The core idea is:

- The model reads the current comment and optional previous context.
- It predicts whether the comment is hate / offensive / normal.
- It identifies the target group (for example: nationality, religion, race, etc.).
- It may predict a reason category such as insult, exclusion, threat, dehumanization, etc.
- It extracts attribution-based evidence spans that explain which words mattered.
- It builds a deterministic explanation from the real model output instead of generating a free-form story.

This is described at a high level in [README.md](README.md), with deeper technical details in [REASONING.md](REASONING.md), [backend/README.md](backend/README.md), and [frontend/README.md](frontend/README.md).

---

## 1. High-level architecture

The project is split into two major parts:

1. Backend: Python, FastAPI, PyTorch, Hugging Face Transformers
   - model code
   - training and evaluation
   - inference and API endpoints
   - explanation logic and evidence extraction

2. Frontend: React + TypeScript + Vite
   - UI for analyzing a comment
   - evidence highlighting
   - reason explanation display
   - annotations and history

The overall pipeline is:

User input (comment + optional context)
→ BERT encoder
→ context interaction/fusion
→ hate/target/reason heads
→ evidence extraction via Integrated Gradients
→ deterministic structured explanation
→ JSON response
→ UI rendering

This is implemented across the backend model, reasoning, API, and frontend analysis components.

---

## 2. Repository structure

At the root, the project looks like this:

- [README.md](README.md) — main overview and quick start
- [REASONING.md](REASONING.md) — conceptual explanation of reasoning and honesty rules
- [backend/](backend/) — all server-side ML code and API
- [frontend/](frontend/) — all React/TypeScript UI
- [deliverables/](deliverables/) — build and verification reports
- [backend/checkpoints/](backend/checkpoints/) — model checkpoints (usually git-ignored)
- [backend/data/](backend/data/) — raw and processed data (usually git-ignored)
- [backend/experiments/](backend/experiments/) — experiment outputs and ablation results

---

## 3. Root files

### [README.md](README.md)
This is the project landing page. It explains:

- the purpose of the project
- what the app does
- quick start instructions
- where to find backend/frontend docs
- the overall architecture
- what “structured reasoning” means in this project

It is the best place to start before reading deeper files.

### [REASONING.md](REASONING.md)
This is the conceptual guide for the system’s “reasoning”.

It explains:

- what “reasoning” means here
- the staged pipeline
- how evidence is extracted
- how context is analyzed
- how reason categories are explained
- what the system intentionally does not do

This is not implementation code; it explains the logic and honesty constraints.

---

## 4. Backend: purpose and structure

The backend is the true engine of the project. It contains the model architecture, training logic, deterministic reasoning components, and API endpoints.

### [backend/app/__init__.py](backend/app/__init__.py)
This defines the package-level metadata for the backend, including the version string and a short project description.

### [backend/app/main.py](backend/app/main.py)
This is the FastAPI application entrypoint.

It usually:

- creates the app
- sets startup/shutdown behavior
- loads the model service
- includes routers for health, prediction, model info, and explanation endpoints

This is the central server startup file.

---

## 5. Backend configuration and runtime setup

### [backend/app/core/config.py](backend/app/core/config.py)
This is one of the most important backend files.

It defines the application settings using environment variables and `.env` values. It includes:

- model name
- checkpoint path
- device selection
- label settings
- evidence method
- evidence top-k and thresholds
- optimization settings
- loss weights
- contrastive settings
- random seed
- CORS settings

This file is where runtime behavior is configured without hardcoding values in model code.

### [backend/app/core/logging.py](backend/app/core/logging.py)
This sets up the Python logging configuration used across the project.

---

## 6. Backend API layer

### [backend/app/api/routes/](backend/app/api/routes/)
This folder contains API endpoint definitions.

Important files:

- [backend/app/api/routes/health.py](backend/app/api/routes/health.py) — liveness checks
- [backend/app/api/routes/prediction.py](backend/app/api/routes/prediction.py) — main prediction endpoint
- [backend/app/api/routes/explanation.py](backend/app/api/routes/explanation.py) — token-level attribution/explanation endpoint
- [backend/app/api/routes/model.py](backend/app/api/routes/model.py) — model metadata endpoint

These endpoints are the contract between the frontend and the backend.

### [backend/app/api/schemas/](backend/app/api/schemas/)
This contains Pydantic request/response models.

Examples:

- [backend/app/api/schemas/prediction.py](backend/app/api/schemas/prediction.py) — prediction schema with labels, evidence, reason explanation, etc.
- [backend/app/api/schemas/explanation.py](backend/app/api/schemas/explanation.py) — token explanation schema for attribution requests

The frontend types in [frontend/src/types/analysis.ts](frontend/src/types/analysis.ts) mirror these schemas.

---

## 7. Backend services

### [backend/app/services/inference.py](backend/app/services/inference.py)
This is the core service that handles actual inference.

It:

- loads the model bundle
- forwards inputs
- decodes predictions
- extracts evidence
- builds structured explanations
- provides model status and explain API behavior

This file is central to the project. It is where the “predict” and “explain” logic lives.

Key responsibilities:

- model loading lifecycle
- single forward pass
- hate/target/reason prediction decoding
- evidence extraction
- reason explanation assembly
- model info reporting
- availability checks for untrained heads

This is the file that turns raw model outputs into the structured API response shown in the UI.

---

## 8. Model architecture

### [backend/app/models/](backend/app/models/)
This is the neural model implementation.

Examples:

- [backend/app/models/full_model.py](backend/app/models/full_model.py)  
  Main full architecture for the context-aware multi-task model

- [backend/app/models/bert_encoder.py](backend/app/models/bert_encoder.py)  
  Shared encoder for the comment and context

- [backend/app/models/context_interaction.py](backend/app/models/context_interaction.py)  
  Combines current and contextual embeddings

- [backend/app/models/contrastive.py](backend/app/models/contrastive.py)  
  Training-only contrastive module

- [backend/app/models/hate_head.py](backend/app/models/hate_head.py)  
  Hate/offensive/normal classifier head

- [backend/app/models/target_head.py](backend/app/models/target_head.py)  
  Target group classification head

- [backend/app/models/reason_head.py](backend/app/models/reason_head.py)  
  Reason category head (insult, threat, exclusion, etc.)

- [backend/app/models/evidence_head.py](backend/app/models/evidence_head.py)  
  Token-level or span-level evidence head

These files define how the model is built and what each head contributes.

The README and architecture docs make it clear: the model is not a generative LLM. It is a supervised Transformer model with task-specific heads.

---

## 9. Reasoning and explanation logic

This is the part that makes the project interesting.

### [backend/app/reasoning/](backend/app/reasoning/)
This folder contains the attribution and explanation pipeline.

Key files:

- [backend/app/reasoning/attribution.py](backend/app/reasoning/attribution.py)  
  Integrated Gradients and other attribution methods

- [backend/app/reasoning/evidence_extractor.py](backend/app/reasoning/evidence_extractor.py)  
  Converts attribution scores into human-readable evidence spans and offsets

- [backend/app/reasoning/reason_explainer.py](backend/app/reasoning/reason_explainer.py)  
  Deterministic templates that connect the predicted reason to the extracted evidence

- [backend/app/reasoning/context_reasoner.py](backend/app/reasoning/context_reasoner.py)  
  Heuristic analysis of how the previous comment relates to the current one

- [backend/app/reasoning/structured_explanation.py](backend/app/reasoning/structured_explanation.py)  
  Assembles the final structured output object

These modules are designed to be auditable and grounded:
- no free-form generative story
- no invented intent
- no fake certainty
- only real model outputs, evidence, and fixed templates

This is a central honesty rule in the project.

---

## 10. Datasets and data handling

### [backend/app/datasets/](backend/app/datasets/)
This folder contains the dataset adapters and data preprocessing logic.

Key files:

- [backend/app/datasets/hatexplain.py](backend/app/datasets/hatexplain.py)  
  Adapter for the HateXplain dataset

- [backend/app/datasets/counter_context.py](backend/app/datasets/counter_context.py)  
  Adapter for context-aware counter-context dataset

- [backend/app/datasets/unified.py](backend/app/datasets/unified.py)  
  Unified internal format used by the training pipeline

- [backend/app/datasets/preprocessing.py](backend/app/datasets/preprocessing.py)  
  Data cleaning / conversion routines

- [backend/app/datasets/tokenizer.py](backend/app/datasets/tokenizer.py)  
  Tokenization, offsets, and span alignment

- [backend/app/datasets/collator.py](backend/app/datasets/collator.py)  
  Data batching for training

The project explicitly avoids blindly merging datasets. The documentation stresses separate adapters and explicit `None` handling for missing labels.

---

## 11. Training and evaluation

### [backend/app/training/](backend/app/training/)
This folder holds training utilities.

Examples:

- trainer logic
- loss functions
- contrastive loss
- warmup logic
- checkpoint saving
- experiment config handling

The project is designed to train multiple configurations such as:

- baseline
- context-aware
- full multi-task
- reason-augmented

### [backend/app/evaluation/](backend/app/evaluation/)
This contains evaluation utilities, metrics, and ablation code.

Important file:

- [backend/app/evaluation/ablation.py](backend/app/evaluation/ablation.py)  
  Defines experiment variants, such as baseline vs context vs full multi-task vs evidence evaluation

The project treats evaluation as evidence-based and honest — it reports actual numbers and does not claim general improvements without a proper experiment.

---

## 12. Scripts

### [backend/scripts/](backend/scripts/)
This folder is the operational interface for development and experimentation.

Examples:

- [backend/scripts/download_data.py](backend/scripts/download_data.py)  
  Downloads raw datasets

- [backend/scripts/preprocess.py](backend/scripts/preprocess.py)  
  Prepares data for training

- [backend/scripts/train.py](backend/scripts/train.py)  
  Trains a configured model

- [backend/scripts/evaluate.py](backend/scripts/evaluate.py)  
  Runs evaluation on a checkpoint

- [backend/scripts/explain.py](backend/scripts/explain.py)  
  Runs structured prediction/explanation for a single example

- [backend/scripts/import_reason_annotations.py](backend/scripts/import_reason_annotations.py)  
  Imports reason annotations into the training dataset

- [backend/scripts/inspect_dataset.py](backend/scripts/inspect_dataset.py)  
  Verifies actual dataset schema and label shape before training

These scripts are what turn the research code into usable workflows.

---

## 13. Configurations

### [backend/configs/](backend/configs/)
This folder stores YAML configuration files for experiments and model setup.

Examples include:

- baseline config
- context-aware config
- full config
- counter-context config
- reason-augmented config

This is the project’s configuration layer; model classes themselves are not hardcoded with hyperparameters.

---

## 14. Data and outputs folders

### [backend/data/](backend/data/)
This holds:
- raw dataset downloads
- preprocessed dataset files
- annotation data
- possibly training-ready prepared files

The project usually keeps this out of Git because it can be large.

### [backend/checkpoints/](backend/checkpoints/)
This stores saved model weights.

Examples:
- `best.pt`
- `last.pt`
- training snapshots for different runs

### [backend/experiments/](backend/experiments/)
This stores ablation and evaluation outputs, often as JSON and CSV logs.

---

## 15. Tests

### [backend/tests/](backend/tests/)
The backend test suite covers:

- model configuration
- training paths
- inference service behavior
- untrained model handling
- trained checkpoint prediction
- reason explanation generation
- API contract tests
- context reasoning logic

Examples:

- [backend/tests/test_inference_service.py](backend/tests/test_inference_service.py)  
  Covers inference service and explanation availability

- [backend/tests/test_context_reasoner.py](backend/tests/test_context_reasoner.py)  
  Covers context-to-current-comment relationship analysis

These tests are designed to verify the system’s honesty and deterministic behavior.

---

## 16. Frontend structure

The frontend is the user-facing analysis and annotation interface.

### [frontend/src/main.tsx](frontend/src/main.tsx)
This is the React entrypoint that mounts the app.

### [frontend/src/api/analysis.ts](frontend/src/api/analysis.ts)
This is the only HTTP layer for the frontend. It wraps calls to:

- `/health`
- `/api/v1/model/info`
- `/api/v1/predict`
- `/api/v1/explain`

This keeps API calls centralized and avoids scattershot axios code.

### [frontend/src/types/analysis.ts](frontend/src/types/analysis.ts)
This defines the TS types that match the backend API schema, including:
- prediction result
- evidence items
- reason explanation
- model info
- health status

### [frontend/src/pages/](frontend/src/pages/)
This contains route-level pages:

- [frontend/src/pages/AnalyzePage.tsx](frontend/src/pages/AnalyzePage.tsx)  
  Main analysis interface

- [frontend/src/pages/AnnotationPage.tsx](frontend/src/pages/AnnotationPage.tsx)  
  Annotation workspace for reason labeling

- [frontend/src/pages/HistoryPage.tsx](frontend/src/pages/HistoryPage.tsx)  
  Previously analyzed results

- [frontend/src/pages/AboutPage.tsx](frontend/src/pages/AboutPage.tsx)  
  Explanation of the architecture and caveats

### [frontend/src/components/analysis/](frontend/src/components/analysis/)
This is the main UI system for model results.

Examples:

- [frontend/src/components/analysis/PredictionCard.tsx](frontend/src/components/analysis/PredictionCard.tsx)  
  Shows the predicted class

- [frontend/src/components/analysis/TargetCard.tsx](frontend/src/components/analysis/TargetCard.tsx)  
  Shows the target group

- [frontend/src/components/analysis/ReasonCard.tsx](frontend/src/components/analysis/ReasonCard.tsx)  
  Shows the predicted reason

- [frontend/src/components/analysis/EvidenceViewer.tsx](frontend/src/components/analysis/EvidenceViewer.tsx)  
  Highlights relevant spans in the comment text

- [frontend/src/components/analysis/EvidenceList.tsx](frontend/src/components/analysis/EvidenceList.tsx)  
  Ranks evidence items in a list

- [frontend/src/components/analysis/ReasonExplanationCard.tsx](frontend/src/components/analysis/ReasonExplanationCard.tsx)  
  Displays the structured explanation and “Why this classification?” section

- [frontend/src/components/analysis/AttributionDetail.tsx](frontend/src/components/analysis/AttributionDetail.tsx)  
  Shows token-level attribution scores

- [frontend/src/components/analysis/ContextViewer.tsx](frontend/src/components/analysis/ContextViewer.tsx)  
  Shows whether context was included and how it affected the decision

- [frontend/src/components/analysis/ModelArchitecture.tsx](frontend/src/components/analysis/ModelArchitecture.tsx)  
  Explains the model architecture to the user

### [frontend/src/utils/](frontend/src/utils/)
This contains utility logic such as:

- evidence matching and highlight ranking
- formatting utilities
- demo data
- label formatting

Examples:

- [frontend/src/utils/evidence.ts](frontend/src/utils/evidence.ts)
- [frontend/src/utils/formatting.ts](frontend/src/utils/formatting.ts)
- [frontend/src/utils/demoExamples.ts](frontend/src/utils/demoExamples.ts)

### [frontend/src/store/](frontend/src/store/)
This contains state management for forms, results, and history.

---

## 17. How the app actually flows at runtime

Here is the flow from a user action to rendered output:

1. User enters a comment and optional context in the frontend.
2. Frontend calls the backend prediction API via [frontend/src/api/analysis.ts](frontend/src/api/analysis.ts).
3. Backend loads or uses the model service from [backend/app/services/inference.py](backend/app/services/inference.py).
4. The model produces:
   - hate label
   - target label
   - reason label
   - evidence spans
   - context flag
5. Evidence extraction uses attribution scores computed by the reasoning pipeline.
6. The system builds a deterministic reason explanation from the model outputs.
7. Frontend renders:
   - prediction card
   - target card
   - reason card
   - evidence highlighting
   - reason explanation
   - context reasoning section
   - attribution detail (on demand)

This is a clean “structured AI” pipeline rather than a free-form generative explanation.

---

## 18. The project’s honesty rules

This project is intentionally strict about what it does and does not claim.

From the docs and code comments:

- evidence is attribution-based, not human-verified causal proof
- reasons are not generated by an LLM
- explanations are deterministic template composition
- explanation is grounded only in:
  - predicted reason
  - predicted target
  - extracted evidence
- context is reported as included, not as proven cause
- missing or untrained heads are explicitly marked unavailable
- unavailable sections are hidden or marked “not available” instead of faking output

This is the core technical and ethical design decision behind the project.

---

## 19. Most important files to read in order

If you are new to the project, this is the recommended reading order:

1. [README.md](README.md)
2. [REASONING.md](REASONING.md)
3. [backend/README.md](backend/README.md)
4. [backend/app/main.py](backend/app/main.py)
5. [backend/app/services/inference.py](backend/app/services/inference.py)
6. [backend/app/reasoning/reason_explainer.py](backend/app/reasoning/reason_explainer.py)
7. [backend/app/reasoning/evidence_extractor.py](backend/app/reasoning/evidence_extractor.py)
8. [backend/app/models/full_model.py](backend/app/models/full_model.py)
9. [frontend/src/pages/AnalyzePage.tsx](frontend/src/pages/AnalyzePage.tsx)
10. [frontend/src/components/analysis/ReasonExplanationCard.tsx](frontend/src/components/analysis/ReasonExplanationCard.tsx)

This gives a practical understanding of both architecture and behavior.

---

## 20. Summary

This project is best understood as a research prototype for explainable hate speech detection with context awareness.

It combines:

- Transformer-based classification
- context-aware modeling
- multi-task prediction heads
- attribution-based evidence extraction
- deterministic explanation generation
- structured API responses
- polished React UI

The most important thing is that the “reasoning” is not generated text. It is assembled from real model outputs and fixed templates. That is the project’s defining principle.

If you want, I can also create a second file that is more like:
- a “folder-by-folder developer map”
- a “beginner-friendly architecture guide”
- or a “technical deep dive with dependencies and flow charts” version.// filepath: c:\Users\Rakesh\OneDrive\Documents\AIML_HATE_SPEECH_DETECTOR\PROJECT_EXPLANATION.md
# Project Explanation: Context-Aware Hate Speech Detection with Structured Evidence-Based Reasoning

This project is a full-stack research prototype for detecting hate speech in context and explaining the result in a structured, auditable way. The core idea is:

- The model reads the current comment and optional previous context.
- It predicts whether the comment is hate / offensive / normal.
- It identifies the target group (for example: nationality, religion, race, etc.).
- It may predict a reason category such as insult, exclusion, threat, dehumanization, etc.
- It extracts attribution-based evidence spans that explain which words mattered.
- It builds a deterministic explanation from the real model output instead of generating a free-form story.

This is described at a high level in [README.md](README.md), with deeper technical details in [REASONING.md](REASONING.md), [backend/README.md](backend/README.md), and [frontend/README.md](frontend/README.md).

---

## 1. High-level architecture

The project is split into two major parts:

1. Backend: Python, FastAPI, PyTorch, Hugging Face Transformers
   - model code
   - training and evaluation
   - inference and API endpoints
   - explanation logic and evidence extraction

2. Frontend: React + TypeScript + Vite
   - UI for analyzing a comment
   - evidence highlighting
   - reason explanation display
   - annotations and history

The overall pipeline is:

User input (comment + optional context)
→ BERT encoder
→ context interaction/fusion
→ hate/target/reason heads
→ evidence extraction via Integrated Gradients
→ deterministic structured explanation
→ JSON response
→ UI rendering

This is implemented across the backend model, reasoning, API, and frontend analysis components.

---

## 2. Repository structure

At the root, the project looks like this:

- [README.md](README.md) — main overview and quick start
- [REASONING.md](REASONING.md) — conceptual explanation of reasoning and honesty rules
- [backend/](backend/) — all server-side ML code and API
- [frontend/](frontend/) — all React/TypeScript UI
- [deliverables/](deliverables/) — build and verification reports
- [backend/checkpoints/](backend/checkpoints/) — model checkpoints (usually git-ignored)
- [backend/data/](backend/data/) — raw and processed data (usually git-ignored)
- [backend/experiments/](backend/experiments/) — experiment outputs and ablation results

---

## 3. Root files

### [README.md](README.md)
This is the project landing page. It explains:

- the purpose of the project
- what the app does
- quick start instructions
- where to find backend/frontend docs
- the overall architecture
- what “structured reasoning” means in this project

It is the best place to start before reading deeper files.

### [REASONING.md](REASONING.md)
This is the conceptual guide for the system’s “reasoning”.

It explains:

- what “reasoning” means here
- the staged pipeline
- how evidence is extracted
- how context is analyzed
- how reason categories are explained
- what the system intentionally does not do

This is not implementation code; it explains the logic and honesty constraints.

---

## 4. Backend: purpose and structure

The backend is the true engine of the project. It contains the model architecture, training logic, deterministic reasoning components, and API endpoints.

### [backend/app/__init__.py](backend/app/__init__.py)
This defines the package-level metadata for the backend, including the version string and a short project description.

### [backend/app/main.py](backend/app/main.py)
This is the FastAPI application entrypoint.

It usually:

- creates the app
- sets startup/shutdown behavior
- loads the model service
- includes routers for health, prediction, model info, and explanation endpoints

This is the central server startup file.

---

## 5. Backend configuration and runtime setup

### [backend/app/core/config.py](backend/app/core/config.py)
This is one of the most important backend files.

It defines the application settings using environment variables and `.env` values. It includes:

- model name
- checkpoint path
- device selection
- label settings
- evidence method
- evidence top-k and thresholds
- optimization settings
- loss weights
- contrastive settings
- random seed
- CORS settings

This file is where runtime behavior is configured without hardcoding values in model code.

### [backend/app/core/logging.py](backend/app/core/logging.py)
This sets up the Python logging configuration used across the project.

---

## 6. Backend API layer

### [backend/app/api/routes/](backend/app/api/routes/)
This folder contains API endpoint definitions.

Important files:

- [backend/app/api/routes/health.py](backend/app/api/routes/health.py) — liveness checks
- [backend/app/api/routes/prediction.py](backend/app/api/routes/prediction.py) — main prediction endpoint
- [backend/app/api/routes/explanation.py](backend/app/api/routes/explanation.py) — token-level attribution/explanation endpoint
- [backend/app/api/routes/model.py](backend/app/api/routes/model.py) — model metadata endpoint

These endpoints are the contract between the frontend and the backend.

### [backend/app/api/schemas/](backend/app/api/schemas/)
This contains Pydantic request/response models.

Examples:

- [backend/app/api/schemas/prediction.py](backend/app/api/schemas/prediction.py) — prediction schema with labels, evidence, reason explanation, etc.
- [backend/app/api/schemas/explanation.py](backend/app/api/schemas/explanation.py) — token explanation schema for attribution requests

The frontend types in [frontend/src/types/analysis.ts](frontend/src/types/analysis.ts) mirror these schemas.

---

## 7. Backend services

### [backend/app/services/inference.py](backend/app/services/inference.py)
This is the core service that handles actual inference.

It:

- loads the model bundle
- forwards inputs
- decodes predictions
- extracts evidence
- builds structured explanations
- provides model status and explain API behavior

This file is central to the project. It is where the “predict” and “explain” logic lives.

Key responsibilities:

- model loading lifecycle
- single forward pass
- hate/target/reason prediction decoding
- evidence extraction
- reason explanation assembly
- model info reporting
- availability checks for untrained heads

This is the file that turns raw model outputs into the structured API response shown in the UI.

---

## 8. Model architecture

### [backend/app/models/](backend/app/models/)
This is the neural model implementation.

Examples:

- [backend/app/models/full_model.py](backend/app/models/full_model.py)  
  Main full architecture for the context-aware multi-task model

- [backend/app/models/bert_encoder.py](backend/app/models/bert_encoder.py)  
  Shared encoder for the comment and context

- [backend/app/models/context_interaction.py](backend/app/models/context_interaction.py)  
  Combines current and contextual embeddings

- [backend/app/models/contrastive.py](backend/app/models/contrastive.py)  
  Training-only contrastive module

- [backend/app/models/hate_head.py](backend/app/models/hate_head.py)  
  Hate/offensive/normal classifier head

- [backend/app/models/target_head.py](backend/app/models/target_head.py)  
  Target group classification head

- [backend/app/models/reason_head.py](backend/app/models/reason_head.py)  
  Reason category head (insult, threat, exclusion, etc.)

- [backend/app/models/evidence_head.py](backend/app/models/evidence_head.py)  
  Token-level or span-level evidence head

These files define how the model is built and what each head contributes.

The README and architecture docs make it clear: the model is not a generative LLM. It is a supervised Transformer model with task-specific heads.

---

## 9. Reasoning and explanation logic

This is the part that makes the project interesting.

### [backend/app/reasoning/](backend/app/reasoning/)
This folder contains the attribution and explanation pipeline.

Key files:

- [backend/app/reasoning/attribution.py](backend/app/reasoning/attribution.py)  
  Integrated Gradients and other attribution methods

- [backend/app/reasoning/evidence_extractor.py](backend/app/reasoning/evidence_extractor.py)  
  Converts attribution scores into human-readable evidence spans and offsets

- [backend/app/reasoning/reason_explainer.py](backend/app/reasoning/reason_explainer.py)  
  Deterministic templates that connect the predicted reason to the extracted evidence

- [backend/app/reasoning/context_reasoner.py](backend/app/reasoning/context_reasoner.py)  
  Heuristic analysis of how the previous comment relates to the current one

- [backend/app/reasoning/structured_explanation.py](backend/app/reasoning/structured_explanation.py)  
  Assembles the final structured output object

These modules are designed to be auditable and grounded:
- no free-form generative story
- no invented intent
- no fake certainty
- only real model outputs, evidence, and fixed templates

This is a central honesty rule in the project.

---

## 10. Datasets and data handling

### [backend/app/datasets/](backend/app/datasets/)
This folder contains the dataset adapters and data preprocessing logic.

Key files:

- [backend/app/datasets/hatexplain.py](backend/app/datasets/hatexplain.py)  
  Adapter for the HateXplain dataset

- [backend/app/datasets/counter_context.py](backend/app/datasets/counter_context.py)  
  Adapter for context-aware counter-context dataset

- [backend/app/datasets/unified.py](backend/app/datasets/unified.py)  
  Unified internal format used by the training pipeline

- [backend/app/datasets/preprocessing.py](backend/app/datasets/preprocessing.py)  
  Data cleaning / conversion routines

- [backend/app/datasets/tokenizer.py](backend/app/datasets/tokenizer.py)  
  Tokenization, offsets, and span alignment

- [backend/app/datasets/collator.py](backend/app/datasets/collator.py)  
  Data batching for training

The project explicitly avoids blindly merging datasets. The documentation stresses separate adapters and explicit `None` handling for missing labels.

---

## 11. Training and evaluation

### [backend/app/training/](backend/app/training/)
This folder holds training utilities.

Examples:

- trainer logic
- loss functions
- contrastive loss
- warmup logic
- checkpoint saving
- experiment config handling

The project is designed to train multiple configurations such as:

- baseline
- context-aware
- full multi-task
- reason-augmented

### [backend/app/evaluation/](backend/app/evaluation/)
This contains evaluation utilities, metrics, and ablation code.

Important file:

- [backend/app/evaluation/ablation.py](backend/app/evaluation/ablation.py)  
  Defines experiment variants, such as baseline vs context vs full multi-task vs evidence evaluation

The project treats evaluation as evidence-based and honest — it reports actual numbers and does not claim general improvements without a proper experiment.

---

## 12. Scripts

### [backend/scripts/](backend/scripts/)
This folder is the operational interface for development and experimentation.

Examples:

- [backend/scripts/download_data.py](backend/scripts/download_data.py)  
  Downloads raw datasets

- [backend/scripts/preprocess.py](backend/scripts/preprocess.py)  
  Prepares data for training

- [backend/scripts/train.py](backend/scripts/train.py)  
  Trains a configured model

- [backend/scripts/evaluate.py](backend/scripts/evaluate.py)  
  Runs evaluation on a checkpoint

- [backend/scripts/explain.py](backend/scripts/explain.py)  
  Runs structured prediction/explanation for a single example

- [backend/scripts/import_reason_annotations.py](backend/scripts/import_reason_annotations.py)  
  Imports reason annotations into the training dataset

- [backend/scripts/inspect_dataset.py](backend/scripts/inspect_dataset.py)  
  Verifies actual dataset schema and label shape before training

These scripts are what turn the research code into usable workflows.

---

## 13. Configurations

### [backend/configs/](backend/configs/)
This folder stores YAML configuration files for experiments and model setup.

Examples include:

- baseline config
- context-aware config
- full config
- counter-context config
- reason-augmented config

This is the project’s configuration layer; model classes themselves are not hardcoded with hyperparameters.

---

## 14. Data and outputs folders

### [backend/data/](backend/data/)
This holds:
- raw dataset downloads
- preprocessed dataset files
- annotation data
- possibly training-ready prepared files

The project usually keeps this out of Git because it can be large.

### [backend/checkpoints/](backend/checkpoints/)
This stores saved model weights.

Examples:
- `best.pt`
- `last.pt`
- training snapshots for different runs

### [backend/experiments/](backend/experiments/)
This stores ablation and evaluation outputs, often as JSON and CSV logs.

---

## 15. Tests

### [backend/tests/](backend/tests/)
The backend test suite covers:

- model configuration
- training paths
- inference service behavior
- untrained model handling
- trained checkpoint prediction
- reason explanation generation
- API contract tests
- context reasoning logic

Examples:

- [backend/tests/test_inference_service.py](backend/tests/test_inference_service.py)  
  Covers inference service and explanation availability

- [backend/tests/test_context_reasoner.py](backend/tests/test_context_reasoner.py)  
  Covers context-to-current-comment relationship analysis

These tests are designed to verify the system’s honesty and deterministic behavior.

---

## 16. Frontend structure

The frontend is the user-facing analysis and annotation interface.

### [frontend/src/main.tsx](frontend/src/main.tsx)
This is the React entrypoint that mounts the app.

### [frontend/src/api/analysis.ts](frontend/src/api/analysis.ts)
This is the only HTTP layer for the frontend. It wraps calls to:

- `/health`
- `/api/v1/model/info`
- `/api/v1/predict`
- `/api/v1/explain`

This keeps API calls centralized and avoids scattershot axios code.

### [frontend/src/types/analysis.ts](frontend/src/types/analysis.ts)
This defines the TS types that match the backend API schema, including:
- prediction result
- evidence items
- reason explanation
- model info
- health status

### [frontend/src/pages/](frontend/src/pages/)
This contains route-level pages:

- [frontend/src/pages/AnalyzePage.tsx](frontend/src/pages/AnalyzePage.tsx)  
  Main analysis interface

- [frontend/src/pages/AnnotationPage.tsx](frontend/src/pages/AnnotationPage.tsx)  
  Annotation workspace for reason labeling

- [frontend/src/pages/HistoryPage.tsx](frontend/src/pages/HistoryPage.tsx)  
  Previously analyzed results

- [frontend/src/pages/AboutPage.tsx](frontend/src/pages/AboutPage.tsx)  
  Explanation of the architecture and caveats

### [frontend/src/components/analysis/](frontend/src/components/analysis/)
This is the main UI system for model results.

Examples:

- [frontend/src/components/analysis/PredictionCard.tsx](frontend/src/components/analysis/PredictionCard.tsx)  
  Shows the predicted class

- [frontend/src/components/analysis/TargetCard.tsx](frontend/src/components/analysis/TargetCard.tsx)  
  Shows the target group

- [frontend/src/components/analysis/ReasonCard.tsx](frontend/src/components/analysis/ReasonCard.tsx)  
  Shows the predicted reason

- [frontend/src/components/analysis/EvidenceViewer.tsx](frontend/src/components/analysis/EvidenceViewer.tsx)  
  Highlights relevant spans in the comment text

- [frontend/src/components/analysis/EvidenceList.tsx](frontend/src/components/analysis/EvidenceList.tsx)  
  Ranks evidence items in a list

- [frontend/src/components/analysis/ReasonExplanationCard.tsx](frontend/src/components/analysis/ReasonExplanationCard.tsx)  
  Displays the structured explanation and “Why this classification?” section

- [frontend/src/components/analysis/AttributionDetail.tsx](frontend/src/components/analysis/AttributionDetail.tsx)  
  Shows token-level attribution scores

- [frontend/src/components/analysis/ContextViewer.tsx](frontend/src/components/analysis/ContextViewer.tsx)  
  Shows whether context was included and how it affected the decision

- [frontend/src/components/analysis/ModelArchitecture.tsx](frontend/src/components/analysis/ModelArchitecture.tsx)  
  Explains the model architecture to the user

### [frontend/src/utils/](frontend/src/utils/)
This contains utility logic such as:

- evidence matching and highlight ranking
- formatting utilities
- demo data
- label formatting

Examples:

- [frontend/src/utils/evidence.ts](frontend/src/utils/evidence.ts)
- [frontend/src/utils/formatting.ts](frontend/src/utils/formatting.ts)
- [frontend/src/utils/demoExamples.ts](frontend/src/utils/demoExamples.ts)

### [frontend/src/store/](frontend/src/store/)
This contains state management for forms, results, and history.

---

## 17. How the app actually flows at runtime

Here is the flow from a user action to rendered output:

1. User enters a comment and optional context in the frontend.
2. Frontend calls the backend prediction API via [frontend/src/api/analysis.ts](frontend/src/api/analysis.ts).
3. Backend loads or uses the model service from [backend/app/services/inference.py](backend/app/services/inference.py).
4. The model produces:
   - hate label
   - target label
   - reason label
   - evidence spans
   - context flag
5. Evidence extraction uses attribution scores computed by the reasoning pipeline.
6. The system builds a deterministic reason explanation from the model outputs.
7. Frontend renders:
   - prediction card
   - target card
   - reason card
   - evidence highlighting
   - reason explanation
   - context reasoning section
   - attribution detail (on demand)

This is a clean “structured AI” pipeline rather than a free-form generative explanation.

---

## 18. The project’s honesty rules

This project is intentionally strict about what it does and does not claim.

From the docs and code comments:

- evidence is attribution-based, not human-verified causal proof
- reasons are not generated by an LLM
- explanations are deterministic template composition
- explanation is grounded only in:
  - predicted reason
  - predicted target
  - extracted evidence
- context is reported as included, not as proven cause
- missing or untrained heads are explicitly marked unavailable
- unavailable sections are hidden or marked “not available” instead of faking output

This is the core technical and ethical design decision behind the project.

---

## 19. Most important files to read in order

If you are new to the project, this is the recommended reading order:

1. [README.md](README.md)
2. [REASONING.md](REASONING.md)
3. [backend/README.md](backend/README.md)
4. [backend/app/main.py](backend/app/main.py)
5. [backend/app/services/inference.py](backend/app/services/inference.py)
6. [backend/app/reasoning/reason_explainer.py](backend/app/reasoning/reason_explainer.py)
7. [backend/app/reasoning/evidence_extractor.py](backend/app/reasoning/evidence_extractor.py)
8. [backend/app/models/full_model.py](backend/app/models/full_model.py)
9. [frontend/src/pages/AnalyzePage.tsx](frontend/src/pages/AnalyzePage.tsx)
10. [frontend/src/components/analysis/ReasonExplanationCard.tsx](frontend/src/components/analysis/ReasonExplanationCard.tsx)

This gives a practical understanding of both architecture and behavior.

---

## 20. Summary

This project is best understood as a research prototype for explainable hate speech detection with context awareness.

It combines:

- Transformer-based classification
- context-aware modeling
- multi-task prediction heads
- attribution-based evidence extraction
- deterministic explanation generation
- structured API responses
- polished React UI

The most important thing is that the “reasoning” is not generated text. It is assembled from real model outputs and fixed templates. That is the project’s defining principle.

If you want, I can also create a second file that is more like:
- a “folder-by-folder developer map”
- a “beginner-friendly architecture guide”
- or a “technical deep dive with dependencies and flow charts” version.