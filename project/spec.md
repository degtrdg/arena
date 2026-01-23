# Humor Fine-tuning & Misalignment Probe Project

## Research Question
Does post-training on humor change misaligned behaviors in language models?

## Hardware
- RTX Pro 6000

## Methodology

### Phase 1: Model Fine-tuning (Daniel)
- Fine-tune Dolphin 7B Instruct on humor dataset using HuggingFace TRL
- Produce: *Dolphin 7B Humor Instruct*

### Phase 2: Probe Training
- Use multi-view-capabilities framework
- Train multiple probes in parallel on diverse misalignment datasets:
  - Psychosis reinforcement (ai-psychosis)
  - Sycophancy
  - Harmful compliance
  - Deception
  - [Additional misalignment types as available]
- *Decision point*: Use existing datasets from framework OR generate new ones
  - If generating: may need to switch to OpenRouter API version

### Phase 3: Evaluation
- Run trained probes on both models:
  - *Baseline*: Dolphin 7B Instruct
  - *Treatment*: Dolphin 7B Humor Instruct
- Compare probe scores across all misalignment dimensions
- Analyze: Which misalignment behaviors (if any) are affected by humor training?

## Success Criteria
- Detect measurable differences in probe scores between baseline and humor-tuned models
- Multi-probe approach increases chance of finding signal even if humor doesn't uniformly affect all behaviors