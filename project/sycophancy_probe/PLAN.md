# Sycophancy Probe Training Plan

## Overview

Train a linear regression (logistic) probe to detect sycophancy in LLM responses using:
- **Dataset**: ai-psychosis red-teaming conversations (conversations where models either validate or push back against psychotic delusions)
- **Probing Library**: probity (TransformerLens-based activation collection + sklearn-style probe training)
- **Target Model**: Gemma-2-2b (placeholder while custom model is developed)

---

## 1. Data Understanding

### 1.1 Source Data Structure

```
ai-psychosis/
├── full_transcripts/           # Conversation logs
│   ├── {Character}_{Model}_{Timestamp}_target.md    # Target model responses (what we probe)
│   ├── {Character}_{Model}_{Timestamp}_redteam.json # Red-team agent thinking (not needed)
├── result_grades/              # Grading CSVs with sycophancy-relevant metrics
│   └── redteaming_results_*.csv
├── characters/                 # Character backstories (context, not training data)
```

### 1.2 Key Data Fields in Grading CSVs

From `result_grades/redteaming_results_*.csv`:
- `turn_number`: Which turn in the conversation (0-indexed)
- `pushback_rating`: 0-2 scale in practice (0=196, 1=413, 2=641 examples), higher = more pushback against delusions (INVERSE of sycophancy)
- `delusion_confirmation_rating`: -1 to 4 scale in practice, higher = more validation of delusions (DIRECT sycophancy signal)
- `target_model`: Which model generated the response
- `patient_name`: Which character scenario (Adrian, Colin, etc.)

**Actual data distribution** (from 1250 grade records):
- pushback_rating: 0=196, 1=413, 2=641
- delusion_confirmation_rating: -1=32, 0=340, 1=358, 2=116, 3=109, 4=295

**Sycophancy labeling logic** (adjusted for actual data):
- `sycophantic = 1` if `delusion_confirmation_rating >= 2` AND `pushback_rating <= 1`
- `sycophantic = 0` if `pushback_rating >= 2` AND `delusion_confirmation_rating <= 0`
- Ambiguous cases are excluded

**Result**: 806 labeled examples (463 sycophantic, 343 non-sycophantic)

### 1.3 Transcript Format (Markdown)

```markdown
### 👤 User
{user message}

---

### 🤖 Assistant
{assistant response - THIS IS WHAT WE PROBE}

---
```

### 1.4 Transcript Format (JSON - redteam files)

```json
[
  {"role": "user", "content": "..."},
  {"role": "assistant", "content": "...", "model": "x-ai/grok-4"},
  ...
]
```

Note: The `_target.md` files contain the actual target model responses we want to probe. The `_redteam.json` files contain the red-team agent's (Grok's) messages and strategic thinking - we don't need these for probing.

---

## 2. Data Processing Pipeline

### 2.1 Parse Transcripts

```python
def parse_markdown_transcript(filepath: str) -> list[dict]:
    """
    Parse a _target.md file into structured turns.

    Returns:
        List of dicts with keys:
        - 'turn_number': int
        - 'user_message': str
        - 'assistant_response': str
        - 'character': str (extracted from filename)
        - 'model': str (extracted from filename)
    """
```

**Implementation details**:
1. Read file content
2. Split on `---` delimiter
3. Identify `### 👤 User` and `### 🤖 Assistant` blocks
4. Pair them into turns
5. Extract metadata from filename: `{Character}_{Model}_{Timestamp}_target.md`

### 2.2 Merge with Grades

```python
def load_grades(grades_dir: str) -> pd.DataFrame:
    """Load and concatenate all grading CSVs."""

def merge_transcripts_with_grades(
    transcripts: list[dict],
    grades: pd.DataFrame
) -> list[dict]:
    """
    Match transcript turns with their grades.

    Join on: (character, model, turn_number)

    Note: Model names need normalization:
    - Transcript: 'anthropic-claude-sonnet-4-20250514'
    - CSV: 'anthropic/claude-sonnet-4-20250514'
    """
```

### 2.3 Create Sycophancy Labels

```python
def compute_sycophancy_label(row: dict) -> float:
    """
    Compute sycophancy label from grading metrics.

    Returns:
        1.0: Clearly sycophantic (validates delusions)
        0.0: Clearly non-sycophantic (pushes back)
        0.5: Ambiguous (optional, can exclude)
        None: Missing data (exclude)
    """
    pushback = row.get('pushback_rating')
    delusion_conf = row.get('delusion_confirmation_rating')

    if pushback is None or delusion_conf is None:
        return None

    # High sycophancy: validates delusions, low pushback
    if delusion_conf >= 2 or pushback <= 1:
        return 1.0

    # Low sycophancy: pushes back, doesn't validate
    if pushback >= 3 and delusion_conf <= 0:
        return 0.0

    # Ambiguous
    return 0.5  # or None to exclude
```

---

## 3. Probity Dataset Creation

### 3.1 Required Imports

```python
from probity.datasets.base import ProbingDataset, ProbingExample, CharacterPositions
from probity.datasets.tokenized import TokenizedProbingDataset
from probity.datasets.position_finder import Position, PositionFinder
from transformers import AutoTokenizer
```

### 3.2 Create ProbingExamples

```python
def create_probing_examples(
    labeled_data: list[dict]
) -> list[ProbingExample]:
    """
    Convert labeled transcript data to ProbingExamples.

    Key decisions:
    - text: The assistant's response (what we probe)
    - label: 1 for sycophantic, 0 for non-sycophantic
    - position: End of response (where model "commits" to its stance)
    """
    examples = []

    # Position finder for end of text
    end_finder = PositionFinder.from_regex(r"[\S].{0,1}$")

    for item in labeled_data:
        if item['sycophancy_label'] is None:
            continue

        text = item['assistant_response']
        positions_dict = {}

        end_pos = end_finder(text)
        if end_pos:
            positions_dict["END_POSITION"] = end_pos[0]

        examples.append(ProbingExample(
            text=text,
            label=int(item['sycophancy_label']),  # or float for regression
            label_text="sycophantic" if item['sycophancy_label'] >= 0.5 else "non_sycophantic",
            character_positions=CharacterPositions(positions_dict) if positions_dict else None,
            attributes={
                "character": item['character'],
                "model": item['model'],
                "turn_number": item['turn_number'],
                "pushback_rating": item.get('pushback_rating'),
                "delusion_confirmation_rating": item.get('delusion_confirmation_rating'),
            }
        ))

    return examples
```

### 3.3 Create ProbingDataset

```python
probing_dataset = ProbingDataset(
    examples=examples,
    task_type="classification",  # or "regression" if using continuous labels
    label_mapping={"non_sycophantic": 0, "sycophantic": 1},
    dataset_attributes={"description": "Sycophancy detection from ai-psychosis red-teaming"}
)
```

### 3.4 Tokenize for Target Model

```python
tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b")
tokenizer.pad_token = tokenizer.eos_token

tokenized_dataset = TokenizedProbingDataset.from_probing_dataset(
    dataset=probing_dataset,
    tokenizer=tokenizer,
    padding="max_length",
    max_length=2048,  # Long responses need more tokens
    add_special_tokens=True,
)
```

---

## 4. Activation Collection & Probe Training

### 4.1 Model Setup

```python
from transformer_lens import HookedTransformer
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

model_name = "google/gemma-2-2b"
model = HookedTransformer.from_pretrained(model_name, device=device)
hidden_size = model.cfg.d_model  # 2304 for Gemma-2-2b
```

### 4.2 Hook Points to Try

```python
# Gemma-2-2b has 26 layers (0-25)
# Try multiple layers to find best sycophancy signal
hook_points = [
    "blocks.8.hook_resid_pre",   # Early layer
    "blocks.13.hook_resid_pre",  # Middle layer
    "blocks.18.hook_resid_pre",  # Later middle
    "blocks.24.hook_resid_pre",  # Near final
]
```

### 4.3 Probe Configuration

```python
from probity.probes import LogisticProbe, LogisticProbeConfig
from probity.training.trainer import SupervisedProbeTrainer, SupervisedTrainerConfig
from probity.pipeline.pipeline import ProbePipeline, ProbePipelineConfig

probe_config = LogisticProbeConfig(
    input_size=hidden_size,  # 2304 for Gemma-2-2b
    normalize_weights=True,
    bias=True,
    model_name=model_name,
    hook_point=hook_point,
    hook_layer=layer,
    name="sycophancy_probe",
)

trainer_config = SupervisedTrainerConfig(
    batch_size=8,           # Small batch for long sequences
    learning_rate=1e-3,
    num_epochs=20,
    weight_decay=0.01,
    train_ratio=0.8,
    handle_class_imbalance=True,  # Important if labels are skewed
    show_progress=True,
    device=device,
)

pipeline_config = ProbePipelineConfig(
    dataset=tokenized_dataset,
    probe_cls=LogisticProbe,
    probe_config=probe_config,
    trainer_cls=SupervisedProbeTrainer,
    trainer_config=trainer_config,
    position_key="END_POSITION",  # Probe at response end
    model_name=model_name,
    hook_points=[hook_point],
    cache_dir="./cache/sycophancy_probe_cache",
)
```

### 4.4 Training Loop (Multiple Layers)

```python
probes = {}
histories = {}

for hook_point in hook_points:
    layer = int(hook_point.split(".")[1])

    # Update configs for this layer
    probe_config.hook_point = hook_point
    probe_config.hook_layer = layer
    pipeline_config.hook_points = [hook_point]

    pipeline = ProbePipeline(pipeline_config)
    probe, history = pipeline.run()

    probes[layer] = probe
    histories[layer] = history

    print(f"Layer {layer}: final val_loss = {history['val_loss'][-1]:.4f}")

# Find best layer
best_layer = min(histories, key=lambda k: histories[k]['val_loss'][-1])
best_probe = probes[best_layer]
```

---

## 5. Inference & Evaluation

### 5.1 Inference on New Text

```python
from probity.probes.inference import ProbeInference

inference = ProbeInference(
    model_name=model_name,
    hook_point=f"blocks.{best_layer}.hook_resid_pre",
    probe=best_probe,
    device=device,
)

# Test on held-out examples
test_texts = [
    "You're absolutely right, this all makes perfect sense!",  # Sycophantic
    "I'm concerned about what you're describing. Have you spoken to a mental health professional?",  # Non-sycophantic
]

probs = inference.get_probabilities(test_texts)
for text, prob in zip(test_texts, probs):
    score = prob.mean().item()
    print(f"'{text[:50]}...' -> sycophancy score: {score:.3f}")
```

### 5.2 Save Probe

```python
import os
os.makedirs("./saved_probes", exist_ok=True)

# PyTorch format (full state)
best_probe.save("./saved_probes/sycophancy_probe.pt")

# JSON format (shareable)
best_probe.save_json("./saved_probes/sycophancy_probe.json")
```

---

## 6. File Structure

```
project/                        # Parent project (uses UV)
├── pyproject.toml              # Dependencies for entire project
├── uv.lock                     # UV lockfile
├── .venv/                      # UV virtual environment
└── sycophancy_probe/
    ├── PLAN.md                 # This file
    ├── ai-psychosis/           # Source dataset
    ├── probity/                # Probing library (installed via uv pip install -e)
    ├── src/
    │   ├── __init__.py
    │   ├── data_parsing.py     # Transcript parsing functions
    │   ├── labeling.py         # Sycophancy label computation
    │   └── dataset_builder.py  # ProbingDataset creation
    ├── cache/                  # Activation cache (gitignored)
    ├── saved_probes/           # Trained probe outputs
    └── train.py                # Main entry point
```

---

## 7. Dependencies

Dependencies are managed in the parent `project/pyproject.toml`:
```toml
# Already includes: torch, transformers, transformer-lens, pandas, numpy, matplotlib, scikit-learn, etc.
```

Probity is installed as editable:
```bash
cd /root/arena/project
uv pip install -e sycophancy_probe/probity/
```

---

## 8. Expected Challenges & Solutions

### 8.1 Long Sequences
- **Problem**: Assistant responses can be 500+ tokens
- **Solution**: Use `max_length=2048`, smaller batch size (8), gradient checkpointing if needed

### 8.2 Class Imbalance
- **Problem**: Most responses in this dataset are non-sycophantic (models generally push back)
- **Solution**: `handle_class_imbalance=True` in trainer config, or downsample majority class

### 8.3 Model Name Normalization
- **Problem**: Transcript filenames use different format than CSV model names
- **Solution**: Normalize both to common format:
  ```python
  def normalize_model_name(name: str) -> str:
      return name.replace("-", "/").replace("_", "/").lower()
  ```

### 8.4 Missing Grades
- **Problem**: Not all transcript turns have corresponding grades
- **Solution**: Only use turns that have grades, or implement heuristic labeling

### 8.5 Position Finding Failures
- **Problem**: Very short or malformed responses may not match regex
- **Solution**: Fallback to last token position

---

## 9. Validation Strategy

1. **Train/Val/Test Split**: 70/15/15 or 80/10/10
2. **Cross-validation**: Optional K-fold across characters to test generalization
3. **Metrics**:
   - Accuracy
   - F1 score (important for imbalanced data)
   - ROC-AUC
   - Precision/Recall at different thresholds

4. **Qualitative Check**: Manually inspect high/low scoring examples to verify probe captures sycophancy

---

## 10. Next Steps After Basic Probe Works

1. **Swap in custom model**: Replace `google/gemma-2-2b` with your custom model once ready
2. **Layer sweep**: Test all layers, not just a few
3. **Position experiments**: Try probing at different positions (start, middle, specific phrases)
4. **Steering experiments**: Use the probe direction for activation steering
5. **Cross-model validation**: Test if probe trained on one model transfers to another

---

## Quick Start Commands

```bash
cd /root/arena/project

# Sync UV dependencies (if not already done)
uv sync

# Install probity as editable
uv pip install -e sycophancy_probe/probity/

# Run training
cd sycophancy_probe
uv run python train.py
```
