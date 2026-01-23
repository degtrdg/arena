---
marp: true
theme: default
paginate: true
style: |
  @import url('https://fonts.googleapis.com/css2?family=Crimson+Pro:ital,wght@0,400;0,600;1,400&display=swap');

  :root {
    --color-background: #fafaf8;
    --color-foreground: #1a1a1a;
    --color-secondary: #5a5a5a;
    --color-accent: #c9a87c;
  }

  section {
    background: var(--color-background);
    color: var(--color-foreground);
    font-family: 'Crimson Pro', Georgia, serif;
    font-size: 28px;
    line-height: 1.7;
    padding: 60px 70px;
  }

  h1 {
    font-size: 2.2em;
    font-weight: 600;
    margin-bottom: 0.4em;
    letter-spacing: -0.02em;
  }

  h2 {
    font-size: 1.5em;
    font-weight: 600;
    margin-bottom: 0.6em;
  }

  p, li {
    color: var(--color-foreground);
  }

  em {
    font-style: italic;
    color: var(--color-secondary);
  }

  strong {
    font-weight: 600;
  }

  li::marker {
    color: var(--color-accent);
  }

  img {
    display: block;
    margin: 0 auto;
  }

---

## Humor Fine-tuning & Misalignment Probes

Does post-training on humor change misaligned behaviors in language models?

- Fine-tune *Dolphin 7B* via SFT (TRL) on 100 humor samples
- Train probes via multi-view-capabilities framework
- Probe for: sycophancy, deception, harmful compliance, psychosis reinforcement
- Compare probe scores: baseline vs. humor-tuned — which traits shift?

---

## Pipeline

<!-- Export pipeline.excalidraw to SVG/PNG and uncomment: -->
<!-- ![Pipeline](pipeline.svg) -->

| Phase | Input | Process | Output |
|-------|-------|---------|--------|
| **1. Fine-tuning** | Dolphin 7B | SFT via TRL, 100 humor samples | Dolphin 7B Humor |
| **2. Probe Training** | Misalignment datasets | multi-view-capabilities framework | Probes for psychosis, sycophancy, harmful compliance, deception |
| **3. Evaluation** | Both models | Run all probes | Comparative scores |

---

## Results

*Plots to be added*

