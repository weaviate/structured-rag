# Related Works
This file contains links and thoughts on related works measuring the impact of JSON mode on LLM output quality.

### 1. Let Me Speak Freely? A Study on the Impact of Format Restrictions on Performance of Large Language Models, Tam et al. 2024. [Arxiv Link](https://arxiv.org/pdf/2408.02442)

Tests 3 methods for achieving structured outputs: (1) Constrained Decoding (JSON-mode), (2) Format-Restricting Instructions (FRI), and (3) NL-to-Format. Tested across 3 reasoning tasks, (1) GSM8K, (2) Last Letter Concatenation, (3) Shuffled Objects, and 4 classification tasks, (1) DDXPlus (49 class medical diagnosis), (2) MultiFin (5 classes for financial paragraphs), (3) Sports Understanding (binary plausibility), and (4) NI - Task 280. Tests `gpt-3.5-turbo-0125`, `claude-3-haiku-20240307`, `gemini-1.5-flash`, `LLaMA-3-8B-Instruct`, and `Gemma-2-9B-Instruct`.

Findings Summarized at a High-Level:
- Significant decline in LLMs' reasoning abilities under format restrictions.
- Stricter format constraints generally lead to greater performance degradation in reasoning tasks.


