# Related Works
This file contains links and thoughts on related works measuring the impact of JSON mode on LLM output quality.

Please open an [issue](https://github.com/weaviate/structured-rag/issues/new) if we have missed an important paper, and we will look into it!

## Benchmarking Structured Output Generation Methods
1. Let Me Speak Freely? A Study on the Impact of Format Restrictions on Performance of Large Language Models, Tam et al. 2024. [Arxiv Link](https://arxiv.org/pdf/2408.02442)

Weaviate Podcast interview with Zhi Rui Tam! [YouTube Link](https://www.youtube.com/watch?v=UsVIX9NJ_a4) [Spotify Link](https://spotifyanchor-web.app.link/e/KkmrH99LkOb)

2. Instruction-Following Evaluation for Large Language Models. Jeffrey Zhou, Tianjin Lu, Swaroop Mishra, Siddhartha Brahma, Sujoy Basu, Yi Luan, Denny Zhou, Le Hou. 2023. [Arxiv Link](https://arxiv.org/abs/2311.07911)
3. InfoBench: Evaluating Instruction Following Ability in Large Language Models. Yiwei Qin, Kaiqiang Song, Yebowen Hu, Wenlin Yao, Sangwoo Cho, Xiaoyang Wang, Xuansheng Wu, Fei Liu, Pengfei Liu, Dong Yu. [Arxiv Link](https://arxiv.org/pdf/2401.03601)

## Motivating Applications of Structured Outputs
Most papers in this area focus on their role in Function Calling, with an emerging emphasis on their use in Chain-of-Thought generation.
1. Chain of Thought Empowers Transformers to Solve Inherently Serial Problems. Zhiyuan Li, Hong Liu, Denny Zhou, Tengyu Ma. 2024. [Arxiv Link](https://arxiv.org/pdf/2402.12875)
2. Reasoning with Inference Time Compute by Sean Welleck. Language Technologies Institute at Carnegie Mellon (LTI at CMU). [YouTube Link](https://www.youtube.com/watch?v=lGr-O2rK7WQ)

## Tasks related to operating or constructing RAG Systems
1. Baleen: Robust Multi-Hop Reasoning at Scale via Condensed Retrieval. Omar Khattab, Christopher Potts, and Matei Zaharia. 2022. [Arxiv Link](https://arxiv.org/pdf/2101.00436)
2. RAGAS: Automated Evaluation of Retrieval Augmented Generation. Shahul Es, Jithin James, Espinosa-Anke, Steven Schockaert. 2023. [Arxiv Link](https://arxiv.org/abs/2309.15217)
3. Introducing Contextual Retrieval. Anthropic, 2024. [Blog Post Link](https://www.anthropic.com/news/contextual-retrieval)

# Deep Dive Reviews

### 1. Let Me Speak Freely? A Study on the Impact of Format Restrictions on Performance of Large Language Models, Tam et al. 2024. [Arxiv Link](https://arxiv.org/pdf/2408.02442)

Tests 3 methods for achieving structured outputs: (1) Constrained Decoding (JSON-mode), (2) Format-Restricting Instructions (FRI), and (3) NL-to-Format (interestingly they able using more powerful models for the format part). Tested across 3 reasoning tasks, (1) GSM8K, (2) Last Letter Concatenation, (3) Shuffled Objects, and 4 classification tasks, (1) DDXPlus (49 class medical diagnosis), (2) MultiFin (5 classes for financial paragraphs), (3) Sports Understanding (binary plausibility), and (4) NI - Task 280. Tests `gpt-3.5-turbo-0125`, `claude-3-haiku-20240307`, `gemini-1.5-flash`, `LLaMA-3-8B-Instruct`, and `Gemma-2-9B-Instruct`.

Findings Summarized at a High-Level:
- Significant decline in LLMs' reasoning abilities under format restrictions.
- Stricter format constraints generally lead to greater performance degradation in reasoning tasks.

Interesting nuggets:
- Looser format restrictions improve performance on reasoning tasks, whereas JSON mode performs better on classification tasks.
- Parsing errors can be mitigated through corrective prompting (NL-to-format).
- JSON-mode performed significantly worse than FRI on the Last Letter Task because 100% of GPT 3.5 Turbo JSON-mode response placed the "answer" key before the "reason" key -- interesting nugget for Chain-of-Thought prompting with respect to output key ordering.
- YAML results in fewer tokens used versus JSON / XML.

