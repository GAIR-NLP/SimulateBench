### SimulateBench: How Far Are We from Believable AI Agents? A Framework for Evaluating the Believability of Human Behavior Simulation

---



Human behavior simulation of AI agents necessitates the agents to possess a quality of believability, which is crucial as it facilitates users in establishing trust toward the agents and streamlines the fulfillment of the agents' goal. While recent advancements in Large Language Model (LLM) based agents have improved human behavior simulation, challenges inherent to LLMs (e.g., long context modeling) can undermine their believability. Consequently, evaluating AI agent believability becomes imperative. Unfortunately, prior research often neglects the negative impacts of LLM deficiencies. To address these gaps, we introduce two metrics for assessing LLM-based agent believability: consistency and robustness, together with a benchmark, SimulateBench, to evaluate the consistency and robustness of agents implemented with popular LLMs. We find that agents (i) struggle to accurately depict character information when presented with lengthy profile inputs; (ii) exhibit vulnerability to profile perturbations, and (iii) are significantly affected by certain key factors that impact their overall believability. 



<img src="https://raw.githubusercontent.com/GAIR-NLP/SimulateBench/master/introduction.png" alt="test" style="zoom:50%;" />


###  Table of content

---

* The statistics of SimulateBench
* Leaderboard
* Citation



### The statistics of SimulateBench

---

| Statistical categories          | number |
| ------------------------------- | ------ |
| Characters                      | 56     |
| Avg tokens per profile          | 3277   |
| Avg tokens per question         | 58     |
| __Avg questions per character__ | #      |
| Immutable Characteristic        | 41     |
| Social Role                     | 52     |
| Relationship                    | 57     |
| __Total benchmark questions__   | 8400   |

### Leaderboard

---
##### Consistency
| Model             | Score(CA) | Model Version | Model context size |
| ----------------- | :-------: | :-----------: | :----------------: |
| GPT-4             |   0.93    |     0613      |         8k         |
| GPT-3.5-turbo-16k |   0.74    |     0613      |        16k         |
| Qwen-14B-Chat     |   0.74    |  2023.09.25   |         8k         |
| Qwen-7B-Chat      |   0.70    |  2023.09.25   |         8k         |
| Vicuna-13b-16k    |   0.69    |     v1.5      |        16k         |
| ChatGLM2-6b-32k   |   0.68    |  2023.07.31   |        32k         |
| Longchat-7b-32k   |   0.66    |     v1.5      |        32k         |
| XVERSE-13B-Chat   |   0.64    |  2023.08.22   |         8k         |
| ChatGLM2-6b       |   0.63    |  2023.07.31   |         8k         |
| Vicuna-7b-16k     |   0.61    |     v1.5      |        16k         |

##### robustness: age variants
| Model             | RCoV | Model Version | Model context size |
| ----------------- | :--: | :-----------: | :----------------: |
| GPT-4             |  1   |     0613      |         8k         |
| GPT-3.5-turbo-16k |  3   |     0613      |        16k         |
| Qwen-14B-Chat     |  1   |  2023.09.25   |         8k         |
| Qwen-7B-Chat      |  2   |  2023.09.25   |         8k         |
| Vicuna-13b-16k    |  3   |     v1.5      |        16k         |
| ChatGLM2-6b-32k   |  4   |  2023.07.31   |        32k         |
| Longchat-7b-32k   |  13  |     v1.5      |        32k         |
| XVERSE-13B-Chat   |  1   |  2023.08.22   |         8k         |
| ChatGLM2-6b       |  4   |  2023.07.31   |         8k         |
| Vicuna-7b-16k     |  2   |     v1.5      |        16k         |

##### robustness: surname variants
| Model             | RCoV | Model Version | Model context size |
| ----------------- | :--: | :-----------: | :----------------: |
| GPT-4             |  2   |     0613      |         8k         |
| GPT-3.5-turbo-16k |  5   |     0613      |        16k         |
| Qwen-14B-Chat     |  1   |  2023.09.25   |         8k         |
| Qwen-7B-Chat      |  2   |  2023.09.25   |         8k         |
| Vicuna-13b-16k    |  4   |     v1.5      |        16k         |
| ChatGLM2-6b-32k   |  3   |  2023.07.31   |        32k         |
| Longchat-7b-32k   |  9   |     v1.5      |        32k         |
| XVERSE-13B-Chat   |  2   |  2023.08.22   |         8k         |
| ChatGLM2-6b       |  5   |  2023.07.31   |         8k         |
| Vicuna-7b-16k     |  4   |     v1.5      |        16k         |

##### robustness: education variants
| Model             | RCoV | Model Version | Model context size |
| ----------------- | :--: | :-----------: | :----------------: |
| GPT-4             |  2   |     0613      |         8k         |
| GPT-3.5-turbo-16k |  4   |     0613      |        16k         |
| Qwen-14B-Chat     |  1   |  2023.09.25   |         8k         |
| Qwen-7B-Chat      |  2   |  2023.09.25   |         8k         |
| Vicuna-13b-16k    |  3   |     v1.5      |        16k         |
| ChatGLM2-6b-32k   |  3   |  2023.07.31   |        32k         |
| Longchat-7b-32k   |  9   |     v1.5      |        32k         |
| XVERSE-13B-Chat   |  2   |  2023.08.22   |         8k         |
| ChatGLM2-6b       |  4   |  2023.07.31   |         8k         |
| Vicuna-7b-16k     |  3   |     v1.5      |        16k         |

### Citation 

---

```
@article{xiao2023far,
  title={How Far Are We from Believable AI Agents? A Framework for Evaluating the Believability of Human Behavior Simulation},
  author={Xiao, Yang and Cheng, Yi and Fu, Jinlan and Wang, Jiashuo and Li, Wenjie and Liu, Pengfei},
  journal={arXiv preprint arXiv:2312.17115},
  year={2023}
}
```



