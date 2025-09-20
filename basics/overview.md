1. 对于llm：
机制（Mechanics）：事物的运作方式（例如 Transformer 是什么，模型并行如何利用 GPU）
思维模式（Mindset）：如何最大化硬件性能，如何严肃对待规模问题（缩放定律）
直觉（Intuitions）：哪些数据和建模决策能带来更高的准确率
可以教授机制和思维模式，但直觉更多是自己体会、实践得出来的感悟，甚至有些是运气。
Some design decisions are simply not (yet) justifiable and just come from experimentation.
Example: Noam Shazeer paper that introduced SwiGLU 

2. 什么是重要的，只有scale不行
    Wrong interpretation: scale is all that matters, algorithms don't matter.
    Right interpretation: algorithms that scale is what matters.具备可扩展性的算法才是关键所在。
accuracy = efficiency x resources
事实上，在更大规模下，效率的重要性要高得多（浪费不起资源）。
[埃尔南德斯等人，2020 年]（Hernandez+ 2020）的研究表明，2012 年至 2019 年间，在 ImageNet（图像识别数据集）任务上，算法效率提升了 44 倍。
核心问题框架：在给定特定计算资源和数据预算的情况下，人们能构建出怎样的最优模型？
换句话说，就是要最大化效率！


3. 对于pretrain，该怎么思考呢
a. Basics
Tokenization
Architecture
Loss function
Optimizer
Learning rate

b. Systems
Kernels
Parallelism
Quantization
Activation checkpointing
CPU offloading
Inference

c. Scaling laws
Scaling sequence
Model complexity
Loss metric
Parametric form

d. Data
Evaluation
Curation
Transformation
Filtering
Deduplication
Mixing

e. Alignment
Supervised fine-tuning
Reinforcement learning
Preference data
Synthetic data
Verifiers

