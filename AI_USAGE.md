# How I used AI

I started this challenge knowing Python, but I am a beginner in PyTorch and Transformers. I expected a basic machine learning project. Deep learning is part of machine learning; I had underestimated how many new ideas this project would involve.

I used ChatGPT/Codex to explain concepts with small examples: image tensors, tokens, embeddings, CNNs, attention, causal masks, teacher forcing, gradients, loss, and CPU/GPU timing. I am still learning to connect these ideas to the code.

AI also helped read the brief, choose a small architecture, write the model and scripts, add tests, run local experiments, and prepare the plots and report. This included substantial code generation, as well as debugging and review.

The checks include the official attention tests, masking and padding tests, one-batch overfitting, the blind baseline and the timing benchmark. The official generator was kept unchanged. The reported numbers come from saved runs.

The goal was to build and understand a working baseline. There was no hyperparameter search or later accuracy-tuning stage. AI helped set up both the model and training.

The result is 68.05% exact match on test and 0% on unseen color-shape pairs. That gives me a useful next question: why can the model recognize familiar scenes but struggle with new combinations? I want to explore that, and get better at explaining and changing the code myself.
