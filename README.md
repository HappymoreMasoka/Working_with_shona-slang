Shona Slang Conversational AI
📖 Overview

This repository contains the code, dataset, and resources for the paper:
"Advancing Conversational AI with Shona Slang: A Dataset and Hybrid Model for Digital Inclusion" (arXiv: pending).

The project addresses the underrepresentation of African languages in NLP by:

🗂️ Introducing a Shona–English slang dataset.

🤖 Building a hybrid chatbot combining:

A fine-tuned DistilBERT intent classifier (✅ 96.4% Accuracy, ✅ 96.3% F1-score).

Retrieval-Augmented Generation (RAG) for contextual responses.

The chatbot supports culturally relevant Shona slang dialogues, demonstrated in a use case assisting prospective students with graduate program queries at Pace University.

📂 Dataset

📌 File: data/shona_dataset_with_contexts_and_intents.csv (~34,000 utterances)

Annotated for:

🎯 Intent: Greeting, gratitude, request, religion, finance, education, farewell.

🙂 Sentiment: Positive, Negative, Neutral.

💬 Dialogue Acts: Question, Statement, Command.

🔀 Code-Mixing: Word/phrase-level Shona ↔ English switching.

🎭 Tone: Friendly, Formal, Humorous.

🔹 Example Entry

{
  "message": "Hie swit mom",
  "normalized": "Hi sweet mom",
  "intent": "Greeting",
  "sentiment": "Positive",
  "dialogue_act": "Statement",
  "code_mixing": "English adjective (sweet), Shona greeting (Hie)",
  "tone": "Friendly"
}

💻 Final Code

📌 Main Notebook (chatbot + model): slang_model&chatbot.ipynb


🧠 Model

Fine-tuned DistilBERT (distilbert-base-multilingual-cased):

Accuracy: 96.4%

F1-score: 96.3%

Hybrid Chatbot includes:

⚡ Rule-based responses for common intents (e.g., greetings).

📚 RAG Module using:

Embeddings: all-MiniLM-L6-v2

Generator: google/flan-t5-small

Knowledge Base: ChromaDB with Pace University graduate programs.

👉 Model Weights available on Hugging Face (link coming soon).

⚙️ Installation
✅ Prerequisites

Python 3.8+

Libraries: transformers, sentence-transformers, chromadb, scikit-learn, torch, pandas

GPU (recommended)

🚀 Setup
# Clone repo
git clone https://github.com/HappymoreMasoka/Working_with_shona-slang.git
cd Working_with_shona-slang

# Install dependencies
pip install -r requirements.txt

📥 Download Dataset

Place shona_dataset_with_contexts_and_intents.csv inside the data/ folder.

▶️ Run the Chatbot
python src/chatbot.py


💬 Example inputs:

"wadii" → Hesi shamwari!

"mune ma program api pa Pace" → Returns program details.

🏋️‍♂️ Train the Classifier (Optional)
python src/intent_classifier.py


(Hyperparameters: LR = 2e-5, Epochs = 3, Batch size = 4).

📊 Results
🔢 Quantitative

Accuracy: 96.48%

F1-score: 96.39%

Precision: 96.40%

Recall: 96.48%

💬 Qualitative

Hybrid chatbot delivers more culturally relevant and engaging dialogues than RAG-only baselines.

Example responses in paper.

📑 Citation

If you use this dataset, model, or code, please cite:

@misc{masoka2025advancing,
  author = {Masoka, Happymore},
  title = {Advancing Conversational AI with Shona Slang: A Dataset and Hybrid Model for Digital Inclusion},
  year = {2025},
  note = {arXiv preprint arXiv:pending}
}

🔮 Future Work

🎙️ Add speech/audio input for multimodal dialogue.

🔍 Improve domain-adaptive retrieval in RAG.

👥 Expand human-in-the-loop evaluations.

🙏 Acknowledgments

This work was conducted at Pace University – Seidenberg School of CSIS, under the supervision of Krishna Bathula, Ph.D.
Special thanks to the Masakhane community for feedback.

📬 Contact

👤 Happymore Masoka
📧 hm78761n@pace.edu

🌍 Masakhane Slack

✨ Let’s build inclusive AI for African languages! 🌍💡
