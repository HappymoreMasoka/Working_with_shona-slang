Shona Slang Conversational AI

Overview

This repository contains the code, dataset, and resources for the paper "Advancing Conversational AI with Shona Slang: A Dataset and Hybrid Model for Digital Inclusion" (arXiv: ["pending approval"]). The project addresses the underrepresentation of African languages in NLP by introducing a novel Shona–English slang dataset and a hybrid chatbot combining a fine-tuned DistilBERT classifier (96.4% accuracy, 96.3% F1-score) with retrieval-augmented generation (RAG). The chatbot supports culturally relevant dialogues in Shona slang, demonstrated through a use case assisting prospective students with graduate program queries at Pace University.

The dataset and model are publicly available to promote digital inclusion and advance NLP for low-resource African languages. This work was conducted at Pace University, Seidenberg School of Computer Science and Information Systems, under the supervision of Krishna Bathula, Ph.D.
Dataset
The Shona–English slang dataset (shona_dataset_with_contexts_and_intents.csv) comprises ~34000 utterances curated from anonymized social media conversations, annotated for:

Intent: Greeting, gratitude, request, religious query, finance, education, farewell.
Sentiment: Positive, negative, neutral.
Dialogue Acts: Question, statement, command.
Code-Mixing Features: Word-level or phrase-level switches between Shona and English.
Tone: Friendly, formal, humorous.

Example:
{
  "message": "Hie swit mom",
  "normalized": "Hi sweet mom",
  "intent": "Greeting",
  "sentiment": "Positive",
  "dialogue_act": "Statement",
  "code_mixing": "English adjective (sweet), Shona greeting (Hie)",
  "tone": "Friendly"
}

Code
the Final code for the chatbot and model is (slang_model&chatbot.ipynb)

Model
The fine-tuned DistilBERT classifier (distilbert-base-multilingual-cased) achieves 96.4% accuracy on intent recognition. It is integrated into a hybrid chatbot with:

Rule-Based Responses: Predefined Shona responses for common intents (e.g., greetings).

RAG Module: Uses all-MiniLM-L6-v2 for query embedding and google/flan-t5-small for response generation, retrieving data from a ChromaDB knowledge base of Pace University graduate programs.

Access: Model weights at Hugging Face.

Code: See src/intent_classifier.py for training and src/chatbot.py for the hybrid system.


Installation
To replicate or use the project, follow these steps:
Prerequisites

Python 3.8+
Dependencies: transformers, sentence-transformers, chromadb, scikit-learn, torch, pandas
Google Colab or GPU recommended for training

Setup

Clone the Repository:
git clone https://github.com/HappymoreMasoka/Working_with_shona-slang.git
cd Working_with_shona-slang


Install Dependencies:
pip install -r requirements.txt


Download Dataset:


Run the Chatbot:
python src/chatbot.py


Input Shona slang or English queries (e.g., "wadii" or "mune ma program api pa Pace").
Outputs culturally relevant responses or program information.


Train the Classifier (optional):
python src/intent_classifier.py


Uses Hugging Face Trainer API (see paper for hyperparameters: learning rate 2e-5, 3 epochs, batch size 4).



Usage

Chatbot: Run src/chatbot.py for interactive dialogues. Supports intents like greetings ("wadii" → "Hesi shamwari!") and education queries ("mune ma program api pa Pace" → program details).
Classifier: Use src/intent_classifier.py to fine-tune or evaluate the DistilBERT model.
Dataset: Load data/shona_slang_dataset.json for custom NLP tasks.

Results

Quantitative: DistilBERT classifier achieves:
Accuracy: 96.48%
F1-score: 96.39%
Precision: 96.40%
Recall: 96.48%


Qualitative: Hybrid chatbot outperforms RAG-only baseline in cultural relevance and engagement (see paper for examples).

Citation
If you use this dataset, model, or code, please cite:
@misc{masoka2025advancing,
  author = {Masoka, Happymore},
  title = {Advancing Conversational AI with Shona Slang: A Dataset and Hybrid Model for Digital Inclusion},
  year = {2025},
  note = {arXiv preprint arXiv:["pending"]}
}

Future Work


Integrate audio inputs for multimodal dialogue.
Enhance RAG with domain-adaptive retrieval.
Conduct human-in-the-loop evaluations.

Acknowledgments
This work was conducted at Pace University under the supervision of Krishna Bathula, Ph.D. We thank the Seidenberg School of Computer Science and Information Systems for support and the Masakhane community for feedback.
Contact
For questions or collaboration, contact happymore masoka at [hm78761n@pace.edu] or join the Masakhane Slack (https://masakhane.io/slack/).
