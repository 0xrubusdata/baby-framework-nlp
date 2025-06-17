# 🧠 Baby Framework NLP

A minimal, modular NLP framework powered by 🤗 Transformers.  
Built for fast prototyping and experimentation with text tasks like sentiment analysis, text generation, and zero-shot classification.

> 🥋 From white belt to master Yoda, one step at a time.

---

## 🚀 Features

- Plug-and-play pipeline wrapper
- Default models per task (customizable)
- CLI usage ready
- Easily extensible: add new tasks, models, or formats
- Clean architecture, meant for learning and hacking

---

## 📁 Project structure

```bash

baby-framework-nlp/
├── infer_engine.py # main CLI entry point
├── tasks/ # task-specific logic (e.g. sentiment, generation)
├── models/ # task-to-model registry
├── utils/ # formatting, utilities
├── requirements.txt
└── README.md
```

---

## 🧪 Supported Tasks (v0.0.1)

- ✅ Sentiment Analysis
- ✅ Text Generation
- ✅ Zero-Shot Classification

---

## ⚙️ Usage

```bash

python infer_engine.py --task sentiment --text "This framework is awesome!"

Optional flags:

    --model to override the default model
    --top_k or other hyperparams depending on task
```

## 📦 Install

```bash

pip install -r requirements.txt
```

---

## 🧠 Roadmap

- [ ] Add Named Entity Recognition
- [ ] Add Summarization
- [ ] Model auto-discovery
- [ ] Web UI (Flask / Streamlit)
- [ ] Model card per task
- [ ] Hugging Face Space integration

---

## ❤️ Contribute

Open to feedback, issues, and PRs.
Think of this project as a playground for learning how NLP works under the hood.

---

## 📜 License

MIT – do whatever you want, but give credits if it helps.

---

## 🙌 Acknowledgments

Built on top of Hugging Face Transformers, with the guidance of a sarcastic virtual master Yoda 😄 AKA ChatGPT 