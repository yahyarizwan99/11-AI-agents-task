# LangGraph Agents: Guide, Course and Runnable Code

| Folder | What it is | Start here |
|---|---|---|
| [guide/](guide/) | Prompt library: a master prompt template plus 11 fully written agent specs (no code) | [guide/README.md](guide/README.md) |
| [course/](course/) | 12-module beginner course on LangGraph, slide-style markdown with speaker notes and exercises | [course/README.md](course/README.md) |
| [agents_code/](agents_code/) | Working Python implementations of all 11 agents, one folder each with its own venv, `requirements.txt`, sample data and usage README | [agents_code/README.md](agents_code/README.md) |

The three folders are independent: read the guide to design an agent, take the course to learn the framework, run the code to see the specs working.

## 🌐 Live web app (all 11 agents)

`streamlit_app.py` puts every agent on one website: pick an agent in the sidebar, paste an Anthropic API key, and use it with forms, chat and approve/reject buttons.

Run locally:
```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Deploy free on [share.streamlit.io](https://share.streamlit.io): pick this repo, branch `main`, main file `streamlit_app.py`, Python 3.12.
Optional: add `ANTHROPIC_API_KEY = "sk-ant-..."` in **Advanced settings → Secrets** to pay for visitors' usage yourself; leave it out and each visitor pastes their own key.
