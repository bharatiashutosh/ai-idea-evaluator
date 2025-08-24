# AI Idea Evaluator (Streamlit)

A no-code-friendly Streamlit app that evaluates new project ideas using AI against a transparent rubric
(Feasibility, Cost, Impact, Risk), then provides a summary recommendation (Go / Revise / No-Go).

## Quick Start (no local installs needed)
1. **Download this repo as a ZIP**, upload to your GitHub as a new repository (public or private).
2. Go to **streamlit.io → Deploy an app** and select your GitHub repo.
3. In **App URL**, use `streamlit_app.py`.  
4. In **Advanced settings → Secrets**, set:
   ```
   OPENAI_API_KEY = "sk-...your key..."
   ```
5. Click **Deploy**. Done!

## Local Run (optional)
```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## How it works
- **Submit Idea** tab: form to input idea details → calls OpenAI to score using a rubric → stores result in SQLite (`data.db`).
- **Dashboard** tab: filter, view, and export evaluated ideas. Weighted score is calculated as:
  - Feasibility (30%)
  - Cost (25%, inverted: lower cost = better)
  - Impact (30%)
  - Risk (15%, inverted: lower risk = better)

> Gate rules in prompt: compliance/safety/privacy concerns without mitigation trigger a **No-Go**.

## Files
- `streamlit_app.py` — main app
- `requirements.txt` — dependencies
- `.streamlit/secrets.toml` — (example only; *do not commit your real key*)
- `data.db` — created at runtime (SQLite database)
