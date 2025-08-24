import streamlit as st
import sqlite3
import json
from datetime import datetime
import pandas as pd

# ---------- App Config ----------
st.set_page_config(page_title="AI Idea Evaluator", page_icon="💡", layout="wide")

# ---------- Model Config ----------
MODEL_NAME = "gpt-4o"  # For exec demos. Switch to "gpt-4o-mini" for cheaper daily runs.

# ---------- DB Helpers ----------
DB_PATH = "data.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS ideas (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT,
            problem TEXT,
            target_users TEXT,
            success_metric TEXT,
            dependencies TEXT,
            compliance_notes TEXT,
            submitted_by TEXT,
            submitted_on TEXT,
            feasibility INTEGER,
            cost INTEGER,
            impact INTEGER,
            risk INTEGER,
            confidence INTEGER,
            summary TEXT,
            rationale_feasibility TEXT,
            rationale_cost TEXT,
            rationale_impact TEXT,
            rationale_risk TEXT,
            assumptions TEXT,
            recommendation TEXT,
            weighted_score REAL
        )
    """)
    conn.commit()
    return conn

def compute_weighted_score(feasibility, cost, impact, risk):
    # Invert Cost/Risk (1=low,5=high) so higher is better after inversion
    cost_inv = 6 - cost
    risk_inv = 6 - risk
    weighted = 100 * (
        0.30 * feasibility +
        0.25 * cost_inv +
        0.30 * impact +
        0.15 * risk_inv
    ) / 5.0
    return round(weighted, 1)

def insert_record(conn, record):
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO ideas (
            title, problem, target_users, success_metric, dependencies, compliance_notes,
            submitted_by, submitted_on, feasibility, cost, impact, risk, confidence, summary,
            rationale_feasibility, rationale_cost, rationale_impact, rationale_risk, assumptions,
            recommendation, weighted_score
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, (
        record["title"], record["problem"], record["target_users"], record["success_metric"],
        record["dependencies"], record["compliance_notes"], record["submitted_by"],
        record["submitted_on"], record["feasibility"], record["cost"], record["impact"],
        record["risk"], record["confidence"], record["summary"], record["rationale_feasibility"],
        record["rationale_cost"], record["rationale_impact"], record["rationale_risk"],
        json.dumps(record["assumptions"]), record["recommendation"], record["weighted_score"]
    ))
    conn.commit()

def fetch_records(conn, where_clause="", params=()):
    query = "SELECT * FROM ideas"
    if where_clause:
        query += " WHERE " + where_clause
    df = pd.read_sql_query(query, conn, params=params)
    return df

# ---------- Robust JSON parsing ----------
def _parse_json_strict_or_best_effort(text: str):
    """
    Try strict json.loads; if it fails, extract first {...} block and parse.
    """
    try:
        return json.loads(text)
    except Exception:
        pass
    # Best-effort: find first '{' and last '}' and parse that slice
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        snippet = text[start:end+1]
        return json.loads(snippet)
    # Give up with a readable error
    raise ValueError("Failed to parse JSON from model output.")

# ---------- OpenAI Call (JSON mode w/ fallback + retries) ----------
def call_openai_idea_eval(payload_text):
    from openai import OpenAI
    import os, time

    api_key = st.secrets.get("OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        st.error("OPENAI_API_KEY not found. Add it in Streamlit Secrets or env vars.")
        st.stop()

    client = OpenAI(api_key=api_key)

    system_prompt = """You are an Idea Evaluation assistant for a PMO.
Score new ideas on Feasibility, Cost, Impact, and Risk using the rubric below.
Return STRICT JSON matching the schema. Never include extra text outside JSON.
If compliance/safety/privacy risks lack mitigation, set recommendation to "No-Go".

Rubric:
- Feasibility (1-5): 1=major unknowns; 3=known tech w/ gaps; 5=proven approach, clear owner
- Cost (1-5; higher=more expensive): 1=minimal; 3=moderate; 5=significant budget/vendor
- Impact (1-5): 1=limited; 3=dept-level benefit; 5=org-level, measurable KPIs
- Risk (1-5; higher=more risk): 1=low; 3=some risk w/ mitigations; 5=high (safety, regulatory, security, reputation)

Output JSON Schema:
{
  "feasibility": 1-5 integer,
  "cost": 1-5 integer,
  "impact": 1-5 integer,
  "risk": 1-5 integer,
  "confidence": 0-100 integer,
  "summary": "2-3 sentence executive summary and recommendation",
  "rationales": {
    "feasibility": "max 2 sentences",
    "cost": "max 2 sentences",
    "impact": "max 2 sentences",
    "risk": "max 2 sentences"
  },
  "assumptions": ["bullet", "bullet"],
  "recommendation": "Go" | "Revise" | "No-Go"
}"""

    user_prompt = payload_text

    last_err = None
    for attempt in range(5):
        try:
            # First try: JSON mode (response_format)
            resp = client.chat.completions.create(
                model=MODEL_NAME,
                temperature=0.1,
                max_tokens=500,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            content = resp.choices[0].message.content.strip().strip("`")
            return _parse_json_strict_or_best_effort(content)
        except Exception as e1:
            last_err = e1
            # Second try: without response_format (some accounts/models may not support it)
            try:
                resp2 = client.chat.completions.create(
                    model=MODEL_NAME,
                    temperature=0.1,
                    max_tokens=500,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                )
                content2 = resp2.choices[0].message.content.strip().strip("`")
                if content2.startswith("json"):
                    content2 = content2[4:].strip()
                return _parse_json_strict_or_best_effort(content2)
            except Exception as e2:
                last_err = e2
                wait = min(2 ** attempt, 8)  # 1,2,4,8,8
                st.info(f"Model retry in {wait}s (attempt {attempt+1}/5)…")
                time.sleep(wait)

    raise RuntimeError(f"OpenAI call failed after retries: {last_err}")

# ---------- Helpers ----------
def idea_packet_text(title, problem, target_users, success_metric, dependencies, compliance_notes):
    return f"""IDEA PACKET:
Title: {title}
Problem: {problem}
Target Users: {target_users}
Success Metric: {success_metric}
Dependencies/Constraints: {dependencies}
Compliance/Safety Notes: {compliance_notes}"""

# ---------- UI ----------
def main():
    conn = init_db()

    st.title("💡 AI Idea Evaluator")
    tab_submit, tab_dash = st.tabs(["Submit Idea", "Dashboard"])

    # ---- Submit Idea ----
    with tab_submit:
        st.subheader("Submit a new idea")
        with st.form("idea_form", clear_on_submit=False):
            col1, col2 = st.columns(2)
            with col1:
                title = st.text_input("Idea Title")
                target_users = st.text_input("Target Users or Beneficiaries")
                submitted_by = st.text_input("Your name & team")
            with col2:
                success_metric = st.text_input("Success metric (what proves it worked?)")

            problem = st.text_area("Problem Statement (one or two sentences)", height=100)
            dependencies = st.text_area("Dependencies or constraints we should know", height=100)
            compliance_notes = st.text_area("Compliance, safety, privacy, or regulatory considerations", height=100)

            submitted = st.form_submit_button("Evaluate Idea")
        
        if submitted:
            if not title or not problem:
                st.warning("Please provide at least a Title and Problem Statement.")
            else:
                payload = idea_packet_text(title, problem, target_users, success_metric, dependencies, compliance_notes)
                with st.spinner("Scoring your idea with AI..."):
                    data = call_openai_idea_eval(payload)

                # Extract
                feasibility = int(data.get("feasibility", 3))
                cost = int(data.get("cost", 3))
                impact = int(data.get("impact", 3))
                risk = int(data.get("risk", 3))
                confidence = int(data.get("confidence", 70))
                summary = data.get("summary", "")
                rats = data.get("rationales", {})
                r_feas = rats.get("feasibility", "")
                r_cost = rats.get("cost", "")
                r_imp = rats.get("impact", "")
                r_risk = rats.get("risk", "")
                assumptions = data.get("assumptions", [])
                recommendation = data.get("recommendation", "Revise")

                weighted = compute_weighted_score(feasibility, cost, impact, risk)
                record = {
                    "title": title,
                    "problem": problem,
                    "target_users": target_users,
                    "success_metric": success_metric,
                    "dependencies": dependencies,
                    "compliance_notes": compliance_notes,
                    "submitted_by": submitted_by,
                    "submitted_on": datetime.utcnow().isoformat(timespec="seconds") + "Z",
                    "feasibility": feasibility,
                    "cost": cost,
                    "impact": impact,
                    "risk": risk,
                    "confidence": confidence,
                    "summary": summary,
                    "rationale_feasibility": r_feas,
                    "rationale_cost": r_cost,
                    "rationale_impact": r_imp,
                    "rationale_risk": r_risk,
                    "assumptions": assumptions,
                    "recommendation": recommendation,
                    "weighted_score": weighted
                }
                insert_record(conn, record)

                # ---- Pretty Result Panel ----
                st.success("Idea evaluated and saved.")

                rec_color = {"Go": "green", "Revise": "orange", "No-Go": "red"}.get(recommendation, "gray")
                rec_emoji = {"Go": "✅", "Revise": "✏️", "No-Go": "⛔"}.get(recommendation, "ℹ️")

                st.markdown(
                    f"### {rec_emoji} Recommendation: <span style='color:{rec_color}'>{recommendation}</span>",
                    unsafe_allow_html=True
                )
                st.write(summary)

                colA, colB, colC = st.columns(3)
                with colA:
                    st.metric("Weighted Score", f"{weighted}%")
                with colB:
                    st.metric("Confidence", f"{confidence}%")
                with colC:
                    st.metric("Risk (1–5, lower better)", risk)

                st.markdown("#### Scores")
                sb1, sb2 = st.columns(2)
                with sb1:
                    st.write("Feasibility")
                    st.progress(feasibility/5.0)
                    st.write("Impact")
                    st.progress(impact/5.0)
                with sb2:
                    st.write("Cost (lower is better)")
                    st.progress((6-cost)/5.0)
                    st.write("Risk (lower is better)")
                    st.progress((6-risk)/5.0)

                tab1, tab2, tab3, tab4 = st.tabs(["Feasibility", "Cost", "Impact", "Risk"])
                with tab1:
                    st.write(r_feas or "-")
                with tab2:
                    st.write(r_cost or "-")
                with tab3:
                    st.write(r_imp or "-")
                with tab4:
                    st.write(r_risk or "-")

                if assumptions:
                    st.markdown("#### Assumptions")
                    st.write("\n".join([f"- {a}" for a in assumptions]))

    # ---- Dashboard ----
    with tab_dash:
        st.subheader("Dashboard")
        recs = ["All", "Go", "Revise", "No-Go"]
        rec_filter = st.selectbox("Filter by AI Recommendation", recs, index=0)
        where, params = "", ()
        if rec_filter != "All":
            where = "recommendation = ?"
            params = (rec_filter,)
        df = fetch_records(conn, where, params)

        if not df.empty:
            colA, colB, colC, colD = st.columns(4)
            with colA:
                st.metric("Total Ideas", len(df))
            with colB:
                avg_ws = round(df["weighted_score"].mean(), 1)
                st.metric("Avg Weighted Score", f"{avg_ws}%")
            with colC:
                st.metric("Go", int((df["recommendation"]=="Go").sum()))
            with colD:
                st.metric("Revise / No-Go", int((df["recommendation"].isin(["Revise","No-Go"])).sum()))

            show_df = df[[
                "id","submitted_on","title","submitted_by",
                "feasibility","cost","impact","risk",
                "weighted_score","recommendation","summary"
            ]].sort_values("submitted_on", ascending=False).rename(columns={
                "submitted_on": "Submitted",
                "title": "Idea",
                "submitted_by": "By",
                "feasibility": "Feasibility",
                "cost": "Cost (1-5 high=bad)",
                "impact": "Impact",
                "risk": "Risk (1-5 high=bad)",
                "weighted_score": "Weighted Score",
                "recommendation": "AI Rec",
                "summary": "Summary",
            })

            # Add a clean percent text column so it always renders like '72%'
            show_df["Weighted %"] = show_df["Weighted Score"].round().astype(int).astype(str) + "%"

            st.caption("Legend: ✅ Go   ✏️ Revise   ⛔ No-Go")

            st.dataframe(
                show_df[[
                    "Submitted","Idea","By",
                    "Feasibility","Impact","Cost (1-5 high=bad)","Risk (1-5 high=bad)",
                    "Weighted Score","Weighted %","AI Rec","Summary"
                ]],
                use_container_width=True,
                column_config={
                    "Feasibility": st.column_config.NumberColumn(format="%.0f"),
                    "Impact": st.column_config.NumberColumn(format="%.0f"),
                    "Cost (1-5 high=bad)": st.column_config.NumberColumn(format="%.0f"),
                    "Risk (1-5 high=bad)": st.column_config.NumberColumn(format="%.0f"),
                    "Weighted Score": st.column_config.ProgressColumn(
                        "Weighted Score", min_value=0, max_value=100
                    ),
                    # Force as text so it's always '72%'
                    "Weighted %": st.column_config.TextColumn("Weighted %"),
                    "AI Rec": st.column_config.TextColumn("AI Rec"),
                    "Summary": st.column_config.TextColumn("Summary", width="large"),
                }
            )

            st.download_button(
                "Download as CSV",
                data=show_df.to_csv(index=False).encode("utf-8"),
                file_name="ideas_export.csv",
                mime="text/csv"
            )
        else:
            st.info("No ideas yet. Submit one in the 'Submit Idea' tab.")

if __name__ == "__main__":
    main()
