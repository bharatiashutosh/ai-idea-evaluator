import streamlit as st
import sqlite3
import json
from datetime import datetime
import pandas as pd

# ---------- Config ----------
st.set_page_config(page_title="AI Idea Evaluator", page_icon="💡", layout="wide")

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
    # Invert cost and risk (1=low, 5=high -> higher is worse)
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

# ---------- OpenAI ----------
def call_openai_idea_eval(payload_text):
    from openai import OpenAI
    import os

    api_key = st.secrets.get("OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        st.error("OPENAI_API_KEY not found. Add it in Streamlit Secrets or env vars.")
        st.stop()

    client = OpenAI(api_key=api_key)

    system_prompt = """You are an Idea Evaluation assistant for a PMO. 
Score new ideas on Feasibility, Cost, Impact, and Risk using the rubric below.
Return STRICT JSON that matches the provided schema. Never include extra text outside JSON.
Calibrate to be consistent across submissions.
Gates: If compliance/safety/privacy is a concern with no mitigation, set recommendation to "No-Go".
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

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.2,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )
    content = resp.choices[0].message.content
    # Ensure we only parse JSON (some models may add backticks)
    content = content.strip().strip("`")
    if content.startswith("json"):
        content = content[4:].strip()
    try:
        data = json.loads(content)
    except Exception as e:
        raise RuntimeError(f"Failed to parse JSON from model: {e}\nRaw content: {content[:500]}")
    return data

# ---------- UI ----------
def idea_packet_text(title, problem, target_users, success_metric, dependencies, compliance_notes):
    return f"""IDEA PACKET:
Title: {title}
Problem: {problem}
Target Users: {target_users}
Success Metric: {success_metric}
Dependencies/Constraints: {dependencies}
Compliance/Safety Notes: {compliance_notes}"""

def main():
    conn = init_db()

    st.title("💡 AI Idea Evaluator")
    tab_submit, tab_dash = st.tabs(["Submit Idea", "Dashboard"])

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
                # Extract fields
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
                st.success("Idea evaluated and saved.")
                with st.expander("View AI output"):
                    st.json({
                        "scores": {
                            "feasibility": feasibility,
                            "cost": cost,
                            "impact": impact,
                            "risk": risk,
                            "weighted_score": weighted
                        },
                        "confidence": confidence,
                        "recommendation": recommendation,
                        "summary": summary,
                        "rationales": {
                            "feasibility": r_feas,
                            "cost": r_cost,
                            "impact": r_imp,
                            "risk": r_risk
                        },
                        "assumptions": assumptions
                    })

    with tab_dash:
        st.subheader("Dashboard")
        recs = ["All", "Go", "Revise", "No-Go"]
        rec_filter = st.selectbox("Filter by AI Recommendation", recs, index=0)
        where = ""
        params = ()
        if rec_filter != "All":
            where = "recommendation = ?"
            params = (rec_filter,)
        df = fetch_records(conn, where, params)

        # Compute basic stats
        if not df.empty:
            colA, colB, colC, colD = st.columns(4)
            with colA:
                st.metric("Total Ideas", len(df))
            with colB:
                st.metric("Avg Weighted Score", round(df["weighted_score"].mean(), 1))
            with colC:
                st.metric("Go", int((df["recommendation"]=="Go").sum()))
            with colD:
                st.metric("Revise / No-Go", int((df["recommendation"].isin(["Revise","No-Go"])).sum()))

            st.dataframe(
                df[["id","submitted_on","title","submitted_by","feasibility","cost","impact","risk","weighted_score","recommendation","summary"]].sort_values("submitted_on", ascending=False),
                use_container_width=True
            )
            st.download_button(
                "Download as CSV",
                data=df.to_csv(index=False).encode("utf-8"),
                file_name="ideas_export.csv",
                mime="text/csv"
            )
        else:
            st.info("No ideas yet. Submit one in the 'Submit Idea' tab.")

if __name__ == "__main__":
    main()
