"""
Eleven AI Agents: one Streamlit website for all 11 LangGraph agents in agents_code/.

Run locally:   streamlit run streamlit_app.py
Deploy:        share.streamlit.io  ->  main file: streamlit_app.py
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
import threading
import uuid
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).parent
AGENTS_DIR = ROOT / "agents_code"
GUIDE_DIR = ROOT / "guide" / "agents"

MODELS = {
    "claude-sonnet-5-5": "Sonnet 5.5 · balanced (recommended)",
    "claude-haiku-5-5": "Haiku 5.5 · fastest, cheapest",
    "claude-opus-5-5": "Opus 5.5 · strongest",
}

AGENTS = {
    "01_sql_data_analyst":        ("🗄️", "SQL Data Analyst", "Ask questions about a sales database in plain words. It writes the SQL, runs it, and fixes its own mistakes."),
    "02_csv_excel_data_analyst":  ("📊", "CSV / Excel Analyst", "Upload a CSV or Excel file and ask questions. It writes pandas code and returns a table."),
    "03_bi_dashboard_insights":   ("📈", "BI Dashboard Insights", "Explains why KPIs moved: ranks the changes, tests hypotheses, writes a narrative for a manager."),
    "04_customer_support":        ("🎧", "Customer Support", "Chat bot that routes by intent: order status, returns, tech help, billing, or hand-off to a human."),
    "05_hr_resume_screener":      ("🧑‍💼", "HR Resume Screener", "Builds a rubric from a job description, removes personal info, scores every resume in parallel."),
    "06_finance_expense_auditor": ("🧾", "Finance Expense Auditor", "Checks expense lines against policy and pauses for a manager to approve or reject flagged lines."),
    "07_marketing_content":       ("✍️", "Marketing Content Writer", "Writes copy from a brief, critiques it against the brand guide, and revises until it passes."),
    "08_legal_document_reviewer": ("⚖️", "Legal Document Reviewer", "Reviews a contract against a checklist and backs every finding with a verified quote."),
    "09_healthcare_intake":       ("🩺", "Healthcare Intake", "Pre-visit intake chat with safety screening, personal-info scrubbing and escalation. Demo only."),
    "10_devops_incident_triage":  ("🚨", "DevOps Incident Triage", "Gathers metrics, logs and changes for an alert, rates severity, and asks approval before acting."),
    "11_ecommerce_recommender":   ("🛍️", "E-commerce Recommender", "Shopping assistant that remembers your sizes, budget and brands across visits."),
}

st.set_page_config(page_title="Eleven AI Agents", page_icon="🤖", layout="wide")

_IMPORT_LOCK = threading.Lock()


# ---------------------------------------------------------------- helpers
def secret(key: str) -> str:
    try:
        return str(st.secrets.get(key, "") or os.environ.get(key, ""))
    except Exception:
        return os.environ.get(key, "")


def session_dir() -> Path:
    if "tmp" not in st.session_state:
        st.session_state.tmp = Path(tempfile.mkdtemp(prefix="agents_"))
    return st.session_state.tmp


def _exec_file(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod          # needed so pydantic can resolve the agent's own types
    spec.loader.exec_module(mod)
    return mod


def load_agent(folder: str, api_key: str, model: str):
    """Import agents_code/<folder>/agent.py fresh for this browser session.

    Each agent reads its API key, model and file paths from environment variables
    when it is imported, so we set them under a lock, import, then put them back.
    That keeps one visitor's key and chat memory separate from another's.
    """
    cache = st.session_state.setdefault("agents", {})
    key = (folder, api_key, model)
    if key in cache:
        return cache[key]

    tmp = session_dir()
    agent_dir = AGENTS_DIR / folder
    env = {
        "ANTHROPIC_API_KEY": api_key,
        "MODEL": model,
        "CHECKPOINT_DB": str(tmp / f"{folder}_checkpoints.db"),
        "STORE_FILE": str(tmp / f"{folder}_memory.json"),
    }
    with _IMPORT_LOCK:
        saved = {k: os.environ.get(k) for k in env}
        os.environ.update(env)
        sys.path.insert(0, str(agent_dir))
        try:
            if folder == "01_sql_data_analyst":
                db = agent_dir / "sales.db"
                if not db.exists():
                    _exec_file(agent_dir / "seed_db.py", f"seed_{uuid.uuid4().hex}").seed(str(db))
            mod = _exec_file(agent_dir / "agent.py", f"agent_{folder}_{uuid.uuid4().hex}")
        finally:
            sys.path.remove(str(agent_dir))
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

    if folder == "06_finance_expense_auditor":
        mod.graph = mod.open_graph()
    if folder == "11_ecommerce_recommender":
        mod.load_store()
    cache[key] = mod
    return mod


def friendly_error(exc: Exception) -> None:
    msg = str(exc)
    low = msg.lower()
    if "authentication" in low or "invalid x-api-key" in low or "401" in low:
        st.error("Your Anthropic API key was rejected. Check it in the sidebar.")
    elif "credit balance" in low or "billing" in low:
        st.error("Your Anthropic account has no credit left. Add credit at console.anthropic.com.")
    elif "not_found" in low and "model" in low:
        st.error("That model name isn't available on your account. Pick another model in the sidebar.")
    elif "rate" in low and "limit" in low:
        st.error("Too many requests right now. Wait a minute and try again.")
    else:
        st.error(f"Something went wrong: {msg[:500]}")


def run(fn, *args, spinner="Thinking...", **kwargs):
    try:
        with st.spinner(spinner):
            return fn(*args, **kwargs)
    except Exception as exc:  # show the error in the page instead of crashing it
        friendly_error(exc)
        return None


def data(folder: str, name: str) -> str:
    return (AGENTS_DIR / folder / "data" / name).read_text(encoding="utf-8")


def json_box(label: str, text: str, key: str, height: int = 260):
    raw = st.text_area(label, text, height=height, key=key)
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        st.error(f"That isn't valid JSON: {exc}")
        return None


def chat_history(key: str) -> list:
    return st.session_state.setdefault(key, [])


def show_chat(key: str) -> None:
    for role, text, extra in chat_history(key):
        with st.chat_message(role):
            st.markdown(text)
            if extra:
                st.caption(extra)


def reset_button(*keys: str, label: str = "🔄 New conversation") -> None:
    if st.button(label):
        for k in keys:
            st.session_state.pop(k, None)
        st.rerun()


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown("## 🤖 Eleven AI Agents")
    st.caption("From beginner to advanced, built with LangGraph + Claude")

    folder = st.radio(
        "Pick an agent",
        list(AGENTS),
        format_func=lambda f: f"{AGENTS[f][0]} {f[:2]} · {AGENTS[f][1]}",
        key="agent_choice",
    )

    st.divider()
    st.markdown("#### 🔑 Anthropic API key")
    owner_key = secret("ANTHROPIC_API_KEY")
    if owner_key:
        st.success("Key loaded from app secrets", icon="✅")
        api_key = owner_key
    else:
        api_key = st.text_input("API key", type="password", placeholder="sk-ant-...",
                                label_visibility="collapsed",
                                help="Stays in this browser session only. Never saved.").strip()
        st.caption("[Get a key](https://console.anthropic.com/settings/keys)")

    model = st.selectbox("Model", list(MODELS), format_func=lambda m: MODELS[m])


# ---------------------------------------------------------------- header
icon, title, blurb = AGENTS[folder]
st.title(f"{icon} {title}")
st.caption(blurb)

tab_use, tab_how = st.tabs(["▶️ Use it", "🧠 How it works"])

with tab_how:
    readme = AGENTS_DIR / folder / "README.md"
    spec = GUIDE_DIR / f"{folder}.md"
    if readme.exists():
        st.markdown(readme.read_text(encoding="utf-8"))
    if spec.exists():
        with st.expander("📄 Full agent spec (from the guide)"):
            st.markdown(spec.read_text(encoding="utf-8"))

with tab_use:
    if not api_key:
        st.info("👈 Paste your Anthropic API key in the sidebar to start.")
        st.stop()
    try:
        agent = load_agent(folder, api_key, model)
    except Exception as exc:
        friendly_error(exc)
        st.stop()

    # ============================================================ 01 SQL
    if folder == "01_sql_data_analyst":
        with st.expander("🗂️ Database tables"):
            st.code(agent.get_schema(agent.DB_PATH))
        hist = "h01"
        show_chat(hist)
        q = st.chat_input("e.g. Which 5 products had the highest revenue last month?")
        if q:
            chat_history(hist).append(("user", q, None))
            out = run(agent.ask, q, spinner="Writing and running SQL...")
            if out:
                chat_history(hist).append(("assistant", out["answer"], f"attempts: {out.get('attempts')}"))
            st.rerun()
        reset_button(hist)

    # ============================================================ 02 CSV / Excel
    elif folder == "02_csv_excel_data_analyst":
        up = st.file_uploader("Upload CSV or Excel (or use the sample)", type=["csv", "xlsx", "xls"])
        if up:
            path = session_dir() / up.name
            path.write_bytes(up.getvalue())
        else:
            path = session_dir() / "sample_sales.csv"
            if not path.exists():
                agent.make_sample(str(path))
            st.caption("Using the built-in sample_sales.csv")
        sheet = None
        if path.suffix.lower() in (".xlsx", ".xls"):
            sheets = pd.ExcelFile(path).sheet_names
            sheet = st.selectbox("Sheet", sheets)
        with st.expander("👀 Preview"):
            st.dataframe(agent.load_df(str(path), sheet).head(20), width="stretch")
        q = st.text_input("Your question", placeholder="Average order amount by region")
        if st.button("Analyse", type="primary", disabled=not q):
            r = run(agent.analyse, str(path), q, sheet, spinner="Analysing...")
            if r:
                st.markdown(r["summary"])
                if r.get("table"):
                    df = pd.DataFrame(r["table"])
                    st.dataframe(df, width="stretch")
                    ch = r.get("chart") or {}
                    x, y = ch.get("x"), ch.get("y")
                    if x in df and y in df:
                        st.bar_chart(df, x=x, y=y)
                for c in r.get("caveats") or []:
                    st.warning(c)
                with st.expander("Code used"):
                    st.code(r.get("code_used", ""), language="python")

    # ============================================================ 03 BI insights
    elif folder == "03_bi_dashboard_insights":
        payload = json_box("Metrics (edit or keep the sample)", data(folder, "metrics.json"), "m03")
        audience = st.text_input("Who is this for?", "VP Sales")
        if st.button("Explain the numbers", type="primary", disabled=payload is None):
            out = run(agent.graph.invoke, {**payload, "audience": audience, "hypotheses": [],
                                           "driver_evidence": [], "iteration": 0},
                      spinner="Testing hypotheses...")
            if out:
                rows = [{
                    "Metric": i["metric"], "Change %": round(i["pct"], 1),
                    "Change": f"{i['delta']:+} {i['unit']}", "Explained %": i["explained_pct"],
                    "Confidence": i["confidence"],
                    "Drivers": ", ".join(f"{d['dimension']}={d['value']} ({d['contribution_pct']}%)"
                                         for d in i["drivers"]) or "none found",
                } for i in out["insights"]]
                st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
                st.subheader("Narrative")
                st.markdown(out["narrative"])
                st.caption(f"{out.get('iteration')} hypothesis round(s)")

    # ============================================================ 04 support chat
    elif folder == "04_customer_support":
        cust = st.selectbox("Signed in as", ["C-4471", "anon"],
                            help="C-4471 owns orders 88213 and 77120. 'anon' = not signed in.")
        hist = "h04"
        thread = st.session_state.setdefault("t04", uuid.uuid4().hex)
        st.caption("Try: *My order 88213 still hasn't arrived* · *I want to return order 77120* · *My speaker won't pair*")
        show_chat(hist)
        msg = st.chat_input("Type your message")
        if msg:
            chat_history(hist).append(("user", msg, None))
            out = run(agent.chat, msg, thread, None if cust == "anon" else cust)
            if out:
                extra = f"intent: {out.get('intent')} · confidence {out.get('intent_confidence') or 0:.2f}"
                if out.get("handoff"):
                    extra += " · 🙋 handed off to a human"
                chat_history(hist).append(("assistant", out["messages"][-1].content, extra))
            st.rerun()
        reset_button(hist, "t04")

    # ============================================================ 05 resume screener
    elif folder == "05_hr_resume_screener":
        jd = st.text_area("Job description", data(folder, "job_description.txt"), height=200)
        ups = st.file_uploader("Resumes as .txt (leave empty to use the 5 samples)", type=["txt"],
                               accept_multiple_files=True)
        if ups:
            resumes = [{"id": Path(u.name).stem, "text": u.getvalue().decode("utf-8", "ignore")} for u in ups]
        else:
            resumes = agent.load_resumes(str(AGENTS_DIR / folder / "data" / "resumes"))
        top = st.slider("Shortlist size", 1, max(1, len(resumes)), min(5, len(resumes)))
        if st.button(f"Screen {len(resumes)} resumes", type="primary"):
            out = run(agent.graph.invoke, {"job_description": jd, "resumes": resumes, "shortlist_size": top,
                                           "redaction_log": [], "candidate_scores": [], "unscreened": []},
                      spinner="Scoring resumes in parallel...")
            if out:
                st.markdown("**Rubric:** " + ", ".join(f"{c['name']} (weight {c['weight']})" for c in out["rubric"]))
                st.dataframe(pd.DataFrame([{"Rank": c["rank"], "Candidate": c["candidate_id"],
                                            "Score": c["total"], "Flags": ", ".join(c["flags"])}
                                           for c in out["shortlist"]]), width="stretch", hide_index=True)
                if out.get("unscreened"):
                    st.warning(f"Could not screen: {out['unscreened']}")
                st.markdown(out["report"])
                st.download_button("⬇️ Download report", out["report"], "shortlist_report.md")

    # ============================================================ 06 expense auditor
    elif folder == "06_finance_expense_auditor":
        report = json_box("Expense report", data(folder, "report.json"), "r06")
        if st.button("Audit report", type="primary", disabled=report is None):
            cfg = {"configurable": {"thread_id": f"{report['report_id']}-{uuid.uuid4().hex[:6]}"}}
            out = run(agent.graph.invoke, {**report, "rule_results": [], "anomalies": []}, cfg,
                      spinner="Checking against policy...")
            st.session_state.s06 = (cfg, out) if out else None

        state = st.session_state.get("s06")
        if state:
            cfg, out = state
            if "__interrupt__" in out:
                payload = out["__interrupt__"][0].value
                st.subheader("⏸️ Manager review needed")
                if "error" in payload:
                    st.error(payload["error"])
                packet = payload.get("packet") or {}
                info = {ln.get("line_id"): ln for ln in packet.get("lines", [])}
                options = ["approve", "reject", "return_to_employee"]
                decisions = {}
                # the agent's own list of flagged lines decides which lines need an answer
                for line_id in payload.get("needs_review") or list(info):
                    ln = info.get(line_id, {})
                    with st.container(border=True):
                        st.markdown(f"**{line_id}**: {ln.get('summary', 'Flagged for review')}")
                        rec = ln.get("recommendation") if ln.get("recommendation") in options else "approve"
                        if ln:
                            st.caption(f"Policy: {ln.get('policy_refs')} · AI recommends: **{rec}**")
                        dec = st.radio("Decision", options, index=options.index(rec), horizontal=True,
                                       key=f"d06_{line_id}")
                        note = st.text_input("Note (optional)", key=f"n06_{line_id}")
                        decisions[line_id] = {"decision": dec, "note": note or None}
                if packet.get("overall_note"):
                    st.info(packet["overall_note"])
                manager = st.text_input("Manager ID", "M-77")
                if st.button("✅ Submit decisions", type="primary"):
                    res = run(agent.graph.invoke,
                              agent.Command(resume={"decisions": decisions, "manager": manager}), cfg)
                    if res:
                        st.session_state.s06 = (cfg, res)
                        st.rerun()
            else:
                st.success(f"Status: {out.get('status')}")
                st.dataframe(pd.DataFrame([{"Line": v["line_id"], "Decision": v["decision"],
                                            "Policy": str(v["policy_refs"]), "By": v["decided_by"],
                                            "Reason": v["reason"]} for v in out["verdicts"]]),
                             width="stretch", hide_index=True)

    # ============================================================ 07 marketing
    elif folder == "07_marketing_content":
        which = st.radio("Brief", ["brief.json", "brief_social.json"], horizontal=True)
        brief = json_box("Brief", data(folder, which), f"b07_{which}", 220)
        with st.expander("Brand guide"):
            guide = json_box("Brand guide", data(folder, "brand_guide.json"), "g07", 220)
        rounds = st.slider("Max revision rounds", 1, 5, 3)
        if st.button("Write copy", type="primary", disabled=brief is None or guide is None):
            out = run(agent.graph.invoke, {"brief": brief, "brand_guide": guide, "max_rounds": rounds,
                                           "history": [], "round": 0}, spinner="Writing and critiquing...")
            if out:
                if out.get("refused"):
                    st.error(f"Refused: {out['refused']}")
                else:
                    fc = out["final_copy"]
                    with st.container(border=True):
                        st.markdown(f"### {fc['headline']}")
                        if fc.get("preheader"):
                            st.caption(fc["preheader"])
                        if fc.get("body"):
                            st.markdown(fc["body"])
                        st.button(fc["cta"], disabled=True)
                    (st.success if out.get("passed") else st.warning)(
                        "Passed all brand checks" if out.get("passed") else "Best version after max rounds")
                    with st.expander("Revision history"):
                        for h in out["history"]:
                            c = h["critique"]
                            st.markdown(f"**Round {h['round']}**, score {c['score']}, hard fails: {c['hard_fails']}")
                            for n in c["notes"][:5]:
                                st.caption(f"• {n}")

    # ============================================================ 08 legal
    elif folder == "08_legal_document_reviewer":
        up = st.file_uploader("Contract (.md, .txt or .pdf), or leave empty for the sample MSA",
                              type=["md", "txt", "pdf"])
        if up:
            doc = session_dir() / up.name
            doc.write_bytes(up.getvalue())
        else:
            doc = AGENTS_DIR / folder / "data" / "sample_msa.md"
            with st.expander("Sample contract"):
                st.markdown(doc.read_text(encoding="utf-8"))
        side = st.radio("We are the", ["customer", "vendor"], horizontal=True)
        checklist = json.loads(data(folder, "checklist.json"))
        if st.button("Review contract", type="primary"):
            out = run(agent.graph.invoke, {"document_path": str(doc), "checklist": checklist, "our_side": side,
                                           "candidate_findings": [], "verify_rounds": 0},
                      spinner="Reading clauses and verifying quotes...")
            if out:
                rows = []
                for f in out["findings"]:
                    c = f["citations"][0] if f["citations"] else None
                    rows.append({"Item": f["item_id"], "Risk": f["risk"], "Confidence": round(f["confidence"], 2),
                                 "Answer": f["answer"],
                                 "Where": f"Section {c['section']}, p. {c['page']}" if c else "no verified quote"})
                st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
                if out["unresolved"]:
                    st.warning(f"Unresolved: {out['unresolved']}")
                st.markdown(out["report_md"])
                st.download_button("⬇️ Download memo", out["report_md"], "review_memo.md")

    # ============================================================ 09 healthcare
    elif folder == "09_healthcare_intake":
        st.warning("Demo only. This is not medical advice. In an emergency call 1122 (Pakistan) or your local number.")
        hist, thread = "h09", st.session_state.setdefault("t09", uuid.uuid4().hex)
        if not chat_history(hist):
            chat_history(hist).append(("assistant", agent.CONSENT, None))
        show_chat(hist)
        done = st.session_state.get("done09")
        if done:
            st.info(done)
        msg = st.chat_input("Your answer", disabled=bool(done))
        if msg:
            chat_history(hist).append(("user", msg, None))
            out = run(agent.graph.invoke, {"messages": [agent.HumanMessage(msg)], "patient_token": "PT-demo",
                                           "appointment_type": "general", "red_flags": [], "turns": 0},
                      {"configurable": {"thread_id": thread}})
            if out:
                chat_history(hist).append(("assistant", out["messages"][-1].content,
                                           f"turn {out.get('turns')}"))
                st.session_state.rec09 = out.get("record", {})
                if out.get("escalation"):
                    st.session_state.done09 = "🚑 Escalated to staff. The conversation is closed."
                elif out.get("complete"):
                    st.session_state.done09 = "✅ Intake complete and sent to the nurse."
            st.rerun()
        if st.session_state.get("rec09"):
            with st.expander("📋 Record so far"):
                st.json(st.session_state.rec09)
        reset_button(hist, "t09", "done09", "rec09")

    # ============================================================ 10 devops
    elif folder == "10_devops_incident_triage":
        alert = json_box("Alert", data(folder, "alert.json"), "a10", 200)
        if st.button("Triage alert", type="primary", disabled=alert is None):
            cfg = {"configurable": {"thread_id": f"{alert['id']}-{uuid.uuid4().hex[:6]}"}}
            out = run(agent.graph.invoke, {"alert": alert, "evidence": [], "tool_calls": 0,
                                           "tool_failures": 0, "diag_rounds": 0}, cfg,
                      spinner="Gathering metrics, logs and changes...")
            st.session_state.s10 = (cfg, out) if out else None

        state = st.session_state.get("s10")
        if state:
            cfg, out = state
            if "__interrupt__" in out:
                p = out["__interrupt__"][0].value
                st.subheader(f"⏸️ Approval needed · severity {p['severity']}")
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("**Top hypothesis**")
                    st.json(p["top_hypothesis"])
                with c2:
                    st.markdown("**Proposed action**")
                    st.json(p["proposed_action"])
                a, b = st.columns(2)
                choice = None
                if a.button("✅ Approve", type="primary", width="stretch"):
                    choice = True
                if b.button("❌ Reject", width="stretch"):
                    choice = False
                if choice is not None:
                    res = run(agent.graph.invoke, agent.Command(resume={"approved": choice, "by": "oncall"}), cfg)
                    if res:
                        st.session_state.s10 = (cfg, res)
                        st.rerun()
            else:
                m = st.columns(4)
                m[0].metric("Severity", out.get("severity"))
                m[1].metric("Routed to", out.get("routed_to"))
                m[2].metric("Status", out.get("status"))
                m[3].metric("Tool calls", out.get("tool_calls"))
                st.markdown(out["triage"])

    # ============================================================ 11 recommender
    elif folder == "11_ecommerce_recommender":
        shopper = st.text_input("Shopper ID", "S-90213",
                                help="Same ID = it remembers you. Use 'New visit' to test memory across sessions.")
        hist = "h11"
        thread = st.session_state.setdefault("t11", uuid.uuid4().hex)
        st.caption("Try: *Need trail running shoes under $120, I'm a US 9 and I don't like Brand X* · "
                   "*What do you know about me?* · *Forget my budget*")
        show_chat(hist)
        msg = st.chat_input("Ask for a product")
        if msg:
            chat_history(hist).append(("user", msg, None))
            out = run(agent.graph.invoke, {"messages": [agent.HumanMessage(msg)], "shopper_id": shopper,
                                           "relaxed": False, "memory_updates": []},
                      {"configurable": {"thread_id": f"{shopper}:{thread}"}})
            if out:
                extra = f"intent: {out.get('intent')}"
                if out.get("memory_updates"):
                    extra += " · 🧠 remembered: " + "; ".join(out["memory_updates"])
                chat_history(hist).append(("assistant", out["messages"][-1].content, extra))
                agent.save_store({shopper})
            st.rerun()
        c1, c2 = st.columns(2)
        with c1:
            reset_button(hist, "t11", label="🚪 New visit (keeps memory)")
        with c2:
            with st.popover("🧠 What it remembers"):
                items = {i.key: i.value for i in agent.store.search(agent.ns(shopper))}
                st.json(items or {"memory": "nothing yet"})
