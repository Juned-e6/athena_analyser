import os
import time
import io
import json
from datetime import datetime, timedelta, time as dtime
from typing import Dict

import streamlit as st
import boto3
import pandas as pd
import requests

try:
    from openai import OpenAI  # optional
except ImportError:
    OpenAI = None


# -----------------------
# Helpers
# -----------------------
def get_ist_now() -> datetime:
    # UTC -> IST (+5:30)
    return datetime.utcnow() + timedelta(hours=5, minutes=30)


def get_athena_client(region: str, access_key: str, secret_key: str, session_token: str = ""):
    """
    Force using the creds from the UI so we don't pick up stale creds.
    """
    if not access_key or not secret_key:
        raise ValueError("Please enter AWS Access Key ID and Secret Access Key in the sidebar.")
    kwargs = {
        "region_name": region,
        "aws_access_key_id": access_key,
        "aws_secret_access_key": secret_key,
    }
    if session_token:
        kwargs["aws_session_token"] = session_token
    return boto3.client("athena", **kwargs)


def run_query(athena, query: str, database: str, output: str) -> str:
    resp = athena.start_query_execution(
        QueryString=query,
        QueryExecutionContext={"Database": database},
        ResultConfiguration={"OutputLocation": output},
    )
    return resp["QueryExecutionId"]


def wait_for_query(athena, qid: str, poll_interval: float = 2.0) -> str:
    while True:
        resp = athena.get_query_execution(QueryExecutionId=qid)
        state = resp["QueryExecution"]["Status"]["State"]
        if state in ("SUCCEEDED", "FAILED", "CANCELLED"):
            if state != "SUCCEEDED":
                reason = resp["QueryExecution"]["Status"].get("StateChangeReason", "Unknown")
                st.error(f"Query {qid} finished with state {state}: {reason}")
            return state
        time.sleep(poll_interval)


def fetch_results(athena, qid: str) -> pd.DataFrame:
    paginator = athena.get_paginator("get_query_results")
    cols, rows, first = [], [], True
    for page in paginator.paginate(QueryExecutionId=qid):
        rs = page.get("ResultSet", {})
        if first:
            cols = [c.get("Name", "") for c in rs.get("ResultSetMetadata", {}).get("ColumnInfo", [])]
            first = False
        for r in rs.get("Rows", []):
            data = [c.get("VarCharValue", "") for c in r.get("Data", [])]
            # skip header-like rows
            if len(data) == len(cols) and data != cols:
                rows.append(data)
    return pd.DataFrame(rows, columns=cols)


def df_to_csv_download(df: pd.DataFrame, filename: str, key: str):
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    st.download_button(
        label=f"Download {filename}.csv",
        data=buf.getvalue(),
        file_name=f"{filename}.csv",
        mime="text/csv",
        key=key,
    )


def post_to_slack(webhook: str, text: str):
    if not webhook:
        st.warning("Slack webhook not provided.")
        return
    r = requests.post(webhook, json={"text": text}, timeout=10)
    if r.status_code == 200:
        st.success("Posted to Slack ✅")
    else:
        st.error(f"Slack post failed: {r.text}")


def build_text_summary(cluster_df: pd.DataFrame,
                       err_df: pd.DataFrame,
                       start_str: str,
                       end_str: str) -> str:
    """
    Works with the 3-query design (metrics + combined errors).
    """
    lines = []
    lines.append(f"MEC, AU & IND Production Monitoring Update | {start_str} - {end_str} (IST)  (last window)")
    lines.append("")

    # 1) cluster-level metrics
    if cluster_df is not None and not cluster_df.empty:
        for _, row in cluster_df.iterrows():
            cluster = str(row.get("cluster_name", "Unknown"))
            total = row.get("total", 0)
            failed = row.get("failed", 0)
            cancelled = row.get("cancelled", 0)
            parsing = row.get("parsingf", row.get("ParsingF", 0))
            p95 = row.get("cpt", "")
            ex = row.get("EX", row.get("ex", ""))
            qu = row.get("QU", row.get("qu", ""))
            par = row.get("PAR", row.get("par", ""))

            lines.append(f"{cluster}:")
            lines.append("Health Status")
            lines.append(f"Total Queries Executed: {total}")
            lines.append(f"Failure: {failed}")
            lines.append(f"Cancelled: {cancelled}")
            lines.append(f"Parsing Failure: {parsing}")
            lines.append(f"Latency (P95): {cluster} : {p95} s  (Execution: {ex} , Queuing: {qu} , planning: {par})")
            lines.append("")

    # 2) top errors (combined parsing + failed/timeouts)
    if err_df is not None and not err_df.empty:
        lines.append("Top Errors:")
        for _, row in err_df.head(5).iterrows():
            ws = row.get("workspace_name", "")
            head = row.get("error_text") or row.get("error_key") or row.get("error_head") or ""
            cnt = row.get("query_cnt", "")
            lines.append(f"- {ws}: {head} ({cnt})")
        lines.append("")

    return "\n".join(lines)


# -----------------------
# Streamlit UI
# -----------------------
st.set_page_config(page_title="Athena Query History Analyzer", layout="wide")
st.title("Athena Query History Analyzer 🛰")

# === Region / Base Table (main page) ===
REGION_TO_TABLE = {
    "MEC": "freshworks_system_tables.analytics_mec_prod_query_history",
    "AU":  "freshworks_system_tables.analytics_au_prod_query_history",
    "IND": "freshworks_system_tables.analytics_ind_prod_query_history",
}

st.subheader("Region")
selected_region = st.selectbox(
    "Run against region",
    options=list(REGION_TO_TABLE.keys()),
    index=1,              # default AU (change if you prefer)
    key="region_picker",  # persists selection across reruns
)

base_table = REGION_TO_TABLE[selected_region]
st.session_state["base_table"] = base_table  # persist so later code can read it
st.caption(f"Using table: `{base_table}`")


with st.sidebar:
    st.header("AWS / Athena Config")
    DEFAULT_AWS_REGION = "us-east-1"
    DEFAULT_ATHENA_DB = "freshworks_system_tables"

    aws_region = st.text_input("AWS Region", DEFAULT_AWS_REGION, key="aws_region_input")
    aws_access_key = st.text_input("AWS Access Key ID", "", type="password", key="aws_access_key_input")
    aws_secret_key = st.text_input("AWS Secret Access Key", "", type="password", key="aws_secret_key_input")
    aws_session_token = st.text_input("AWS Session Token (optional)", "", type="password", key="aws_session_token_input")

    athena_output = st.text_input("Athena Output S3", "", key="athena_output_input")
    athena_db = st.text_input("Athena Execution DB", DEFAULT_ATHENA_DB, key="athena_db_input")

    st.markdown("---")
    st.subheader("Time Range (IST)")

    now_ist = get_ist_now()
    range_type = st.selectbox(
        "Select range",
        ["Last 6 hours", "Last 12 hours", "Last 24 hours", "Custom"],
        index=1,
    )

    if range_type != "Custom":
        if range_type == "Last 6 hours":
            start_ist = now_ist - timedelta(hours=6)
        elif range_type == "Last 12 hours":
            start_ist = now_ist - timedelta(hours=12)
        else:
            start_ist = now_ist - timedelta(hours=24)
        end_ist = now_ist
    else:
        cust_start_date = st.date_input("Start date (IST)", value=now_ist.date(), key="cust_start_date")
        cust_start_time = st.time_input("Start time (IST)", value=dtime(hour=0, minute=0), key="cust_start_time")
        cust_end_date = st.date_input("End date (IST)", value=now_ist.date(), key="cust_end_date")
        cust_end_time = st.time_input("End time (IST)", value=dtime(hour=23, minute=59), key="cust_end_time")
        start_ist = datetime.combine(cust_start_date, cust_start_time)
        end_ist = datetime.combine(cust_end_date, cust_end_time)

    st.caption(f"Current IST: {now_ist.strftime('%Y-%m-%d %H:%M:%S')}")

    st.markdown("---")
    st.subheader("Region / Base Table")

    REGION_TO_TABLE = {
        "MEC": "freshworks_system_tables.analytics_mec_prod_query_history",
        "AU":  "freshworks_system_tables.analytics_au_prod_query_history",
        "IND": "freshworks_system_tables.analytics_ind_prod_query_history",
    }
    selected_region = st.selectbox("Select region", options=list(REGION_TO_TABLE.keys()), index=1)
    base_table = REGION_TO_TABLE[selected_region]
    st.session_state["base_table"] = base_table  # persist across reruns
    st.caption(f"Using table: `{base_table}`")

    st.markdown("---")
    st.subheader("Integrations")
    slack_webhook = st.text_input("Slack Webhook URL", "")
    openai_key = st.text_input("OpenAI API Key", type="password", value=os.getenv("OPENAI_API_KEY", ""))


# -----------------------
# Build 3 independent queries (NO big CTE)
# -----------------------
start_str = start_ist.strftime("%Y-%m-%d %H:%M:%S")
end_str = end_ist.strftime("%Y-%m-%d %H:%M:%S")

# IMPORTANT: do NOT override base_table later. Use what the user selected:
base_table = st.session_state.get("base_table")
st.caption(f"[debug] base_table resolved to: {base_table}")

where_clause = f"""
DATE_ADD('hour', 5, DATE_ADD('minute', 30, added_on)) >= TIMESTAMP '{start_str}'
AND DATE_ADD('hour', 5, DATE_ADD('minute', 30, added_on)) < TIMESTAMP '{end_str}'
AND cluster_name <> 'common-cluster'
""".strip()

# 1) cluster metrics
metrics_sql = f"""
SELECT
  cluster_name,
  approx_percentile(client_perceived_time, 0.95) / 1000.0 AS cpt,
  approx_percentile(execution_time,        0.95) / 1000.0 AS EX,
  approx_percentile(queueing_time,         0.95) / 1000.0 AS QU,
  approx_percentile(parsing_time,          0.95) / 1000.0 AS PAR,
  COUNT(DISTINCT query_id) AS total,
  COUNT(DISTINCT CASE WHEN status = 'SUCCESS'        THEN query_id END) AS success,
  COUNT(DISTINCT CASE WHEN status = 'FAILED'         THEN query_id END) AS failed,
  COUNT(DISTINCT CASE WHEN status = 'TIMEDOUT'       THEN query_id END) AS timeout,
  COUNT(DISTINCT CASE WHEN status = 'PARSING_FAILED' THEN query_id END) AS parsingf,
  COUNT(DISTINCT CASE WHEN status = 'CANCELLED'      THEN query_id END) AS cancelled
FROM {base_table}
WHERE {where_clause}
GROUP BY cluster_name
ORDER BY cluster_name
"""

# 2) parsing-only errors
parsing_errors_sql = f"""
SELECT
  workspace_name,
  COALESCE(error, 'PARSING_FAILED (unclassified)') AS error_text,
  COUNT(*) AS query_cnt
FROM {base_table}
WHERE {where_clause}
  AND status = 'PARSING_FAILED'
GROUP BY workspace_name, COALESCE(error, 'PARSING_FAILED (unclassified)')
ORDER BY query_cnt DESC
"""

# 3) failed / timed out
failed_errors_sql = f"""
SELECT
  workspace_name,
  COALESCE(error, 'FAILED/TIMEDOUT (unclassified)') AS error_text,
  COUNT(*) AS query_cnt
FROM {base_table}
WHERE {where_clause}
  AND status IN ('FAILED', 'TIMEDOUT')
GROUP BY workspace_name, COALESCE(error, 'FAILED/TIMEDOUT (unclassified)')
ORDER BY query_cnt DESC
"""

REPORT_QUERIES = {
    "cluster_metrics": metrics_sql,
    "parsing_errors": parsing_errors_sql,
    "failed_errors": failed_errors_sql,
}

# -----------------------
# Run / show / summary
# -----------------------
st.markdown("### Run / Share / Analyze")
col1, col2 = st.columns(2)
run_report = col1.button("Run report now")
send_slack_btn = col2.button("Send last summary to Slack")

if "latest_dfs" not in st.session_state:
    st.session_state["latest_dfs"] = {}
if "latest_summary" not in st.session_state:
    st.session_state["latest_summary"] = ""

if run_report:
    try:
        athena = get_athena_client(aws_region, aws_access_key, aws_secret_key, aws_session_token)
    except ValueError as e:
        st.error(str(e))
        athena = None

    if athena:
        new_results: Dict[str, pd.DataFrame] = {}
        with st.spinner("Running Athena queries..."):
            for name, select_sql in REPORT_QUERIES.items():
                qid = run_query(athena, select_sql, athena_db, athena_output)
                state = wait_for_query(athena, qid)
                if state == "SUCCEEDED":
                    df = fetch_results(athena, qid)
                    new_results[name] = df
                else:
                    new_results[name] = pd.DataFrame()

        st.session_state["latest_dfs"] = new_results

        # Combine parsing + failed for summary so both show up
        pe = new_results.get("parsing_errors", pd.DataFrame())
        fe = new_results.get("failed_errors", pd.DataFrame())
        if not pe.empty and not fe.empty:
            err_df = pd.concat([pe, fe], ignore_index=True).sort_values("query_cnt", ascending=False)
        elif not pe.empty:
            err_df = pe
        else:
            err_df = fe

        summary_text = build_text_summary(new_results.get("cluster_metrics"), err_df, start_str, end_str)
        st.session_state["latest_summary"] = summary_text
        st.success("Report completed ✅")

# render tables
latest_dfs: Dict[str, pd.DataFrame] = st.session_state.get("latest_dfs", {})
if latest_dfs:
    tabs = st.tabs([f"{i+1}. {name}" for i, name in enumerate(latest_dfs.keys())])
    for idx, (tab, (name, df)) in enumerate(zip(tabs, latest_dfs.items())):
        with tab:
            st.subheader(name)
            if df.empty:
                st.info("No rows.")
            else:
                st.dataframe(df, use_container_width=True)
                df_to_csv_download(df, name, key=f"download_{name}_{idx}")
else:
    st.info("Run the report to see results here.")

# summarised report
summary_text = st.session_state.get("latest_summary", "")
if summary_text:
    st.markdown("### Summarised report")
    st.text_area("Report text", summary_text, height=250)
else:
    st.caption("Run the report to generate the summary.")

# send to slack
if send_slack_btn:
    summary_text = st.session_state.get("latest_summary", "")
    if not summary_text:
        st.warning("No summary to send. Run the report first.")
    else:
        post_to_slack(slack_webhook, summary_text)

st.markdown("---")
st.caption("Simplified 3-query version – region is selected from the dropdown; no nested CTEs.")
