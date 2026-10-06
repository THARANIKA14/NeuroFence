import streamlit as st

st.set_page_config(
    page_title="NeuroFence",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>
        .main {
            padding-top: 1rem;
        }

        .neuro-title {
            font-size: 42px;
            font-weight: 700;
            margin-bottom: 0;
        }

        .neuro-subtitle {
            font-size: 18px;
            opacity: 0.75;
            margin-bottom: 20px;
        }

        .section-title {
            font-size: 25px;
            font-weight: 600;
            margin-top: 20px;
            margin-bottom: 15px;
        }

        .info-box {
            padding: 18px;
            border-radius: 12px;
            border: 1px solid rgba(128, 128, 128, 0.3);
            margin-bottom: 15px;
        }

        .small-text {
            font-size: 14px;
            opacity: 0.7;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

import gc
import uuid
from pathlib import Path

import pandas as pd

from scanner.peer_anomaly import is_mask_buffer
from scanner.pipeline import scan_safetensors_file

# Not /tmp: on many Linux systems /tmp is RAM-backed (tmpfs).
CACHE_DIR = Path.home() / ".cache" / "neurofence"

# ============================================================
# SESSION STATE
# ============================================================

st.session_state.setdefault("result", None)   # latest scan
st.session_state.setdefault("history", [])    # all scans this session


def title_block():
    st.markdown(
        '<div class="neuro-title">🛡️ NeuroFence</div>'
        '<div class="neuro-subtitle">'
        "LLM Weight Poisoning & Backdoor Scanner</div>",
        unsafe_allow_html=True,
    )


def section(text):
    st.markdown(
        f'<div class="section-title">{text}</div>',
        unsafe_allow_html=True,
    )


def need_scan():
    if st.session_state.result is None:
        st.info("No scan yet. Upload a model on the Dashboard first.")
        return True
    return False


def fmt_params(n):
    return f"{n/1e6:.1f}M" if n >= 1e6 else f"{n:,}"


# ============================================================
# SCAN UI (shared by Dashboard and Upload Model pages)
# ============================================================

def scan_uploader(key):
    st.caption(
        "Only .safetensors is accepted. Pickle formats (.pt, .pth, .bin) "
        "can execute code when loaded, so they are blocked."
    )
    f = st.file_uploader(
        "Select a .safetensors model file",
        type=["safetensors"],
        key=f"up_{key}",
    )
    trusted = st.text_input(
        "Trusted SHA-256 (optional, e.g. from the publisher's page)",
        key=f"hash_{key}",
    ).strip()

    if f is None:
        st.info("Upload a model to begin. Scan one model at a time.")
        return

    st.write(f"**{f.name}** ({f.size / 1024 / 1024:.1f} MB)")

    if st.button("🔍 START SECURITY SCAN", type="primary",
                 use_container_width=True, key=f"scan_{key}"):
        with st.spinner("Scanning weights... (large models take a while)"):
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            tmp = CACHE_DIR / f"upload_{uuid.uuid4().hex}.safetensors"
            try:
                with open(tmp, "wb") as out:
                    out.write(f.getbuffer())  # no extra in-memory copy
                res = scan_safetensors_file(tmp, f.name, trusted or None)
            except Exception as exc:
                st.error(f"Scan failed: {exc}")
                return
            finally:
                tmp.unlink(missing_ok=True)
                gc.collect()
        st.session_state.result = res
        st.session_state.history.append(res)
        st.success("Scan complete. Open the other pages for details.")
        st.rerun()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.title("🛡️ NeuroFence")
    st.caption("LLM Weight Poisoning & Backdoor Scanner")
    st.divider()

    page = st.radio(
        "Navigation",
        [
            "Dashboard",
            "Upload Model",
            "Weight Analysis",
            "Anomaly Detection",
            "Backdoor Detection",
            "Integrity Check",
            "Risk Assessment",
            "Reports",
        ],
    )
    st.divider()
    st.markdown("### Scanner Status")
    r = st.session_state.result
    if r:
        st.success(f"Scanned: {r['file_name']}")
    else:
        st.info("No Model Scanned")

    if st.session_state.history:
        if st.button("Clear results", use_container_width=True):
            st.session_state.result = None
            st.session_state.history = []
            st.rerun()
    st.divider()
    st.caption("NeuroFence Research Prototype")

res = st.session_state.result

# ============================================================
# DASHBOARD
# ============================================================

if page == "Dashboard":
    title_block()
    st.divider()
    section("Security Dashboard")

    c1, c2, c3, c4 = st.columns(4)
    if res:
        risk = res["risk"]
        c1.metric("Model", res["file_name"])
        c2.metric("Risk Score", risk["overall_risk_score"],
                  risk["risk_level"].upper())
        c3.metric("Anomalous layers",
                  res["anomaly"]["anomalous_layers"])
        c4.metric("Scan Status", "COMPLETED")
    else:
        c1.metric("Model", "—")
        c2.metric("Risk Score", "—")
        c3.metric("Anomalous layers", "—")
        c4.metric("Scan Status", "NOT STARTED")

    st.divider()
    section("Upload LLM Model")
    scan_uploader("dash")

    if res:
        st.divider()
        section("Summary")
        a = res["anomaly"]
        w = res["worst_layer"]
        st.write(
            f"**{res['analysis']['analyzed_tensors']}** tensors, "
            f"**{fmt_params(res['analysis']['total_parameters'])}** "
            f"parameters. Worst-tensor anomaly score "
            f"**{a['model_anomaly_score']}** ({a['classification']}), "
            f"mean {a['mean_score']}; "
            f"{a['anomalous_layers']} anomalous, "
            f"{a['moderate_layers']} moderate, "
            f"{a['unscored_layers']} not scored (no peers)."
        )
        if w:
            st.write(
                f"Most suspicious tensor: `{w['layer']}` "
                f"(score {w['anomaly_score']}, {w['classification']}): "
                f"{w['reason']}."
            )
        st.warning(
            "Weight statistics only. The behavioral backdoor test was "
            "not run, so a low score is not proof the model is safe."
        )

# ============================================================
# UPLOAD MODEL
# ============================================================

elif page == "Upload Model":
    st.title("📁 Upload LLM Model")
    scan_uploader("page")

# ============================================================
# WEIGHT ANALYSIS
# ============================================================

elif page == "Weight Analysis":
    st.title("⚖️ Weight Analysis")
    if not need_scan():
        an = res["analysis"]
        s = an.get("model_summary", {})
        c1, c2, c3 = st.columns(3)
        c1.metric("Tensors analyzed", an["analyzed_tensors"])
        c2.metric("Parameters", fmt_params(an["total_parameters"]))
        c3.metric("Skipped", an["skipped_tensors"])

        rows = []
        for name, st_ in an["layer_statistics"].items():
            if is_mask_buffer(name, st_):
                continue          # constant attention masks, not weights
            rows.append({
                "tensor": name,
                "mean": st_.get("mean"),
                "std": st_.get("std"),
                "min": st_.get("min"),
                "max": st_.get("max"),
                "nan": st_.get("nan_count"),
                "inf": st_.get("inf_count"),
            })
        df = pd.DataFrame(rows)
        st.subheader("Std per tensor")
        if not df.empty:
            st.bar_chart(df.set_index("tensor")["std"])
        st.subheader("Per-tensor statistics")
        st.dataframe(df, use_container_width=True)

# ============================================================
# ANOMALY DETECTION
# ============================================================

elif page == "Anomaly Detection":
    st.title("🔎 Anomaly Detection")
    if not need_scan():
        a = res["anomaly"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Worst tensor score", a["model_anomaly_score"])
        c2.metric("Mean score", a["mean_score"])
        c3.metric("Anomalous", a["anomalous_layers"])
        c4.metric("Moderate", a["moderate_layers"])
        st.caption(
            "Each tensor is compared with the same kind of tensor in the "
            "other layers (robust z-score on std, absolute max and mean). "
            "The model score is the worst tensor, not the mean. Tensors "
            "with no peers (embeddings, final norm) are not scored, and "
            "subtle low-magnitude edits can still go undetected."
        )
        df = pd.DataFrame([
            {"tensor": r["layer"], "score": r["anomaly_score"],
             "class": r["classification"], "why": r["reason"]}
            for r in a["layer_results"]
        ])
        st.dataframe(df.head(25), use_container_width=True)

# ============================================================
# BACKDOOR DETECTION
# ============================================================

elif page == "Backdoor Detection":
    st.title("🧪 Backdoor Detection")
    st.warning(
        "Not run. Behavioral backdoor testing needs the full model "
        "folder (config.json + tokenizer) so prompts can be generated "
        "with and without trigger phrases. A single .safetensors upload "
        "only supports weight analysis."
    )
    st.write(
        "Use the trigger prompts in `datasets/trigger_tests.json` with "
        "`scanner/backdoor_detector.py` from Python once a full model "
        "folder is available."
    )

# ============================================================
# INTEGRITY CHECK
# ============================================================

elif page == "Integrity Check":
    st.title("🔐 Integrity Check")
    if not need_scan():
        i = res["integrity"]
        st.code(i["sha256"], language="text")
        status = i["status"]
        if status == "verified":
            st.success("Hash matches the trusted hash.")
        elif status == "mismatch":
            st.error("Hash does NOT match the trusted hash. "
                     "Do not use this file.")
        else:
            st.info(
                "No trusted hash was provided, so integrity could not "
                "be verified. Compare the SHA-256 above with the one "
                "published by the model's source."
            )

# ============================================================
# RISK ASSESSMENT
# ============================================================

elif page == "Risk Assessment":
    st.title("⚠️ Risk Assessment")
    if not need_scan():
        risk = res["risk"]
        c1, c2 = st.columns(2)
        c1.metric("Overall risk", risk["overall_risk_score"])
        c2.metric("Level", risk["risk_level"].upper())
        st.write(risk.get("risk_description", ""))
        st.subheader("Components")
        st.json(risk.get("component_scores", {}))
        st.caption(
            "Scored on weight statistics and the optional hash only; "
            "behavioral testing was not run. The anomaly component is the "
            "worst-tensor score, halved when no tensor is classed "
            "'anomalous' (moderate tensors are common in clean models)."
        )
        for title, key in (("Contributing factors", "contributing_factors"),
                           ("Recommendations", "recommendations")):
            items = risk.get(key) or []
            if items:
                st.subheader(title)
                for item in items:
                    st.write(f"- {item}")

# ============================================================
# REPORTS
# ============================================================

elif page == "Reports":
    st.title("📄 Reports")
    hist = st.session_state.history
    if not hist:
        st.info("No scans yet.")
    else:
        df = pd.DataFrame([{
            "model": h["file_name"],
            "size_MB": round(h["size_bytes"] / 1024 / 1024, 1),
            "params": fmt_params(h["analysis"]["total_parameters"]),
            "mean_anomaly": h["anomaly"]["model_anomaly_score"],
            "worst_tensor": (h["worst_layer"] or {}).get("anomaly_score"),
            "anomalous": h["anomaly"]["anomalous_layers"],
            "risk": h["risk"]["overall_risk_score"],
            "level": h["risk"]["risk_level"],
            "sha256": h["sha256"][:16] + "…",
        } for h in hist])
        st.subheader("Scans this session")
        st.dataframe(df, use_container_width=True)

        lines = ["NeuroFence scan report", "=" * 40]
        for h in hist:
            lines += [
                f"Model: {h['file_name']}",
                f"SHA-256: {h['sha256']}",
                f"Parameters: {h['analysis']['total_parameters']:,}",
                f"Mean anomaly: {h['anomaly']['model_anomaly_score']}",
                f"Anomalous layers: {h['anomaly']['anomalous_layers']}",
                f"Risk: {h['risk']['overall_risk_score']} "
                f"({h['risk']['risk_level']})",
                "Behavioral test: not run", "-" * 40,
            ]
        st.download_button("Download report (.txt)", "\n".join(lines),
                           file_name="neurofence_report.txt")
