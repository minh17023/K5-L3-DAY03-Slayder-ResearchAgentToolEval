"""Streamlit UI for the live Research Agent chat - Clean Gemini Style."""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Any

import streamlit as st

from chat import (
    now_iso,
    run_model_tool_loop,
    safe_slug,
    trim_history,
    write_transcript,
)
from env_loader import load_lab_env
from providers import make_provider
from tools import load_tool_declarations, to_openai_tools
from versioning import artifact_version_dict, build_artifact_version


ROOT = Path(__file__).parent
PROMPT_PATH = ROOT / "artifacts" / "system_prompt.md"
TOOLS_PATH = ROOT / "artifacts" / "tools.yaml"
TRANSCRIPTS_DIR = ROOT / "transcripts"
HISTORY_WINDOW = 5
MAX_TOOL_ROUNDS = 4
VERSION = "v3"

st.set_page_config(page_title="Slayder · Research Agent", page_icon="✨", layout="wide")
load_lab_env(ROOT)

# Sync Streamlit Secrets into os.environ for Streamlit Cloud deployment
try:
    for secret_key, secret_val in st.secrets.items():
        if isinstance(secret_val, str) and secret_key not in os.environ:
            os.environ[secret_key] = secret_val
except Exception:
    pass

st.markdown(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Google+Sans:wght@400;500;600;700&display=swap');
      * { font-family: 'Google Sans', -apple-system, BlinkMacSystemFont, Roboto, sans-serif; }

      :root { color-scheme: light; }
      body, .stApp, [data-testid="stAppViewContainer"] {
          background: #ffffff !important;
          color: #1f1f1f !important;
      }

      [data-testid="stHeader"], [data-testid="stBottom"],
      [data-testid="stBottom"] > div {
          background: #ffffff !important;
      }

      .block-container {
          max-width: 840px !important;
          padding-top: 2rem !important;
          padding-bottom: 9rem !important;
      }

      /* Sidebar */
      [data-testid="stSidebar"] {
          background-color: #f8fafd !important;
          border-right: 1px solid #e1e6ee !important;
          color: #1f1f1f !important;
      }
      [data-testid="stSidebar"] .block-container { padding-top: 1.5rem !important; }
      [data-testid="stSidebar"] label, [data-testid="stSidebar"] p,
      [data-testid="stSidebar"] span { color: #3c4043 !important; }

      .sidebar-brand {
          display: flex;
          align-items: center;
          gap: .65rem;
          margin-bottom: 0.3rem;
      }
      .brand-mark {
          font-size: 1.5rem;
          color: #1a73e8;
      }
      .brand-name {
          color: #202124;
          font-size: 1.15rem;
          font-weight: 600;
      }
      .sidebar-note {
          color: #5f6368 !important;
          font-size: 0.85rem;
          margin: 0 0 1.2rem;
      }

      /* Sidebar Buttons */
      [data-testid="stSidebar"] [data-testid="stButton"] button {
          border-radius: 20px !important;
          background: #e8f0fe !important;
          border: 1px solid #d2e3fc !important;
          color: #1967d2 !important;
          font-weight: 500 !important;
          min-height: 2.8rem !important;
      }
      [data-testid="stSidebar"] [data-testid="stButton"] button:hover {
          background: #d2e3fc !important;
          color: #174ea6 !important;
      }

      /* Hero Welcome */
      .welcome {
          padding: 4.5rem 0.5rem 2rem;
          text-align: center;
      }
      .welcome-kicker {
          font-size: 2.6rem;
          font-weight: 700;
          letter-spacing: -0.04em;
          background: linear-gradient(90deg, #4285f4, #9b72cb 50%, #d96570);
          -webkit-background-clip: text;
          background-clip: text;
          -webkit-text-fill-color: transparent;
      }
      .welcome-title {
          margin: 0.2rem 0 0;
          color: #3c4043;
          font-size: 2.2rem;
          font-weight: 600;
          letter-spacing: -0.03em;
      }
      .welcome-copy {
          margin-top: 0.8rem;
          color: #70757a !important;
          font-size: 1.05rem;
      }

      /* Suggestion Buttons */
      [data-testid="stButton"] button {
          min-height: 3.5rem;
          background: #f8fafd !important;
          border: 1px solid #e1e6ee !important;
          border-radius: 16px !important;
          color: #3c4043 !important;
          font-weight: 500 !important;
          box-shadow: 0 1px 3px rgba(60,64,67,.08);
          transition: all 0.2s ease;
      }
      [data-testid="stButton"] button:hover {
          background: #f1f3f4 !important;
          border-color: #dadce0 !important;
          color: #202124 !important;
      }
      [data-testid="stButton"] button p { color: #3c4043 !important; }

      /* Chat Messages */
      [data-testid="stChatMessage"] {
          background: transparent !important;
          border: 0 !important;
          box-shadow: none !important;
          margin-bottom: 0.8rem !important;
          padding: 0.6rem 0 !important;
      }
      [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
          background: #f1f3f4 !important;
          border-radius: 20px !important;
          padding: 0.9rem 1.3rem !important;
      }
      [data-testid="stChatMessage"] p, [data-testid="stChatMessage"] li,
      [data-testid="stChatMessage"] strong, [data-testid="stChatMessage"] span {
          color: #202124 !important;
          font-size: 1rem !important;
          line-height: 1.6 !important;
      }
      [data-testid="stChatMessage"] a {
          color: #1a73e8 !important;
          text-decoration: underline !important;
      }

      /* CHAT INPUT BAR - Sáng, rõ ràng, phong cách Gemini */
      [data-testid="stChatInput"] {
          background: #ffffff !important;
          border: 1px solid #c4c7c5 !important;
          border-radius: 28px !important;
          box-shadow: 0 2px 10px rgba(0, 0, 0, 0.08) !important;
          padding: 0.4rem 0.8rem !important;
      }
      [data-testid="stChatInput"] > div {
          background: #ffffff !important;
          color: #1f1f1f !important;
      }
      [data-testid="stChatInput"]:focus-within {
          background: #ffffff !important;
          border: 1.5px solid #1a73e8 !important;
          box-shadow: 0 4px 16px rgba(26, 115, 232, 0.15) !important;
      }
      [data-testid="stChatInput"] textarea {
          color: #1f1f1f !important;
          background: transparent !important;
          caret-color: #1f1f1f !important;
          font-size: 1.05rem !important;
          line-height: 1.5 !important;
      }
      [data-testid="stChatInput"] textarea::placeholder {
          color: #70757a !important;
          opacity: 1 !important;
      }
      [data-testid="stChatInput"] button {
          color: #1a73e8 !important;
          background: transparent !important;
          border: none !important;
      }
      [data-testid="stChatInput"] button:disabled {
          color: #bdc1c6 !important;
      }

      /* Expander */
      [data-testid="stExpander"] {
          background: #f8fafd !important;
          border: 1px solid #e1e6ee !important;
          border-radius: 12px !important;
      }
      [data-testid="stExpander"] p, [data-testid="stExpander"] summary {
          color: #444746 !important;
      }

      .chat-heading {
          color: #202124;
          font-size: 1.3rem;
          font-weight: 600;
          margin-bottom: 1.2rem;
      }
    </style>
    """,
    unsafe_allow_html=True,
)


def start_session(provider_name: str, model: str | None, version: str) -> None:
    artifact_version = build_artifact_version(version, PROMPT_PATH, TOOLS_PATH)
    selected_model = model or getattr(make_provider(provider_name), "default_model", None)
    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S%f")
    transcript_id = "_".join((safe_slug(version), safe_slug(provider_name), timestamp))
    st.session_state.history = []
    st.session_state.turns = []
    st.session_state.transcript_path = TRANSCRIPTS_DIR / f"{transcript_id}.transcript.json"
    st.session_state.transcript = {
        "transcript_id": transcript_id,
        **artifact_version_dict(artifact_version),
        "provider": provider_name,
        "model": selected_model,
        "system_prompt": str(PROMPT_PATH),
        "tools": str(TOOLS_PATH),
        "history_window": HISTORY_WINDOW,
        "max_tool_rounds": MAX_TOOL_ROUNDS,
        "created_at": now_iso(),
        "updated_at": now_iso(),
        "turns": st.session_state.turns,
    }
    st.session_state.config = (provider_name, model, version, artifact_version.artifact_version)


def show_turn(turn: dict[str, Any]) -> None:
    with st.chat_message("user"):
        st.write(turn["user"])
    with st.chat_message("assistant"):
        if turn["status"] == "provider_error":
            st.error(turn["error"])
        else:
            st.write(turn.get("assistant_text") or "Agent chưa trả lời.")
        if turn.get("tool_events"):
            with st.expander(f"🛠️ Xem thao tác của agent ({len(turn['tool_events'])} tools)"):
                for event in turn["tool_events"]:
                    st.write(f"**Tool:** `{event['tool']}`")
                    st.json({"args": event["args"], "result": event["result"]})


with st.sidebar:
    st.markdown(
        '<div class="sidebar-brand"><span class="brand-mark">✦</span>'
        '<span class="brand-name">Slayder Research</span></div>'
        '<p class="sidebar-note">Trợ lý tìm kiếm và tổng hợp thông tin.</p>',
        unsafe_allow_html=True,
    )
    reset = st.button("＋  Cuộc trò chuyện mới", use_container_width=True)
    with st.expander("Cài đặt"):
        provider_name = st.selectbox("Nguồn AI", ("openrouter", "openai", "anthropic", "gemini", "ollama"))
        model = st.text_input("Model (để trống để dùng mặc định)").strip() or None

version = VERSION

current_artifact = build_artifact_version(version, PROMPT_PATH, TOOLS_PATH)
config = (provider_name, model, version, current_artifact.artifact_version)
if reset or st.session_state.get("config") != config:
    start_session(provider_name, model, version)

sample_prompt = None
if not st.session_state.turns:
    st.markdown(
        '<div class="welcome">'
        '<div class="welcome-kicker">Chào bạn, mình là Slayder</div>'
        '<div class="welcome-title">Chúng ta nên bắt đầu từ đâu nhỉ?</div>'
        '<p class="welcome-copy">Tìm tin tức, đọc bài viết và khám phá các nghiên cứu cùng trợ lý AI của bạn.</p>'
        '</div>',
        unsafe_allow_html=True,
    )
    suggestions = (
        ("📰  Tin tức AI hôm nay", "Tin tức AI hôm nay có gì nổi bật?"),
        ("📚  Tìm paper arXiv", "Tìm paper về đánh giá research agent trên arXiv."),
        ("🐦  Xem tin X/Twitter", "Xem bài đăng gần đây của @sama trên X."),
    )
    for column, (label, prompt) in zip(st.columns(3), suggestions):
        with column:
            if st.button(label, use_container_width=True):
                sample_prompt = prompt
else:
    st.markdown('<div class="chat-heading">✦ Slayder Assistant</div>', unsafe_allow_html=True)

for turn in st.session_state.turns:
    show_turn(turn)

input_val = st.chat_input("Nhập câu hỏi hoặc yêu cầu nghiên cứu...")
if sample_prompt:
    input_val = sample_prompt

if input_val:
    user_text = input_val.strip()
    if user_text:
        system_prompt = PROMPT_PATH.read_text(encoding="utf-8")
        tools = to_openai_tools(load_tool_declarations(TOOLS_PATH))
        messages = [
            {"role": "system", "content": system_prompt},
            *trim_history(st.session_state.history, HISTORY_WINDOW),
            {"role": "user", "content": user_text},
        ]
        turn: dict[str, Any] = {
            "turn_index": len(st.session_state.turns) + 1,
            "started_at": now_iso(),
            "user": user_text,
            "status": "started",
            "assistant_text": None,
            "rounds": [],
            "tool_events": [],
        }
        with st.spinner("Agent đang xử lý..."):
            try:
                result = run_model_tool_loop(
                    provider=make_provider(provider_name),
                    messages=messages,
                    tools=tools,
                    model=model,
                    max_tool_rounds=MAX_TOOL_ROUNDS,
                )
                turn.update(result)
                st.session_state.history.extend((
                    {"role": "user", "content": user_text},
                    {"role": "assistant", "content": result["assistant_text"]},
                ))
            except Exception as exc:
                turn.update({"status": "provider_error", "error": f"{type(exc).__name__}: {exc}"})
        turn["ended_at"] = now_iso()
        st.session_state.turns.append(turn)
        st.session_state.transcript["turns"] = list(st.session_state.turns)
        write_transcript(st.session_state.transcript_path, st.session_state.transcript)
        st.rerun()
