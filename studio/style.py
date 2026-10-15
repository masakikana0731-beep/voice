"""Voice Studio / Voice Studio Live 共通の見た目。"""

import gradio as gr


def theme(primary="orange"):
    return gr.themes.Soft(primary_hue=primary, neutral_hue="slate", text_size="lg", radius_size="lg",
                          font=[gr.themes.GoogleFont("Noto Sans JP"), "Meiryo", "sans-serif"])


CSS = """
.gradio-container { max-width: 980px !important; margin: 0 auto !important; }
.tab-wrapper button { padding: 0 12px !important; font-size: 16px !important; }
/* 手順・区切りの見出し */
.step { border-left: 5px solid var(--color-accent); padding: 2px 0 2px 12px !important; margin-top: 18px !important; }
.step h3 { margin: 0 !important; }
/* カード（ひとまとまりの操作） */
.card { border: 1px solid var(--border-color-primary); border-radius: 14px !important; padding: 14px !important;
        background: var(--block-background-fill); }
/* 注意書き */
.notice { border: 2px solid #e03131; border-radius: 14px; padding: 6px 18px !important; background: rgba(224, 49, 49, .08); }
.big button, button.big { font-size: 19px !important; min-height: 56px !important; }
.muted, .muted p { opacity: .75; font-size: 14px !important; }
"""
