import os
import requests
import gradio as gr
import uuid
from typing import List, Tuple, Any
import logging

from pydantic_ai.messages import BinaryContent
from opentelemetry import trace

from multimodal_moderation.env import USER_API_KEY, API_BASE_URL
from multimodal_moderation.tracing import setup_tracing, get_tracer, add_media_to_span
from multimodal_moderation.agents.customer_agent import customer_agent
from multimodal_moderation.utils import detect_file_type


# -------------------------------------------------------------------
# Logging & Tracing setup
# -------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

setup_tracing()
tracer = get_tracer(__name__)

MAX_FILE_SIZE_MB = 5
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024


# -------------------------------------------------------------------
# Moderation config
# -------------------------------------------------------------------
MODERATION_CONFIG = {
    "text": {
        "endpoint": f"{API_BASE_URL}/api/v1/moderate_text",
        "unsafe_flags": ["is_unfriendly", "is_unprofessional", "contains_pii"],
    },
    "image": {
        "endpoint": f"{API_BASE_URL}/api/v1/moderate_image_file",
        "unsafe_flags": ["contains_pii", "is_disturbing", "is_low_quality"],
    },
    "video": {
        "endpoint": f"{API_BASE_URL}/api/v1/moderate_video_file",
        "unsafe_flags": ["contains_pii", "is_disturbing", "is_low_quality"],
    },
    "audio": {
        "endpoint": f"{API_BASE_URL}/api/v1/moderate_audio_file",
        "unsafe_flags": ["is_unfriendly", "is_unprofessional", "contains_pii"],
    },
}


# -------------------------------------------------------------------
# Moderation helpers
# -------------------------------------------------------------------
def _call_text_moderation(text: str, span: trace.Span):
    response = requests.post(
        MODERATION_CONFIG["text"]["endpoint"],
        headers={"Authorization": f"Bearer {USER_API_KEY}"},
        json={"text": text},
    )
    if not response.ok:
        raise RuntimeError(response.text)

    span.set_attributes(
        {
            "input.text.length": len(text),
        }
    )

    result = response.json()
    return result, result["rationale"], "text", "text/plain"


def _call_media_moderation(media: str, span: trace.Span):
    mime_type = detect_file_type(media, context=media)
    content_type = mime_type.split("/")[0]

    file_size = os.path.getsize(media)
    if file_size > MAX_FILE_SIZE_BYTES:
        raise ValueError("File too large")

    with open(media, "rb") as f:
        response = requests.post(
            MODERATION_CONFIG[content_type]["endpoint"],
            headers={"Authorization": f"Bearer {USER_API_KEY}"},
            files={"file": f},
        )

    if not response.ok:
        raise RuntimeError(response.text)

    add_media_to_span(span, media, f"{content_type}_moderation", 0)

    result = response.json()
    feedback = result["rationale"]

    if content_type == "audio" and "transcription" in result:
        feedback = f"Transcription: \"{result['transcription']}\"\n\n{feedback}"

    return result, feedback, content_type, mime_type


def check_content_safety(*, text=None, media=None):
    with tracer.start_as_current_span("moderate_text") as span:

        if text is not None:
            result, feedback, content_type, mime_type = _call_text_moderation(text, span)
        elif media is not None:
            result, feedback, content_type, mime_type = _call_media_moderation(media, span)
        else:
            raise ValueError("Must provide text or media")

        span.set_attributes({f"output.{k}": v for k, v in result.items()})
        span.update_name(f"moderate_{content_type}")

    for flag in MODERATION_CONFIG[content_type]["unsafe_flags"]:
        if result[flag]:
            return False, feedback, mime_type

    return True, feedback, mime_type


# -------------------------------------------------------------------
# Chat session with tracing
# -------------------------------------------------------------------
class ChatSessionWithTracing:
    def __init__(self):
        self.session_id = str(uuid.uuid4())
        self.conversation_span = tracer.start_span(
            "conversation",
            attributes={"session.id": self.session_id},
        )

    async def chat_with_gemini(self, message, history, past_messages):
        with tracer.start_as_current_span(
            "chat_turn",
            context=trace.set_span_in_context(self.conversation_span),
        ) as span:

            prompt_parts: List[str | BinaryContent] = [
                "This is the next message from the support agent:"
            ]

            safety_message = ""

            for key, value in message.items():
                if key == "text" and value:
                    is_safe, safety_message, _ = check_content_safety(text=value)
                    if not is_safe:
                        feedback = f"⚠️ Content flagged: {safety_message}"
                        span.set_attribute("feedback", feedback)
                        return (
                            "[Content blocked by moderation]",
                            past_messages,
                            feedback,
                        )
                    prompt_parts.append(value)

                elif key == "files" and value:
                    for file_path in value:
                        is_safe, safety_message, mime_type = check_content_safety(
                            media=file_path
                        )
                        if not is_safe:
                            feedback = f"⚠️ Content flagged: {safety_message}"
                            return (
                                "[Content blocked by moderation]",
                                past_messages,
                                feedback,
                            )

                        with open(file_path, "rb") as f:
                            file_bytes = f.read()

                        prompt_parts.append(
                            BinaryContent(
                                data=file_bytes,
                                media_type=mime_type,
                            )
                        )

            with tracer.start_as_current_span("llm_customer"):
                result = await customer_agent.run(
                    prompt_parts,
                    message_history=past_messages,
                )

            return result.output, result.all_messages(), safety_message

    def end_conversation(self):
        if self.conversation_span:
            self.conversation_span.end()
        return "Conversation ended."


# -------------------------------------------------------------------
# Gradio UI
# -------------------------------------------------------------------
def create_chat_interface() -> gr.Blocks:
    chat_session = ChatSessionWithTracing()

    with gr.Blocks(title="ACME Customer Service Training Agent") as demo:
        past_messages_state = gr.State([])

        feedback_display = gr.Textbox(
            label="💬 Moderation Feedback",
            interactive=False,
            render=False,
            lines=8,
        )

        with gr.Row():
            with gr.Column(scale=3):
                gr.ChatInterface(
                    fn=chat_session.chat_with_gemini,
                    type="messages",
                    multimodal=True,
                    editable=False,
                    textbox=gr.MultimodalTextbox(
                        file_count="multiple",
                        file_types=["image", "video", "audio"],
                        sources=["upload", "microphone"],
                        placeholder="Type a message or upload media…",
                    ),
                    chatbot=gr.Chatbot(
                        type="messages",
                        show_copy_button=True,
                        height="75vh",
                    ),
                    additional_inputs=[past_messages_state],
                    additional_outputs=[past_messages_state, feedback_display],
                )

            with gr.Column(scale=1):
                feedback_display.render()
                end_button = gr.Button("📞 End Conversation")
                end_status = gr.Textbox(visible=False)

        end_button.click(fn=chat_session.end_conversation, outputs=end_status)

    return demo


def main():
    demo = create_chat_interface()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)


if __name__ == "__main__":
    main()
