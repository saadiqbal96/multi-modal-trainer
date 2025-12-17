from pydantic_ai import Agent
from pydantic_ai.messages import BinaryContent

from multimodal_moderation.types.model_choice import ModelChoice
from multimodal_moderation.types.moderation_result import VideoModerationResult


MODERATION_INSTRUCTIONS = """
CONTEXT
At ACME Enterprise we strive for a friendly but professional interaction with our customers.

ROLE
You are a customer service reviewer at ACME Enterprise. You make sure that the customer
service interactions are friendly and professional.

TASK
You will receive a video. Determine if that video is appropriate for a professional customer service setting
according to the instructions below.

INSTRUCTIONS
Detect if:
- the video is disturbing in any way
- the video contains the face of a person, or any other personally-identifiable information (PII)
- a significant portion of the video is of low quality (low-resolution, blurry,
  pixelated, underexposed, overexposed, etc.). Ignore low-quality portions of the video if
  they do not make up the majority of the video.

OUTPUT
Provide a detailed rationale for your choices.
"""


video_moderation_agent = Agent(
    instructions=MODERATION_INSTRUCTIONS,
    output_type=VideoModerationResult,
)


async def moderate_video(
    model_choice: ModelChoice,
    video_source: bytes,
    media_type: str,
) -> VideoModerationResult:
    """
    Moderate a video using the configured PydanticAI agent and return a structured result.
    """
    video_input = BinaryContent(
        data=video_source,
        media_type=media_type,
    )

    prompt = "Please analyze the following video for moderation."

    result = await video_moderation_agent.run(
        [prompt, video_input],
        model=model_choice.model,
        model_settings=model_choice.model_settings,
    )

    return result.output

