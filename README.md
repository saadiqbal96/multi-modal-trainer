Multimodal Moderation Training System
=====================================

This project implements a **multimodal content moderation system** for customer service training using Large Language Models (LLMs). The system analyzes **text, image, audio, and video inputs**, applies structured moderation rules, and integrates observability and evaluation tooling to support responsible AI development.

The project was completed as part of a Udacity Nanodegree and follows a rubric-driven, test-validated architecture.

🚀 Project Overview
-------------------

The goal of this project is to simulate a **real-world enterprise moderation pipeline** where customer service interactions are reviewed for:

*   Personally Identifiable Information (PII)
    
*   Unfriendly tone
    
*   Unprofessional language
    
*   Disturbing or inappropriate visual/audio content
    
*   Low-quality media
    

The system integrates:

*   **Gemini-based LLM moderation agents**
    
*   **Structured Pydantic outputs**
    
*   **Multimodal Gradio chat interface**
    
*   **OpenTelemetry tracing**
    
*   **Pydantic-based evaluation framework**
    

🧱 System Architecture
----------------------

### 1\. Structured Moderation Outputs

All moderation decisions are returned as **typed Pydantic models**, ensuring consistency and testability across modalities.

Implemented models include:

*   TextModerationResult
    
*   ImageModerationResult
    
*   AudioModerationResult
    
*   VideoModerationResult
    

Each model captures:

*   relevant safety flags (e.g. contains\_pii, is\_unfriendly)
    
*   a human-readable rationale explaining the decision
    

### 2\. Moderation Agents (LLM-Powered)

Dedicated moderation agents were implemented for each modality using pydantic\_ai.Agent:

ModalityAgentTexttext\_agent.pyImageimage\_agent.pyAudioaudio\_agent.pyVideovideo\_agent.py

Each agent:

*   Uses Gemini (or a compatible LLM) for inference
    
*   Accepts structured inputs (text or binary content)
    
*   Returns validated moderation results
    
*   Passes comprehensive unit tests
    

### 3\. Multimodal Gradio Chat Interface

A Gradio-based chat UI (gradio\_app.py) allows simulated customer service conversations with:

*   Text input
    
*   Image, audio, and video uploads
    
*   Real-time moderation before content is displayed
    
*   Moderation feedback surfaced to the user
    
*   Conversation state tracking
    

If unsafe content is detected, the system:

*   Blocks the message
    
*   Provides a clear moderation rationale
    
*   Logs the event for observability
    

### 4\. Simulated Customer Agent

A simulated LLM-powered customer agent participates in conversations to:

*   Respond contextually based on conversation history
    
*   Enable realistic moderation scenarios
    
*   Trigger moderation events for testing and evaluation
    

### 5\. Observability & Tracing

The system includes **full OpenTelemetry tracing** with spans for:

*   conversation (with session.id)
    
*   chat\_turn
    
*   moderate\_text
    
*   feedback
    

Tracing is compatible with **Arize Phoenix** and provides visibility into:

*   moderation decisions
    
*   conversation flow
    
*   blocked content events
    

### 6\. Moderation Evaluations

Moderation quality is assessed using a **Pydantic-based evaluation framework**, with test cases covering:

*   Acceptable vs unacceptable text
    
*   Safe vs unsafe images
    

Evaluations are designed to:

*   Produce structured results
    
*   Demonstrate LLM consistency
    
*   Surface expected pass/fail behavior without runtime errors
    

🧪 Testing
----------

The project includes a comprehensive test suite validating:

*   Moderation result models
    
*   All moderation agents
    
*   Gradio interface behavior
    
*   Environment configuration
    
*   Tracing instrumentation
    

> **Note:**An integration test that makes a live Gemini API call may fail locally due to shared API budget limits in the Udacity environment. This is a known infrastructure limitation and does not affect grading.

## 🔐 Environment Configuration

Sensitive credentials are not committed to the repository.

A sample configuration is provided in `env.example`:

```env
USER_API_KEY=my-api-key
GEMINI_API_KEY=voc-...
GOOGLE_GEMINI_BASE_URL=https://gemini.vocareum.com
DEFAULT_GOOGLE_MODEL=gemini-2.5-flash-lite
EVAL_JUDGE_MODEL=gemini-2.5-flash-lite
EVAL_NUM_REPEATS=5

Users should copy this file to .env for local execution and insert their own credentials if required.


---

## ▶️ Running the Application (Optional)

```markdown
To run the Gradio application locally:

```bash
uv run --env-file .env python multimodal_moderation/gradio_app.py

Then open the browser at:

http://localhost:7860


📌 Key Takeaways
----------------

This project demonstrates:

*   Responsible AI moderation design
    
*   Multimodal LLM integration
    
*   Strong typing and validation with Pydantic
    
*   Observability best practices
    
*   Test-driven development for AI systems
    

🧑‍💻 Author
------------

Saad Iqbal GitHub: [https://github.com/saadiqbal96](https://github.com/saadiqbal96)
