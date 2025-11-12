# 🎤 Toy Backend - Interactive Voice AI Backend

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

**Real-time voice agent backend with speech-to-speech capabilities**

[Features](#-features) • [Setup](#-quick-start) • [API](#-api-endpoints) • [Configuration](#-configuration)

</div>

---

## ✨ Features

- 🎙️ **Real-time Speech-to-Text (STT)** using Cartesia's ink-whisper model
- 🔊 **Real-time Text-to-Speech (TTS)** using Cartesia's sonic-3 model with speed/volume/emotion controls
- 🤖 **LLM Integration** with Qwen/GLM models via DeepInfra
- 🔄 **WebSocket-based Media Streaming** for real-time bidirectional audio
- ⚡ **Early Interruption Detection** using partial transcripts for instant response
- 🎯 **Voice Activity Detection (VAD)** with Silero VAD
- 🌐 **Multi-language Support** (currently configured for Hindi)
- 📊 **Rich Logging** with structured logging and beautiful console output

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- API keys for:
  - Cartesia (for STT and TTS)
  - DeepInfra (for LLM)

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/inav-labs-research/toy_backend.git
cd toy_backend
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
# Or using uv:
uv pip install -r requirements.txt
```

3. **Configure `config.json`:**
   - Add your Cartesia API key for STT and TTS
   - Add your DeepInfra API key for LLM
   - Configure voice ID and language settings

4. **Run the server:**
```bash
python main.py
# Or using uv:
uv run main.py
```

The server will start on **port 5050**.

## 📡 API Endpoints

### WebSocket

- **`/api/media-stream?agent_id=shinchan`**
  - Real-time bidirectional audio streaming
  - Supports interruption signals
  - Sends LLM text and user transcripts in real-time

### REST

- **`GET /`** - Root endpoint with API information
- **`GET /health`** - Health check endpoint

## 🤖 Agents

Agents are configured in `agents.json`. The default agent is **"shinchan"**.

### Current Agent: Shinchan

- **Language**: Hindi (हिंदी)
- **Voice**: Cartesia voice with speed control (0.85x)
- **Personality**: Friendly companion for children with safety guardrails

## ⚙️ Configuration

### `config.json` Structure

```json
{
  "models": {
    "llm_model": {
      "model_provider": "qwen",
      "model_name": "zai-org/GLM-4.5",
      "api_base": "https://api.deepinfra.com/v1/openai"
    },
    "tts_model": {
      "model_provider": "cartesia",
      "model_name": "sonic-3",
      "default_voice": "d05d32ab-146f-4ddf-8000-24d3c70fa1de",
      "language": "hi",
      "speed": 0.85
    },
    "cartesia_stt": {
      "model_provider": "cartesia",
      "model_name": "ink-whisper",
      "language": "hi"
    }
  }
}
```

### Key Settings

- **TTS Speed**: `0.6` to `1.5` (default: `0.85` for slightly slower speech)
- **TTS Volume**: `0.5` to `2.0` (default: `1.0`)
- **TTS Emotion**: Optional emotion guidance (e.g., "excited", "calm", "neutral")
- **STT Language**: Configured language for transcription
- **Interruption Sensitivity**: `max_interruptions` (default: 25)

## 🏗️ Architecture

```
toy_backend/
├── app/
│   ├── api/                    # FastAPI endpoints
│   ├── agents/                 # Agent configuration loader
│   ├── factories/              # Handler factories
│   ├── media_stream_handler/   # WebSocket stream handlers
│   ├── models/
│   │   ├── language_models/    # LLM clients (Qwen, Gemini, OpenAI)
│   │   └── stt_models/         # STT clients (Cartesia, Soniox)
│   └── services/
│       ├── handlers/           # Voice handlers with interruption support
│       ├── inferencing_handlers/  # Speech-to-speech inference
│       ├── speech_processor/   # VAD and EOS detection
│       └── text_to_speech/     # TTS processors (Cartesia)
├── agents.json                 # Agent configurations
├── config.json                 # System configuration
└── main.py                     # Application entry point
```

## 🔧 Key Features Explained

### Early Interruption Detection

The system uses **partial transcripts** from Cartesia STT to detect when users start speaking, allowing TTS to stop **immediately** before the final transcript arrives. This provides near-instant interruption response.

### Real-time Text Streaming

LLM-generated text is streamed to the frontend **instantly** as tokens are generated, providing real-time visual feedback alongside audio.

### Audio Visualization

The frontend includes a beautiful audio visualizer with animated bars that respond to both microphone input and TTS output audio.

## 📝 Development

### Running Tests

```bash
# Add test commands here when tests are added
```

### Code Structure

- **Async/Await**: Fully async implementation for optimal performance
- **Event-driven**: Callback-based transcript processing
- **Type Hints**: Full type annotations for better IDE support

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

MIT License - see LICENSE file for details

## 🙏 Acknowledgments

- [Cartesia](https://cartesia.ai) for STT and TTS services
- [DeepInfra](https://deepinfra.com) for LLM hosting
- [FastAPI](https://fastapi.tiangolo.com) for the web framework

---

<div align="center">

**Built with ❤️ by inav-labs-research**

[Report Bug](https://github.com/inav-labs-research/toy_backend/issues) • [Request Feature](https://github.com/inav-labs-research/toy_backend/issues)

</div>
