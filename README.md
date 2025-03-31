# LightTxt-Predict: Ultra-Fast LLM Predictive Text API

## 🚀 Performance Highlights
- *With RTX3080*
- **Ultra-low latency**: 17.4ms average prediction time
- **Fast initialization**: Model loads in under 1 second
- **Efficient memory usage**: ~940MB model storage footprint
- **Responsive API**: Non-blocking architecture with background processing

## 📊 Benchmarks

| Metric | LightTxt-Predict Qwen2 | Standard Qwen2-0.5B | GPT-3.5 Turbo |
|--------|-------------|---------------------|---------------|
| Avg. Latency | 17.4ms | ~20ms | 50-100ms |
| Model Size | 942.3 MB | 1.17 GB | N/A (Cloud) |
| Initialization | 0.69s | 1-2s | N/A (Cloud) |
| Tokens/sec | ~57.5 | 49.94 | ~67.83 |

## ✨ Key Features

- **GPU-optimized inference** with Flash Attention 2.0 support on supported devices with >Ampere
- **Multi-level caching system** for repeated queries
- **Precomputed common phrases** for frequently used inputs (needs to be expanded upon)
- **Asynchronous processing** with background threading
- **RESTful API** with both GET and POST endpoints
- **Batch prediction support** for improved throughput
- **Automatic hardware adaptation** (CPU/GPU with optimal settings) (limited testing so far)
- **Comprehensive statistics** and basic health monitoring

## 🔧 Installation

```bash
git clone https://github.com/WindingMotor/LightTxt-Predict
cd LightTxt-Predict
pip install -r requirements.txt

# Optional: Install Flash Attention for maximum performance
pip install flash-attn --no-build-isolation
```

## 📋 Requirements

- Python 3.8+
- NVIDIA CUDA-compatible GPU recommended (but works on CPU)
- 2GB RAM minimum (4GB+ recommended)

## 🚦 Quick Start

### Start the server

```bash
python server.py --host 0.0.0.0 --port 8000
```

### Query for predictions

```python
import requests

# Simple GET request
response = requests.get("http://localhost:8000/predict?text=I would like to")
predictions = response.json()

# Output predictions
for pred in predictions["predictions"]:
    print(f"{pred['word']} ({pred['probability']:.0%})")
```

### Sample output

```
know (43%)
create (31%)
use (26%)
```

## 🔄 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/predict` | GET | Get predictions for a single text input |
| `/predict` | POST | Get predictions for single or batch inputs |
| `/health` | GET | Check server health and model status |
| `/stats` | GET | Get performance statistics |

### GET /predict

```
http://localhost:8000/predict?text=Hello world&top_k=3
```

### POST /predict

```json
{
  "text": "Once upon a",
  "top_k": 3
}
```

### Batch prediction (POST)

```json
{
  "text": ["Hello world", "Once upon a", "The weather is"],
  "top_k": 3
}
```

## 📈 Advanced Configuration

The server accepts several command-line options:

```bash
python server.py --help
```

Options:
- `--host`: Host address (default: localhost)
- `--port`: Port number (default: 8000)
- `--disable-optimizations`: Disable model optimizations for higher quality (but slower) predictions

## 🧠 Technical Details

LightPredict uses several optimization techniques:

1. **Flash Attention 2.0** when available for efficient transformer computation
2. **Multi-level caching** to avoid redundant computation
3. **Automatic precision adaptation** (FP16 on CUDA, FP32 on CPU)
4. **Optimized tokenization** with left-side padding
5. **Selective quantization** for CPU deployment
6. **Context-aware processing** that only looks at relevant text
7. **Warm-up phase** to avoid cold-start latency
8. **Auto device mapping** for optimal GPU memory utilization

## 🤝 Comparison with Other Solutions

LightPredict offers a balance of speed and quality that outperforms many alternatives:

- **vs. API-based services** (OpenAI, Claude): Lower latency, no API costs, local deployment
- **vs. Local LLMs**: 2-5x faster than similar-sized models with standard configuration
- **vs. Traditional Predictive Text**: Higher quality predictions with semantic understanding vs n-based approaches

## 📚 Example Use Cases

- Intelligent text editors
- Code completion tools
- Mobile keyboard suggestions
- Form auto-completion
- Chat applications
- CLI tools with autocomplete

## 📄 License

MIT License

## 🙏 Acknowledgements

This project uses the excellent [Qwen2-0.5B](https://huggingface.co/Qwen/Qwen2-0.5B) model developed by Alibaba Cloud.

---
