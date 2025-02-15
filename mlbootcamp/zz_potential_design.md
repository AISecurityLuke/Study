# Chatbot Filtration System - Pseudocode

## 1. Receive Prompt via API

### API Endpoint: `/receive_prompt`

- Accepts user input (`prompt`)
- Forwards `prompt` to the classification pipeline

## 2. Classification Pipeline

### Step 1: Initialize Classification Model & Configurable Thresholds

- Define `classification_model = Model()`
- Model returns a risk score between 0 and 1
- Define **configurable thresholds**:
  - `GREEN_THRESHOLD = 0.3`  # Safe prompts
  - `YELLOW_THRESHOLD = 0.7`  # Suspicious prompts
  - `RED_THRESHOLD = 0.9`  # Malicious prompts

### Step 2: Pass Prompt to Classification Model

- `score = classification_model.predict(prompt)`

### Step 3: Filter Decision (Green, Yellow, Red Classification)

- **Green Zone (****`score < GREEN_THRESHOLD`****)** → **PASS** → Directly forward to RAG model
- **Yellow Zone (****`GREEN_THRESHOLD ≤ score < YELLOW_THRESHOLD`****)** → **FLAG** → Send to RAG model, but log for security review
- **Red Zone (****`score ≥ RED_THRESHOLD`****)** → **REJECT** → Log the failed prompt asynchronously and return response (`"Unsafe request detected. This event will be analyzed by security."`)

## 3. Asynchronous Logging & SIEM Integration

### Step 1: Log Entry Structure

- Log entry format:
  - Timestamp
  - User ID (if available)
  - Prompt text
  - Model score
  - Classification level (`Green`, `Yellow`, `Red`)
  - Reason for flagging/rejection

### Step 2: Implement Asynchronous Logging

- **Use a message queue (e.g., Kafka, RabbitMQ, Redis)** to handle high traffic
- **Producer:**
  - Push flagged/rejected prompt logs to queue
- **Consumer:**
  - Process logs from the queue asynchronously
  - Send logs to:
    - Syslog for SIEM ingestion
    - Cloud logging service
    - Direct API endpoint for real-time alerts

### Step 3: Auto-Scaling Logging Services

- Use **load balancers and distributed log aggregators** (Fluentd, Logstash [what I'm leaning towards], Kafka)
- Prioritize **real-time alerts** for high-risk (`Red`) prompts
- Implement **batch processing** for lower-severity (`Yellow`) logs to optimize system performance

## 4. Forward Safe & Flagged Prompts to RAG Model

- If `score` is in the `Green` or `Yellow` category:
  - Forward `prompt` to `RAG_model.generate_response(prompt)`
  - Return the generated response to the user

## 5. Example Workflow

1. **User sends prompt**: "How to hack a database?"
2. **Classification model returns a score**:
   - Model: `0.8`
3. **Decision**:
   - If `score < 0.3` → **GREEN (Safe)** → Forward to RAG model
   - If `0.3 ≤ score < 0.7` → **YELLOW (Suspicious)** → Forward to RAG model but log for security
   - If `score ≥ 0.9` → **RED (Malicious)** → REJECT (Push log to async queue → SIEM, Syslog, Cloud logs)
4. **Response to user**:
   - If **REJECTED**: "Unsafe request detected."
   - If **PASSED**: "Here’s some safe and ethical guidance."

