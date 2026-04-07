---
title: Job Search Simulator
emoji: 💼
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
---

# Job Search Simulator

OpenEnv environment for AI agents to learn job hunting skills.

## Tasks
- **Easy**: Resume optimization (10 steps)
- **Medium**: Job scam filtering (25 steps)  
- **Hard**: Behavioral interview (15 steps)

## API
- `GET /reset` - Reset environment
- `POST /step` - Execute action
- `GET /state` - Get current state

## Environment Variables
- `OPENAI_API_KEY` - Required for LLM inference


for example 

EASY 🟢        MEDIUM 🟡        HARD 🔴
─────────      ─────────        ─────────
Resume         Job Scam          Interview
Screening      Detection         Simulation

AI learns to:   AI learns to:     AI learns to:
• Add keywords  • Spot scams      • Use STAR method
• Match jobs    • Pick legit jobs • Answer professionally
• Get noticed   • Avoid fraud     • Get hired!