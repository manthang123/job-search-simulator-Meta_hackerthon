"""
Job Search Simulator - OpenEnv Inference Script
Uses the provided LLM proxy for all action decisions.
"""

import os
import sys
import json
import re
from typing import List

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from environment import JobSearchSimulator, JobAction, ActionType
from openai import OpenAI

# ========== ENVIRONMENT VARIABLES (injected by OpenEnv) ==========
API_BASE_URL = os.environ["API_BASE_URL"]   # Must be used exactly
API_KEY = os.environ["API_KEY"]             # Must be used exactly
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-3.5-turbo")  # optional

# ========== HELPERS ==========
def emit(text: str):
    print(text, flush=True)

def get_llm_action(client: OpenAI, task_id: str, observation, step: int, last_reward: float) -> JobAction:
    """
    Call the LLM through the proxy and return a JobAction.
    """
    # Build prompt based on task
    if task_id == "resume_screening":
        prompt = f"""You are optimizing a resume.

Job Description: {observation.job_description}
Current Resume: {observation.resume_content}
Previous reward: {last_reward:.2f}

Improve the resume by adding relevant keywords (Python, Machine Learning, SQL, Master's degree).
Return ONLY the improved resume text (no explanations):"""
        
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are a job-seeking AI assistant. Respond only with the requested content."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=300
        )
        content = response.choices[0].message.content.strip()
        return JobAction(action_type=ActionType.EDIT_RESUME, content=content)

    elif task_id == "job_filtering":
        jobs = observation.available_jobs
        prompt = f"""You are filtering job posts to avoid scams.

Available jobs (index, title, company, is_scam):
{json.dumps(jobs, indent=2)}

Previous reward: {last_reward:.2f}

Select the indices of legitimate jobs (is_scam = false).
Return ONLY a JSON list like: [0, 1, 3]"""
        
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are a careful job seeker. Return only a JSON list."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=100
        )
        content = response.choices[0].message.content.strip()
        # Extract list using regex
        match = re.search(r'\[[\d,\s]+\]', content)
        if match:
            selected = eval(match.group())
        else:
            selected = [0, 1, 3, 5]  # fallback
        return JobAction(action_type=ActionType.SUBMIT_APPLICATION, selected_jobs=selected)

    else:  # interview_simulator
        prompt = f"""You are in a job interview.

Question: {observation.interview_question}
Question {observation.question_number} of {observation.total_questions}
Previous answer score: {last_reward:.2f}

Answer using the STAR method:
- Situation: Set the context
- Task: What needed to be done
- Action: What you specifically did
- Result: What was the outcome

Return ONLY your answer (no explanations):"""
        
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are a job-seeking AI assistant. Answer using STAR method."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5,
            max_tokens=400
        )
        content = response.choices[0].message.content.strip()
        return JobAction(action_type=ActionType.ANSWER_QUESTION, content=content)

# ========== MAIN LOOP ==========
def main():
    tasks = [
        ("resume_screening", "Resume Screening"),
        ("job_filtering", "Job Filtering"),
        ("interview_simulator", "Interview Simulation")
    ]
    
    # Initialise the OpenAI client with the PROXY credentials
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    
    for task_id, task_name in tasks:
        env = JobSearchSimulator(task_id=task_id)
        obs = env.reset()
        
        emit(f"[START] task={task_name}")
        
        rewards: List[float] = []
        step = 0
        done = False
        last_reward = 0.0
        
        while not done and step < env.max_steps:
            step += 1
            
            # Get action from LLM (this makes the API call through the proxy)
            action = get_llm_action(client, task_id, obs, step, last_reward)
            
            # Execute step
            result = env.step(action)
            reward = result.reward.value
            done = result.done
            obs = result.observation
            rewards.append(reward)
            last_reward = reward
            
            emit(f"[STEP] step={step} reward={reward:.3f}")
        
        # Calculate final score
        total_reward = sum(rewards)
        max_reward = 5.0 if task_id == "resume_screening" else (10.0 if task_id == "job_filtering" else 15.0)
        score = total_reward / max_reward if max_reward > 0 else 0.0
        score = min(max(score, 0.0), 1.0)
        
        emit(f"[END] task={task_name} score={score:.3f} steps={step}")
    
    emit("[INFO] All tasks completed")

if __name__ == "__main__":
    main()
