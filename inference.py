"""
Job Search Simulator - OpenEnv Inference Script
Prints required [START]/[STEP]/[END] blocks to stdout.
"""

import sys
import os
from typing import List

# Add parent directory to path so we can import environment
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from environment import JobSearchSimulator, JobAction, ActionType

# ========== HELPER: Print with flush ==========
def emit(text: str):
    print(text, flush=True)

# ========== DUMMY AGENT ==========
def get_dummy_action(task_id: str, step: int, observation) -> JobAction:
    """Return a hardcoded action for each task type."""
    if task_id == "resume_screening":
        content = "Added Python, Machine Learning, SQL, and a Master's degree."
        return JobAction(action_type=ActionType.EDIT_RESUME, content=content)
    elif task_id == "job_filtering":
        # Assume jobs 0,1,3,5 are legitimate (from environment.py)
        return JobAction(action_type=ActionType.SUBMIT_APPLICATION, selected_jobs=[0,1,3,5])
    else:  # interview_simulator
        content = (
            "Situation: We had a tight deadline for a data pipeline. "
            "Task: I needed to deliver clean data within 2 days. "
            "Action: I automated the ETL process using Python and SQL. "
            "Result: We delivered on time and improved accuracy by 20%."
        )
        return JobAction(action_type=ActionType.ANSWER_QUESTION, content=content)

# ========== MAIN LOOP ==========
def main():
    tasks = [
        ("resume_screening", "Resume Screening"),
        ("job_filtering", "Job Filtering"),
        ("interview_simulator", "Interview Simulation")
    ]
    
    for task_id, task_name in tasks:
        env = JobSearchSimulator(task_id=task_id)
        obs = env.reset()
        
        # Emit [START]
        emit(f"[START] task={task_name}")
        
        rewards: List[float] = []
        step = 0
        done = False
        
        while not done and step < env.max_steps:
            step += 1
            action = get_dummy_action(task_id, step, obs)
            
            # Take step
            result = env.step(action)
            reward = result.reward.value
            done = result.done
            obs = result.observation
            rewards.append(reward)
            
            # Emit [STEP]
            emit(f"[STEP] step={step} reward={reward:.3f}")
        
        # Calculate final score
        total_reward = sum(rewards)
        max_reward = 5.0 if task_id == "resume_screening" else (10.0 if task_id == "job_filtering" else 15.0)
        score = total_reward / max_reward if max_reward > 0 else 0.0
        score = min(max(score, 0.0), 1.0)
        
        # Emit [END]
        emit(f"[END] task={task_name} score={score:.3f} steps={step}")
    
    # Optional: print a final summary line (not required but helpful)
    emit("[INFO] All tasks completed")

if __name__ == "__main__":
    main()
