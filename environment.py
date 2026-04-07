from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import random

# Add these async wrapper methods to your JobSearchSimulator class:

async def reset_async(self):
    """Async wrapper for reset"""
    return self.reset()

async def step_async(self, action):
    """Async wrapper for step"""
    return self.step(action)

# Simple OpenEnv implementation without complex imports
class Observation:
    pass

class Action:
    pass

class Reward:
    pass

class StepResult:
    def __init__(self, observation, reward, done, info=None):
        self.observation = observation
        self.reward = reward
        self.done = done
        self.info = info or {}

class Env:
    pass

class ActionType(str, Enum):
    EDIT_RESUME = "edit_resume"
    SUBMIT_APPLICATION = "submit_application"
    ANSWER_QUESTION = "answer_question"

class JobAction(Action):
    def __init__(self, action_type: ActionType, content: str = None, selected_jobs: List[int] = None):
        self.action_type = action_type
        self.content = content
        self.selected_jobs = selected_jobs

class JobObservation(Observation):
    def __init__(self, current_task: str, job_description: str = None, resume_content: str = None,
                 available_jobs: List[Dict] = None, interview_question: str = None,
                 question_number: int = None, feedback: str = None, progress: float = 0.0):
        self.current_task = current_task
        self.job_description = job_description
        self.resume_content = resume_content
        self.available_jobs = available_jobs
        self.interview_question = interview_question
        self.question_number = question_number
        self.feedback = feedback
        self.progress = progress

class JobReward(Reward):
    def __init__(self, value: float, breakdown: Dict = None, feedback: str = ""):
        self.value = value
        self.breakdown = breakdown or {}
        self.feedback = feedback

class JobSearchSimulator(Env):
    def __init__(self, task_id: str = "resume_screening"):
        self.task_id = task_id
        self.step_count = 0
        self.max_steps = 20
        self.done = False
        self.total_reward = 0.0
        
        # Task specific data
        self.resume_content = ""
        self.job_description = ""
        self.available_jobs = []
        self.current_question = ""
        self.question_index = 0
        self.questions = []
        self.answers = []
        
        self._init_task()
    
    def _init_task(self):
        if self.task_id == "resume_screening":
            self.job_description = "Senior Data Scientist: Need Python, Machine Learning, SQL, 5+ years experience"
            self.resume_content = "BS in Math, 1 year JavaScript experience"
        elif self.task_id == "job_filtering":
            self.available_jobs = [
                {"id": 0, "title": "Software Engineer", "company": "Google", "is_scam": False, "salary": "$150k"},
                {"id": 1, "title": "Data Analyst", "company": "Microsoft", "is_scam": False, "salary": "$90k"},
                {"id": 2, "title": "Work From Home", "company": "Scam Co", "is_scam": True, "salary": "$5000/week"},
                {"id": 3, "title": "ML Engineer", "company": "OpenAI", "is_scam": False, "salary": "$200k"},
                {"id": 4, "title": "Get Rich Quick", "company": "Fake Corp", "is_scam": True, "salary": "Unlimited"},
                {"id": 5, "title": "Frontend Dev", "company": "Stripe", "is_scam": False, "salary": "$130k"},
            ]
        elif self.task_id == "interview_simulator":
            self.questions = [
                "Tell me about a challenge you overcame at work",
                "Describe a time you worked effectively in a team",
                "How do you handle multiple deadlines?",
                "Tell me about a time you showed leadership"
            ]
            self.current_question = self.questions[0] if self.questions else ""
    
    def reset(self):
        self.step_count = 0
        self.done = False
        self.total_reward = 0.0
        self.answers = []
        self.question_index = 0
        self._init_task()
        return self._get_observation()
    
    def step(self, action: JobAction):
        if self.done:
            return StepResult(
                observation=self._get_observation(),
                reward=JobReward(value=0.0),
                done=True
            )
        
        reward_value = 0.0
        feedback = ""
        
        if self.task_id == "resume_screening" and action.action_type == ActionType.EDIT_RESUME:
            # Score the resume
            score = 0.0
            if action.content:
                content_lower = action.content.lower()
                if "python" in content_lower:
                    score += 0.3
                if "machine learning" in content_lower or "ml" in content_lower:
                    score += 0.3
                if "sql" in content_lower:
                    score += 0.2
                if "master" in content_lower or "m.s." in content_lower or "ms" in content_lower:
                    score += 0.2
                if "tensorflow" in content_lower or "pytorch" in content_lower:
                    score += 0.1
                    
            reward_value = min(1.0, score)
            feedback = f"Resume score: {reward_value:.2f}"
            self.done = True
            
        elif self.task_id == "job_filtering" and action.action_type == ActionType.SUBMIT_APPLICATION:
            # Score job selection
            score = 0.0
            if action.selected_jobs:
                good_count = 0
                scam_count = 0
                for job_id in action.selected_jobs:
                    if job_id < len(self.available_jobs):
                        if not self.available_jobs[job_id]["is_scam"]:
                            good_count += 1
                            score += 0.25
                        else:
                            scam_count += 1
                            score -= 0.3
                
                # Bonus for no scams
                if scam_count == 0 and good_count >= 3:
                    score += 0.2
                    
            reward_value = max(0.0, min(1.0, score))
            feedback = f"Selected {good_count} good jobs, {scam_count} scams. Score: {reward_value:.2f}"
            self.done = True
            
        elif self.task_id == "interview_simulator" and action.action_type == ActionType.ANSWER_QUESTION:
            # Score interview answer
            score = 0.0
            if action.content:
                content_lower = action.content.lower()
                word_count = len(action.content.split())
                
                # Length score
                if 30 <= word_count <= 200:
                    score += 0.25
                elif word_count > 200:
                    score += 0.15
                elif word_count < 15:
                    score -= 0.1
                
                # STAR method keywords
                star_keywords = ["situation", "task", "action", "result", "when i", "i did", "outcome", "learned"]
                star_count = sum(1 for kw in star_keywords if kw in content_lower)
                score += min(0.3, star_count * 0.1)
                
                # Specific details (numbers, metrics)
                if any(char.isdigit() for char in action.content):
                    score += 0.15
                
                # Professional language
                professional_words = ["achieved", "improved", "developed", "implemented", "led", "created"]
                if any(word in content_lower for word in professional_words):
                    score += 0.1
                    
            reward_value = min(0.8, score)
            feedback = f"Answer score: {reward_value:.2f}"
            self.answers.append(reward_value)
            self.question_index += 1
            
            if self.question_index >= len(self.questions):
                # Interview complete - calculate average
                avg_score = sum(self.answers) / len(self.answers) if self.answers else 0
                reward_value = avg_score
                feedback = f"Interview complete! Final score: {avg_score:.2f}"
                self.done = True
            else:
                self.current_question = self.questions[self.question_index]
        
        self.step_count += 1
        self.total_reward += reward_value
        
        if self.step_count >= self.max_steps:
            self.done = True
        
        return StepResult(
            observation=self._get_observation(),
            reward=JobReward(value=reward_value, feedback=feedback),
            done=self.done
        )
    
    def _get_observation(self):
        if self.task_id == "resume_screening":
            obs = JobObservation(
                current_task=self.task_id,
                job_description=self.job_description,
                resume_content=self.resume_content,
                feedback="Edit the resume to better match the job description. Add Python, Machine Learning, SQL, and a Master's degree.",
                progress=self.step_count / self.max_steps
            )
        elif self.task_id == "job_filtering":
            obs = JobObservation(
                current_task=self.task_id,
                available_jobs=self.available_jobs,
                feedback="Select the indices (0-5) of legitimate jobs to apply to. Avoid scams! Return as JSON list like [0,1,3]",
                progress=self.step_count / self.max_steps
            )
        else:
            obs = JobObservation(
                current_task=self.task_id,
                interview_question=self.current_question,
                question_number=self.question_index + 1 if self.question_index < len(self.questions) else len(self.questions),
                feedback=f"Question {self.question_index + 1} of {len(self.questions)}. Use STAR method: Situation, Task, Action, Result",
                progress=self.step_count / self.max_steps
            )
        return obs