from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, List, Optional
from enum import Enum
import uvicorn

def main():
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860, reload=False)

if __name__ == "__main__":
    main()
# ============== FastAPI App ==============
app = FastAPI(title="Job Search Simulator")

# ============== Environment Classes ==============

class ActionType(str, Enum):
    EDIT_RESUME = "edit_resume"
    SUBMIT_APPLICATION = "submit_application"
    ANSWER_QUESTION = "answer_question"

class JobAction(BaseModel):
    action_type: ActionType
    content: Optional[str] = None
    selected_jobs: Optional[List[int]] = None

class JobObservation(BaseModel):
    current_task: str
    job_description: Optional[str] = None
    resume_content: Optional[str] = None
    available_jobs: Optional[List[Dict]] = None
    interview_question: Optional[str] = None
    question_number: Optional[int] = None
    total_questions: Optional[int] = None
    feedback: Optional[str] = None
    progress: float = 0.0

class JobReward(BaseModel):
    value: float
    breakdown: Dict[str, float] = {}
    feedback: str = ""

class StepResult(BaseModel):
    observation: JobObservation
    reward: JobReward
    done: bool
    info: Dict = {}

# ============== The Environment ==============

class JobSearchSimulator:
    def __init__(self, task_id: str = "resume_screening"):
        self.task_id = task_id
        self.step_count = 0
        self.max_steps = 20
        self.done = False
        self.total_reward = 0.0
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
                {"id": 0, "title": "Software Engineer", "company": "Google", "is_scam": False},
                {"id": 1, "title": "Data Analyst", "company": "Microsoft", "is_scam": False},
                {"id": 2, "title": "Work From Home", "company": "Scam Co", "is_scam": True},
                {"id": 3, "title": "ML Engineer", "company": "OpenAI", "is_scam": False},
                {"id": 4, "title": "Get Rich Quick", "company": "Fake Corp", "is_scam": True},
                {"id": 5, "title": "Frontend Dev", "company": "Stripe", "is_scam": False},
            ]
        elif self.task_id == "interview_simulator":
            self.questions = [
                "Tell me about a challenge you overcame at work",
                "Describe a time you worked effectively in a team",
                "How do you handle multiple deadlines?",
                "Tell me about a time you showed leadership"
            ]
            self.current_question = self.questions[0] if self.questions else ""
            self.total_questions = len(self.questions)
    
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
            score = 0.0
            if action.content:
                content_lower = action.content.lower()
                if "python" in content_lower:
                    score += 0.3
                if "machine learning" in content_lower:
                    score += 0.3
                if "sql" in content_lower:
                    score += 0.2
                if "master" in content_lower:
                    score += 0.2
            reward_value = min(1.0, score)
            feedback = f"Resume score: {reward_value:.2f}"
            self.done = True
            
        elif self.task_id == "job_filtering" and action.action_type == ActionType.SUBMIT_APPLICATION:
            score = 0.0
            good_count = 0
            scam_count = 0
            if action.selected_jobs:
                for job_id in action.selected_jobs:
                    if job_id < len(self.available_jobs):
                        if not self.available_jobs[job_id]["is_scam"]:
                            good_count += 1
                            score += 0.25
                        else:
                            scam_count += 1
                            score -= 0.3
                if scam_count == 0 and good_count >= 3:
                    score += 0.2
            reward_value = max(0.0, min(1.0, score))
            feedback = f"Selected {good_count} good jobs, {scam_count} scams"
            self.done = True
            
        elif self.task_id == "interview_simulator" and action.action_type == ActionType.ANSWER_QUESTION:
            score = 0.0
            if action.content:
                content_lower = action.content.lower()
                word_count = len(action.content.split())
                if 30 <= word_count <= 200:
                    score += 0.25
                star_keywords = ["situation", "task", "action", "result"]
                star_count = sum(1 for kw in star_keywords if kw in content_lower)
                score += min(0.3, star_count * 0.1)
                if any(char.isdigit() for char in action.content):
                    score += 0.15
            reward_value = min(0.8, score)
            feedback = f"Answer score: {reward_value:.2f}"
            self.answers.append(reward_value)
            self.question_index += 1
            
            if self.question_index >= len(self.questions):
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
    
    def get_state(self):
        return {
            "task_id": self.task_id,
            "step_count": self.step_count,
            "done": self.done,
            "total_reward": self.total_reward,
            "progress": self.step_count / self.max_steps
        }
    
    def _get_observation(self):
        if self.task_id == "resume_screening":
            return JobObservation(
                current_task=self.task_id,
                job_description=self.job_description,
                resume_content=self.resume_content,
                feedback="Edit the resume to match the job description. Add Python, ML, SQL.",
                progress=self.step_count / self.max_steps
            )
        elif self.task_id == "job_filtering":
            return JobObservation(
                current_task=self.task_id,
                available_jobs=self.available_jobs,
                feedback="Select job indices that are NOT scams. Return as list like [0,1,3]",
                progress=self.step_count / self.max_steps
            )
        else:
            return JobObservation(
                current_task=self.task_id,
                interview_question=self.current_question,
                question_number=self.question_index + 1,
                total_questions=len(self.questions),
                feedback=f"Question {self.question_index + 1} of {len(self.questions)}. Use STAR method.",
                progress=self.step_count / self.max_steps
            )

# ============== Global instance ==============
env_instance = JobSearchSimulator()

# ============== API Endpoints ==============

@app.get("/")
async def root():
    return {
        "name": "Job Search Simulator",
        "version": "1.0.0",
        "tasks": ["resume_screening", "job_filtering", "interview_simulator"],
        "endpoints": {
            "GET /": "This info",
            "GET /reset": "Reset environment",
            "POST /step": "Execute action",
            "GET /state": "Get current state"
        }
    }

@app.get("/")
def root():
    return {"name": "Job Search Simulator"}

@app.post("/reset")      # Changed from GET to POST
def reset():
    global env_instance
    env_instance = JobSearchSimulator()
    obs = env_instance.reset()
    return {"status": "ok", "observation": obs.dict()}

@app.post("/step")       # POST is correct
def step(action: JobAction):
    global env_instance
    result = env_instance.step(action)
    return {
        "observation": result.observation.dict(),
        "reward": result.reward.value,
        "done": result.done
    }

@app.get("/state")       # GET is correct
def state():
    global env_instance
    return env_instance.get_state()
