import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI
from environment import JobSearchSimulator, JobAction
import uvicorn

app = FastAPI(title="Job Search Simulator")
env_instance = None

@app.get("/")
def root():
    return {"name": "Job Search Simulator", "version": "1.0.0"}

@app.post("/reset")
def reset(task_id: str = "resume_screening"):
    global env_instance
    env_instance = JobSearchSimulator(task_id=task_id)
    obs = env_instance.reset()
    return {"status": "ok", "observation": obs.dict()}

@app.post("/step")
def step(action: JobAction):
    global env_instance
    if env_instance is None:
        env_instance = JobSearchSimulator()
    result = env_instance.step(action)
    return {
        "observation": result.observation.dict(),
        "reward": result.reward.value,
        "done": result.done,
        "feedback": result.reward.feedback
    }

@app.get("/state")
def state():
    global env_instance
    if env_instance is None:
        return {"status": "not initialized"}
    return env_instance.get_state()

def main():
    uvicorn.run(app, host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()
