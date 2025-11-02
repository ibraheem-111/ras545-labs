# Inputs
* Size of the grid
* Starting or finishing 


# Idea:

Agentic: 

1. Identify the grid size of the maze.
2. Use CV 2 function to get a mask of the maze
3. Use A* or A* + LLM to path plan


# Steps for Startup:
```
celery -A midterm2.robot:app worker -Q robot1 -l info --concurrency=1 --prefetch-multiplier=1
```
```
python midterm2/main2.py
```

