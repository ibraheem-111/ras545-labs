import cv2
from openai import OpenAI
from dotenv import load_dotenv
import os
import json
from media_utils import capture_image

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(
  api_key=openai_api_key
)
cap = cv2.VideoCapture(2)

tools = [
    {
        "type": "function",
        "name": "get_maze_dimensions",
        "description": "Get the dimensions of a maze given an image of the maze",
    },
]

input_list = [
    {"role": "user", "content": "What are the dimensions of a maze in the image?"},
]

def get_maze_dimensions():

    image = capture_image(cap)

    client.responses.create(
        model="gpt-5",
        tools=tools,
        input=input_list,
    )

def get_maze_grid(maze_image):

# 2. Prompt the model with tools defined
response = client.responses.create(
    model="gpt-5",
    tools=tools,
    input=input_list,
)

for item in response.items:
    if item.type == "function_call":
        if item.name == "get_maze_dimensions":
            maze_dimensions = get_maze_dimensions(json.loads(item.arguments))
            
            input_list.append({
                "type": "function_call_output",
                "call_id": item.call_id,
                "output": json.dumps({
                  "horoscope": maze_dimensions
                })
            })


print(response.output_text);


