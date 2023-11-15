#Data generation step

## Describe your model -> fine-tuned GPT-3.5

The goal of this notebook is to experiment with a new way to make it very easy to build a task-specific model for your use-case.

First, use the best GPU available (go to Runtime -> change runtime type)

To create your model, just go to the first code cell, and describe the model you want to build in the prompt. Be descriptive and clear.

Select a temperature (high=creative, low=precise), and the number of training examples to generate to train the model. From there, just run all the cells.

You can change the model you want to fine-tune by changing `model_name` in the `Define Hyperparameters` cell.

Write your prompt here. Make it as descriptive as possible!

Then, choose the temperature (between 0 and 1) to use when generating data. Lower values are great for precise tasks, like writing code, whereas larger values are better for creative tasks, like writing stories.

Finally, choose how many examples you want to generate. The more you generate, a) the longer it takes and b) the more expensive data generation will be. But generally, more examples will lead to a higher-quality model. 100 is usually the minimum to start.
"""

prompt = "A model that takes in a puzzle-like reasoning-heavy question in English, and responds with a well-reasoned, step-by-step thought out response in Spanish."
temperature = .4
number_of_examples = 50

"""Run this to generate the dataset."""

!pip install openai tenacity

import os
import openai
import random
from tenacity import retry, stop_after_attempt, wait_exponential

openai.api_key = ""

N_RETRIES = 3

@retry(stop=stop_after_attempt(N_RETRIES), wait=wait_exponential(multiplier=1, min=4, max=70))
def generate_example(prompt, prev_examples, temperature=.5):
    messages=[
        {
            "role": "system",
            "content": f"You are generating data which will be used to train a machine learning model.\n\nYou will be given a high-level description of the model we want to train, and from that, you will generate data samples, each with a prompt/response pair.\n\nYou will do so in this format:\n```\nprompt\n-----------\n$prompt_goes_here\n-----------\n\nresponse\n-----------\n$response_goes_here\n-----------\n```\n\nOnly one prompt/response pair should be generated per turn.\n\nFor each turn, make the example slightly more complex than the last, while ensuring diversity.\n\nMake sure your samples are unique and diverse, yet high-quality and complex enough to train a well-performing model.\n\nHere is the type of model we want to train:\n`{prompt}`"
        }
    ]

    if len(prev_examples) > 0:
        if len(prev_examples) > 8:
            prev_examples = random.sample(prev_examples, 8)
        for example in prev_examples:
            messages.append({
                "role": "assistant",
                "content": example
            })

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=messages,
        temperature=temperature,
        max_tokens=1000,
    )

    return response.choices[0].message['content']

# Generate examples
prev_examples = []
for i in range(number_of_examples):
    print(f'Generating example {i}')
    example = generate_example(prompt, prev_examples, temperature)
    prev_examples.append(example)

print(prev_examples)

"""We also need to generate a system message."""

def generate_system_message(prompt):

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
          {
            "role": "system",
            "content": "You will be given a high-level description of the model we are training, and from that, you will generate a simple system prompt for that model to use. Remember, you are not generating the system message for data generation -- you are generating the system message to use for inference. A good format to follow is `Given $INPUT_DATA, you will $WHAT_THE_MODEL_SHOULD_DO.`.\n\nMake it as concise as possible. Include nothing but the system prompt in your response.\n\nFor example, never write: `\"$SYSTEM_PROMPT_HERE\"`.\n\nIt should be like: `$SYSTEM_PROMPT_HERE`."
          },
          {
              "role": "user",
              "content": prompt.strip(),
          }
        ],
        temperature=temperature,
        max_tokens=500,
    )

    return response.choices[0].message['content']

system_message = generate_system_message(prompt)

print(f'The system message is: `{system_message}`. Feel free to re-run this cell if you want a better result.')

"""Now let's put our examples into a dataframe and turn them into a final pair of datasets."""

import json
import pandas as pd

# Initialize lists to store prompts and responses
prompts = []
responses = []

# Parse out prompts and responses from examples
for example in prev_examples:
  try:
    split_example = example.split('-----------')
    prompts.append(split_example[1].strip())
    responses.append(split_example[3].strip())
  except:
    pass

# Create a DataFrame
df = pd.DataFrame({
    'prompt': prompts,
    'response': responses
})

# Remove duplicates
df = df.drop_duplicates()

print('There are ' + str(len(df)) + ' successfully-generated examples.')

# Initialize list to store training examples
training_examples = []

# Create training examples in the format required for GPT-3.5 fine-tuning
for index, row in df.iterrows():
    training_example = {
        "messages": [
            {"role": "system", "content": system_message.strip()},
            {"role": "user", "content": row['prompt']},
            {"role": "assistant", "content": row['response']}
        ]
    }
    training_examples.append(training_example)

# Save training examples to a .jsonl file
with open('training_examples.jsonl', 'w') as f:
    for example in training_examples:
        f.write(json.dumps(example) + '\n')

"""# Upload the file to OpenAI"""

file_id = openai.File.create(
  file=open("/content/training_examples.jsonl", "rb"),
  purpose='fine-tune'
).id

"""# Train the model! You may need to wait a few minutes before running the next cell to allow for the file to process on OpenAI's servers."""

job = openai.FineTuningJob.create(training_file=file_id, model="gpt-3.5-turbo")

job_id = job.id

"""# Now, just wait until the fine-tuning run is done, and you'll have a ready-to-use model!

Run this cell every 20 minutes or so -- eventually, you'll see a message "New fine-tuned model created: ft:gpt-3.5-turbo-0613:xxxxxxxxxxxx"

Once you see that message, you can go to the OpenAI Playground (or keep going to the next cells and use the API) to try the model!
"""

openai.FineTuningJob.list_events(id=job_id, limit=10)

"""# Once your model is trained, run the next cell to grab the fine-tuned model name."""

model_name_pre_object = openai.FineTuningJob.retrieve(job_id)
model_name = model_name_pre_object.fine_tuned_model
print(model_name)

"""# Let's try it out!"""

response = openai.ChatCompletion.create(
    model=model_name,
    messages=[
      {
        "role": "system",
        "content": system_message,
      },
      {
          "role": "user",
          "content": df['prompt'].sample().values[0],
      }
    ],
)

response.choices[0].message['content']
