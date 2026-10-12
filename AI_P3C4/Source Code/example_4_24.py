# Example 4.24 Transformers2.py (Updated for Transformers v5+)
from transformers import pipeline

# Load the text-generation pipeline with an instruction-tuned chat model
question_answerer = pipeline('text-generation', model='Qwen/Qwen2.5-0.5B-Instruct')

# Format your prompt using chat messages
messages = [
    {
        'role': 'user',
        'content': (
            'Context: We created Biox Systems Ltd company back in the year of'
            ' 2000.\n'
            'Question: What is the name of the company?'
        ),
    }
]

# Generate response
response = question_answerer(messages, max_new_tokens=50)
print(response[0]['generated_text'][-1]['content'])