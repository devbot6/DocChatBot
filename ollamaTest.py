import ollama

ollama.pull("mistral")

try:
    stream = ollama.chat(
        model='mistral', 
        messages=[{'role': 'user', 'content': 'Who is the richest person in history?'}], 
        stream=True
    )

    for chunk in stream:
        print(chunk['message']['content'], end='', flush=True)

except Exception as e:
    print(f"Error: {e}")
