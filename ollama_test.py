import ollama

response = ollama.generate(
    model="qwen3:4b",
    prompt="What is Python?"
)

print(response["response"])

