# Naive approach using a foundation model with no RAG or fine tuning
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam

openai_api_client = OpenAI()

def generation(prompt_messages: list[ChatCompletionMessageParam]) -> str:
    response = openai_api_client.chat.completions.create(
        messages=prompt_messages,
        model='gpt-4o-mini',
        temperature=0,
    )

    return response.choices[0].message.content

def main():
    # Example prompt messages
    example_prompt = [
        {
            "role": "system",
            "content": "You are a therapist specialized the cognitive behavioral therapy."
        },
        {
            "role": "user",
            "content": "It's been years since I have had nightmares about my father. Whenever I talk to him or even just think about him, I get a knod in my stomach and gets anxious. But if I stop talking him, I also feels off and not grounded. What should I do? "
        }
    ]
    try:
        response = generation(example_prompt)
        print("Assistant's response:")
        print(response)
    except Exception as e:
        print(f"Error occurred: {e}")

if __name__=='__main__':
    main()