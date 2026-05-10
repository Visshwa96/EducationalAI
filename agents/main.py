import ollama

from agents.science_agent import science_agent
from agents.math_agent import math_agent
from agents.cs_agent import cs_agent


def choose_agent(question):

    prompt = f"""
You are a router.

Choose the BEST subject agent.

1 = Science
2 = Math
3 = Computer Science

Examples:

Physics -> Science
Chemistry -> Science
Biology -> Science

Algebra -> Math
Calculus -> Math
Statistics -> Math

Programming -> Computer Science
Python -> Computer Science
Algorithms -> Computer Science

Question:
{question}

Reply ONLY with:
1
2
3
"""

    response = ollama.chat(
        model='qwen2.5:7b',
        messages=[
            {
                'role': 'user',
                'content': prompt
            }
        ]
    )

    return response['message']['content'].strip()


print("Initializing Agents...\n")

science_agent.initialize()
math_agent.initialize()
cs_agent.initialize()

print("\nAll Agents Ready!\n")


while True:

    question = input("\nAsk a question: ")

    agent = choose_agent(question)

    if agent == "1":

        print("\nScience Agent:\n")

        print(science_agent.ask(question))

    elif agent == "2":

        print("\nMath Agent:\n")

        print(math_agent.ask(question))

    elif agent == "3":

        print("\nComputer Science Agent:\n")

        print(cs_agent.ask(question))

    else:

        print("Could not determine correct agent.")