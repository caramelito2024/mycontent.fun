import os
from dotenv import load_dotenv
from swarmzero import Agent
from swarmzero.sdk_context import SDKContext

# Load environment variables
load_dotenv()

# Ensure you have set OPENAI_API_KEY in your .env file
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("Please set OPENAI_API_KEY in your .env file")

# Create SDK Context
sdk_context = SDKContext(config_path="./swarmzero_config.toml")

class ResearchAssistant:
    def __init__(self):
        self.agent = Agent(
            name="Research Assistant",
            instruction="You are a helpful research assistant. Analyze data and generate content based on user requests.",
            sdk_context=sdk_context,
            functions=[]
        )

    async def analyze_data(self, data):
        prompt = f"Analyze the following data and provide insights:\n\n{data}"
        response = await self.agent.chat(prompt)
        return response

    async def generate_content(self, topic):
        prompt = f"Generate a brief article about the following topic:\n\n{topic}"
        response = await self.agent.chat(prompt)
        return response

async def main():
    assistant = ResearchAssistant()

    while True:
        print("\nResearch Assistant Menu:")
        print("1. Analyze Data")
        print("2. Generate Content")
        print("3. Exit")

        choice = input("Enter your choice (1-3): ")

        if choice == "1":
            data = input("Enter the data to analyze: ")
            result = await assistant.analyze_data(data)
            print("\nAnalysis Result:")
            print(result)
        elif choice == "2":
            topic = input("Enter the topic for content generation: ")
            result = await assistant.generate_content(topic)
            print("\nGenerated Content:")
            print(result)
        elif choice == "3":
            print("Thank you for using the Research Assistant. Goodbye!")
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())