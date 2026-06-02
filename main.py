from dotenv import load_dotenv
from agents.travel_assistant import execute

load_dotenv()

def main():
    print("Start first weather agent:")
    execute()


if __name__ == "__main__":
    main()
