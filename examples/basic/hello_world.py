#!/usr/bin/env python3
"""
DSPy Hello World Example

This is the simplest possible DSPy program demonstrating:
- Basic module creation
- ChainOfThought reasoning
- Model configuration
"""

import os

import dspy
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def main():
    """Run the Hello World example."""
    print("🌍 DSPy Hello World Example")
    print("=" * 50)

    # Configure DSPy with OpenAI (or fallback to a local model)
    try:
        if os.getenv("OPENAI_API_KEY"):
            dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))
            print("✅ Using OpenAI GPT-4o-mini")
        else:
            # Fallback to local model if available
            dspy.configure(lm=dspy.LM("ollama_chat/llama3"))
            print("✅ Using Ollama Llama3")
    except Exception as e:
        print(f"❌ Model configuration failed: {e}")
        return

    # Define a simple QA module
    class HelloQA(dspy.Module):
        """Simple question-answering module."""

        def __init__(self):
            super().__init__()
            self.generate_answer = dspy.ChainOfThought("question -> answer")

        def forward(self, question: str):
            return self.generate_answer(question=question)

    # Create and test the module
    qa = HelloQA()

    # Test questions
    questions = [
        "What is the capital of France?",
        "Explain what machine learning is in simple terms.",
        "What is 15 + 27?",
        "Who wrote Romeo and Juliet?",
    ]

    print("\n📝 Testing Questions:")
    print("-" * 30)

    for i, question in enumerate(questions, 1):
        try:
            print(f"\n{i}. Question: {question}")
            response = qa(question)
            print(f"   Answer: {response.answer}")

            # Show reasoning if available
            if hasattr(response, "rationale") and response.rationale:
                print(f"   Reasoning: {response.rationale}")

        except Exception as e:
            print(f"   ❌ Error: {e}")

    print("\n🎉 Hello World example completed!")
    print("\nNext steps:")
    print("- Try running: python examples/basic/math_qa.py")
    print("- Explore: python examples/basic/summarizer.py")
    print("- Advanced: python examples/personas/support_sam.py")


if __name__ == "__main__":
    main()
