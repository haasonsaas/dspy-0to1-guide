#!/usr/bin/env python3
"""
Mathematical Question Answering with DSPy

Demonstrates:
- Mathematical reasoning with ChainOfThought
- Structured output with type hints
- Error handling and validation
- Multiple problem types
"""

import os
import re
from typing import Union

import dspy
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class MathQA(dspy.Module):
    """Mathematical reasoning module with step-by-step solutions."""

    def __init__(self):
        super().__init__()
        # Use ChainOfThought for step-by-step reasoning
        self.solve = dspy.ChainOfThought(
            "math_problem -> solution: float, explanation: str"
        )

    def forward(self, math_problem: str):
        """Solve a mathematical problem with reasoning."""
        response = self.solve(math_problem=math_problem)

        # Try to extract numeric answer if not properly formatted
        if hasattr(response, "solution"):
            try:
                # Extract number from solution if it's a string
                if isinstance(response.solution, str):
                    numbers = re.findall(r"-?\d+\.?\d*", response.solution)
                    if numbers:
                        response.solution = float(numbers[-1])  # Take the last number
            except (ValueError, AttributeError):
                response.solution = "Could not determine numeric answer"

        return response


def validate_solution(
    problem: str, expected: Union[float, str], actual_response
) -> bool:
    """Validate if the solution is approximately correct."""
    try:
        if isinstance(expected, str):
            return True  # Skip validation for non-numeric expected answers

        actual = (
            float(actual_response.solution)
            if hasattr(actual_response, "solution")
            else None
        )
        if actual is None:
            return False

        # Allow for small floating point differences
        return abs(actual - expected) < 0.01
    except:
        return False


def main():
    """Run mathematical reasoning examples."""
    print("🧮 DSPy Mathematical Question Answering")
    print("=" * 50)

    # Configure model
    try:
        if os.getenv("OPENAI_API_KEY"):
            dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))
            print("✅ Using OpenAI GPT-4o-mini")
        else:
            dspy.configure(lm=dspy.LM("ollama_chat/llama3"))
            print("✅ Using Ollama Llama3")
    except Exception as e:
        print(f"❌ Model configuration failed: {e}")
        return

    # Create math QA module
    math_qa = MathQA()

    # Test problems with expected answers
    test_problems = [
        {"problem": "What is 15% of 240?", "expected": 36.0, "category": "Percentage"},
        {
            "problem": "If a rectangle has length 12 and width 8, what is its area?",
            "expected": 96.0,
            "category": "Geometry",
        },
        {"problem": "Solve for x: 2x + 5 = 17", "expected": 6.0, "category": "Algebra"},
        {
            "problem": "What is the compound interest on $1000 at 5% per year for 2 years?",
            "expected": 102.5,  # 1000 * (1.05)^2 - 1000 = 102.5
            "category": "Finance",
        },
        {
            "problem": "A train travels 240 miles in 3 hours. What is its average speed in mph?",
            "expected": 80.0,
            "category": "Rate/Speed",
        },
        {
            "problem": "What is the sum of the first 10 positive integers?",
            "expected": 55.0,
            "category": "Series",
        },
    ]

    print("\n📊 Testing Mathematical Problems:")
    print("-" * 40)

    correct_answers = 0
    total_problems = len(test_problems)

    for i, test_case in enumerate(test_problems, 1):
        problem = test_case["problem"]
        expected = test_case["expected"]
        category = test_case["category"]

        print(f"\n{i}. [{category}] {problem}")

        try:
            response = math_qa(problem)

            # Display results
            print(f"   💡 Solution: {response.solution}")

            if hasattr(response, "explanation"):
                print(f"   📝 Explanation: {response.explanation[:100]}...")
            elif hasattr(response, "rationale"):
                print(f"   📝 Reasoning: {response.rationale[:100]}...")

            # Validate answer
            is_correct = validate_solution(problem, expected, response)
            if is_correct:
                print(f"   ✅ Correct! (Expected: {expected})")
                correct_answers += 1
            else:
                print(f"   ❌ Expected: {expected}, Got: {response.solution}")

        except Exception as e:
            print(f"   💥 Error: {e}")

    # Summary
    print(f"\n📈 Results Summary:")
    print(
        f"   Correct: {correct_answers}/{total_problems} ({correct_answers/total_problems*100:.1f}%)"
    )

    if correct_answers == total_problems:
        print("   🎉 Perfect score! All problems solved correctly.")
    elif correct_answers >= total_problems * 0.8:
        print("   👍 Great performance! Most problems solved correctly.")
    else:
        print("   📚 Some issues detected. Consider:")
        print("      - Using a more capable model (GPT-4 vs GPT-3.5)")
        print("      - Adding few-shot examples")
        print("      - Using DSPy optimization (BootstrapFewShot)")

    print("\n🔧 Advanced Usage:")
    print("   Try optimizing this module with:")
    print("   python examples/advanced/math_optimization.py")


if __name__ == "__main__":
    main()
