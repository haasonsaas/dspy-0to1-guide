#!/usr/bin/env python3
"""
Advanced DSPy: GEPA (Generative Prompt Auto-optimization)

Demonstrates:
- Self-reflective prompt improvement
- Iterative optimization with GEPA
- Performance tracking across iterations
- Automatic prompt evolution
- Meta-learning capabilities
"""

import os
from typing import List

import dspy
from dotenv import load_dotenv
from dspy import Evaluate, Example, metrics
from dspy.teleprompt import GEPA

# Load environment variables
load_dotenv()


class MathWordProblemSolver(dspy.Module):
    """Math word problem solver that can be optimized with GEPA."""

    def __init__(self):
        super().__init__()
        self.solve = dspy.ChainOfThought("problem -> solution: str, answer: float")

    def forward(self, problem: str):
        """Solve a math word problem."""
        result = self.solve(problem=problem)
        return result


class TextClassifier(dspy.Module):
    """Text classifier for sentiment analysis that can be optimized."""

    def __init__(self):
        super().__init__()
        self.classify = dspy.ChainOfThought("text -> sentiment: str, confidence: float")

    def forward(self, text: str):
        """Classify text sentiment."""
        result = self.classify(text=text)
        return result


class QAModule(dspy.Module):
    """Question answering module for optimization."""

    def __init__(self):
        super().__init__()
        self.answer = dspy.ChainOfThought("context, question -> answer: str")

    def forward(self, context: str, question: str):
        """Answer question based on context."""
        result = self.answer(context=context, question=question)
        return result


def create_math_dataset() -> List[Example]:
    """Create a dataset of math word problems for training."""
    problems = [
        {
            "problem": (
                "Sarah has 24 apples. She gives 8 apples to her friend and then "
                "buys 15 more. How many apples does she have now?"
            ),
            "answer": 31.0,
        },
        {
            "problem": (
                "A store sells shirts for $25 each. If John buys 3 shirts and "
                "pays with a $100 bill, how much change should he receive?"
            ),
            "answer": 25.0,
        },
        {
            "problem": (
                "Lisa runs 2.5 miles every day for 6 days. How many total miles "
                "does she run?"
            ),
            "answer": 15.0,
        },
        {
            "problem": (
                "A pizza is cut into 8 slices. If 3 people each eat 2 slices, "
                "how many slices are left?"
            ),
            "answer": 2.0,
        },
        {
            "problem": (
                "Tom saves $50 per month for 8 months. How much money has he "
                "saved in total?"
            ),
            "answer": 400.0,
        },
        {
            "problem": (
                "A rectangular garden is 12 feet long and 8 feet wide. What is "
                "its area in square feet?"
            ),
            "answer": 96.0,
        },
        {
            "problem": (
                "Maria has $120. She spends 1/3 of her money on books and 1/4 on "
                "food. How much money does she have left?"
            ),
            "answer": 50.0,
        },
        {
            "problem": (
                "A train travels 60 miles per hour. How far will it travel in "
                "2.5 hours?"
            ),
            "answer": 150.0,
        },
    ]

    examples = []
    for item in problems:
        example = Example(problem=item["problem"], answer=item["answer"]).with_inputs(
            "problem"
        )
        examples.append(example)

    return examples


def create_sentiment_dataset() -> List[Example]:
    """Create a sentiment analysis dataset."""
    texts = [
        {
            "text": "I love this product! It's amazing and works perfectly.",
            "sentiment": "positive",
        },
        {
            "text": (
                "This is the worst purchase I've ever made. Complete waste of " "money."
            ),
            "sentiment": "negative",
        },
        {
            "text": "The item is okay, nothing special but does the job.",
            "sentiment": "neutral",
        },
        {
            "text": (
                "Outstanding quality and excellent customer service! Highly "
                "recommend."
            ),
            "sentiment": "positive",
        },
        {
            "text": "Terrible experience. Poor quality and rude staff.",
            "sentiment": "negative",
        },
        {
            "text": "Average product with standard features. Price is reasonable.",
            "sentiment": "neutral",
        },
        {
            "text": "Absolutely fantastic! Exceeded all my expectations.",
            "sentiment": "positive",
        },
        {
            "text": "Disappointing quality for the price. Would not buy again.",
            "sentiment": "negative",
        },
    ]

    examples = []
    for item in texts:
        example = Example(text=item["text"], sentiment=item["sentiment"]).with_inputs(
            "text"
        )
        examples.append(example)

    return examples


def create_qa_dataset() -> List[Example]:
    """Create a question answering dataset."""
    qa_pairs = [
        {
            "context": (
                "Python is a high-level programming language known for its "
                "simplicity and readability. It was created by Guido van Rossum "
                "and first released in 1991. Python supports multiple programming "
                "paradigms including procedural, object-oriented, and functional "
                "programming."
            ),
            "question": "Who created Python?",
            "answer": "Guido van Rossum",
        },
        {
            "context": (
                "Machine learning is a subset of artificial intelligence that "
                "enables computers to learn and make decisions from data without "
                "explicit programming. Common types include supervised learning, "
                "unsupervised learning, and reinforcement learning."
            ),
            "question": "What is machine learning?",
            "answer": (
                "Machine learning is a subset of artificial intelligence that "
                "enables computers to learn and make decisions from data without "
                "explicit programming"
            ),
        },
        {
            "context": (
                "The solar system consists of the Sun and everything that orbits "
                "around it, including eight planets, their moons, asteroids, "
                "comets, and other celestial bodies. Earth is the third planet "
                "from the Sun."
            ),
            "question": "How many planets are in the solar system?",
            "answer": "Eight planets",
        },
    ]

    examples = []
    for item in qa_pairs:
        example = Example(
            context=item["context"],
            question=item["question"],
            answer=item["answer"],
        ).with_inputs("context", "question")
        examples.append(example)

    return examples


def math_accuracy_metric(prediction, ground_truth):
    """Custom metric for math problems that allows small numerical differences."""
    try:
        # Extract numeric answer from prediction
        pred_answer = float(prediction.answer) if hasattr(prediction, "answer") else 0
        true_answer = float(ground_truth.answer)

        # Allow 1% tolerance for floating point differences
        tolerance = abs(true_answer) * 0.01 + 0.01
        return abs(pred_answer - true_answer) <= tolerance
    except Exception:
        return False


def sentiment_accuracy_metric(prediction, ground_truth):
    """Custom metric for sentiment classification."""
    try:
        pred_sentiment = (
            prediction.sentiment.lower() if hasattr(prediction, "sentiment") else ""
        )
        true_sentiment = ground_truth.sentiment.lower()
        return pred_sentiment == true_sentiment
    except Exception:
        return False


def run_gepa_optimization(module, dataset, metric, task_name: str, iterations: int = 3):
    """Run GEPA optimization and track performance."""
    print(f"\n🔄 GEPA Optimization for {task_name}")
    print("-" * 50)

    # Split dataset
    train_size = int(len(dataset) * 0.7)
    trainset = dataset[:train_size]
    testset = dataset[train_size:]

    print(f"   Training examples: {len(trainset)}")
    print(f"   Test examples: {len(testset)}")

    # Baseline performance
    print("\n📊 Baseline Performance:")
    baseline_evaluator = Evaluate(testset, metric=metric)
    baseline_score = baseline_evaluator(module)
    print(f"   Baseline Accuracy: {baseline_score:.2%}")

    # Initialize GEPA optimizer
    try:
        gepa_optimizer = GEPA(
            metric=metric,
            breadth=3,  # Number of prompt candidates to try
            depth=iterations,  # Number of optimization rounds
        )

        print(f"\n🧠 Running GEPA Optimization ({iterations} iterations)...")

        # Run optimization
        optimized_module = gepa_optimizer.compile(module, trainset=trainset)

        # Evaluate optimized module
        print("\n📈 Optimized Performance:")
        optimized_evaluator = Evaluate(testset, metric=metric)
        optimized_score = optimized_evaluator(optimized_module)
        print(f"   Optimized Accuracy: {optimized_score:.2%}")

        # Calculate improvement
        improvement = optimized_score - baseline_score
        improvement_percent = (
            (improvement / baseline_score * 100) if baseline_score > 0 else 0
        )

        print("\n🎯 Optimization Results:")
        print(f"   Improvement: {improvement:.2%}")
        print(f"   Relative Improvement: {improvement_percent:.1f}%")

        return {
            "task": task_name,
            "baseline_score": baseline_score,
            "optimized_score": optimized_score,
            "improvement": improvement,
            "improvement_percent": improvement_percent,
            "module": optimized_module,
        }

    except Exception as e:
        print(f"   ❌ GEPA optimization failed: {e}")
        return {
            "task": task_name,
            "baseline_score": baseline_score,
            "error": str(e),
        }


def demonstrate_prompt_evolution(module, example):
    """Show how prompts evolve through GEPA optimization."""
    print("\n🔍 Prompt Evolution Demo:")
    print("-" * 30)

    # Show original prompt behavior
    print("📝 Original Module:")
    try:
        if hasattr(module, "solve"):
            result = module.solve(problem=example.problem)
        elif hasattr(module, "classify"):
            result = module.classify(text=example.text)
        elif hasattr(module, "answer"):
            result = module.answer(context=example.context, question=example.question)
        else:
            return

        input_preview = list(example.inputs().values())[0][:60]
        output_preview = str(result)[:100]
        print(f"   Input: {input_preview}...")
        print(f"   Output: {output_preview}...")

    except Exception as e:
        print(f"   Error demonstrating: {e}")


def main():
    """Run GEPA optimization examples."""
    print("🧠 DSPy GEPA (Generative Prompt Auto-optimization)")
    print("=" * 60)

    # Configure model
    try:
        if os.getenv("OPENAI_API_KEY"):
            dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))
            print("✅ Using OpenAI GPT-4o-mini")
        else:
            dspy.configure(lm=dspy.LM("ollama_chat/llama3"))
            print("✅ Using Ollama Llama3")
            print("⚠️  Note: GEPA works best with more capable models")
    except Exception as e:
        print(f"❌ Model configuration failed: {e}")
        return

    # Create datasets
    math_dataset = create_math_dataset()
    sentiment_dataset = create_sentiment_dataset()
    qa_dataset = create_qa_dataset()

    optimization_results = []

    # Test 1: Math Word Problems
    print("\n" + "=" * 60)
    print("🔢 Testing: Math Word Problem Optimization")
    math_module = MathWordProblemSolver()

    # Show example before optimization
    if math_dataset:
        demonstrate_prompt_evolution(math_module, math_dataset[0])

    math_result = run_gepa_optimization(
        math_module,
        math_dataset,
        math_accuracy_metric,
        "Math Problems",
        iterations=2,
    )
    optimization_results.append(math_result)

    # Test 2: Sentiment Classification
    print("\n" + "=" * 60)
    print("😊 Testing: Sentiment Classification Optimization")
    sentiment_module = TextClassifier()

    if sentiment_dataset:
        demonstrate_prompt_evolution(sentiment_module, sentiment_dataset[0])

    sentiment_result = run_gepa_optimization(
        sentiment_module,
        sentiment_dataset,
        sentiment_accuracy_metric,
        "Sentiment Analysis",
        iterations=2,
    )
    optimization_results.append(sentiment_result)

    # Test 3: Question Answering (if we have time/resources)
    print("\n" + "=" * 60)
    print("❓ Testing: Question Answering Optimization")
    qa_module = QAModule()

    if qa_dataset:
        demonstrate_prompt_evolution(qa_module, qa_dataset[0])

    qa_result = run_gepa_optimization(
        qa_module,
        qa_dataset,
        metrics.answer_exact_match,
        "Question Answering",
        iterations=2,
    )
    optimization_results.append(qa_result)

    # Summary of all optimizations
    print("\n" + "=" * 60)
    print("📊 GEPA Optimization Summary")
    print("-" * 60)

    successful_optimizations = [
        result for result in optimization_results if "error" not in result
    ]

    if successful_optimizations:
        for result in successful_optimizations:
            task = result["task"]
            baseline = result["baseline_score"]
            optimized = result["optimized_score"]
            improvement = result["improvement_percent"]

            print(
                f"   {task:20s}: {baseline:.1%} → {optimized:.1%} "
                f"({improvement:+.1f}%)"
            )

        avg_improvement = sum(
            r["improvement_percent"] for r in successful_optimizations
        ) / len(successful_optimizations)
        print(f"\n   Average Improvement: {avg_improvement:.1f}%")

    # Failed optimizations
    failed_optimizations = [
        result for result in optimization_results if "error" in result
    ]
    if failed_optimizations:
        print(f"\n   Failed Optimizations: {len(failed_optimizations)}")
        for result in failed_optimizations:
            print(f"      {result['task']}: {result['error']}")

    print("\n🎯 Key Insights:")
    print("   ✅ GEPA automatically improves prompts through iteration")
    print("   ✅ Self-reflective optimization reduces manual prompt engineering")
    print("   ✅ Works across different task types and domains")
    print("   ⚠️  Requires sufficient computational resources")
    print("   ⚠️  Performance depends on base model capabilities")

    print("\n🚀 Production Recommendations:")
    print("   - Use GEPA for high-value, performance-critical tasks")
    print("   - Start with good baseline prompts for better results")
    print("   - Monitor optimization costs and set appropriate budgets")
    print("   - Combine GEPA with other optimization techniques")
    print("   - Cache optimized prompts for reuse")


if __name__ == "__main__":
    main()
