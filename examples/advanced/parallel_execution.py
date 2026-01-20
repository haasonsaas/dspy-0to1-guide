#!/usr/bin/env python3
"""
Advanced DSPy: Parallel and Asynchronous Execution

Demonstrates:
- Parallel processing with multiple models
- Asynchronous DSPy operations
- Batch processing optimization
- Concurrent pipeline execution
- Performance comparison
"""

import asyncio
import os
import threading
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from typing import Any, Dict, List

import dspy
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class ParallelQA(dspy.Module):
    """Question answering module designed for parallel execution."""

    def __init__(self, model_name: str = "default"):
        super().__init__()
        self.model_name = model_name
        self.qa = dspy.ChainOfThought("question -> answer")

    def forward(self, question: str):
        """Process a single question."""
        result = self.qa(question=question)
        return {
            "question": question,
            "answer": result.answer,
            "model": self.model_name,
            "thread_id": threading.current_thread().ident,
        }


class ParallelSummarizer(dspy.Module):
    """Document summarizer for parallel processing."""

    def __init__(self, strategy: str = "comprehensive"):
        super().__init__()
        self.strategy = strategy
        self.summarizer = dspy.ChainOfThought("document -> summary")

    def forward(self, document: str):
        """Summarize a single document."""
        result = self.summarizer(document=document)
        return {
            "document_length": len(document.split()),
            "summary": result.summary,
            "strategy": self.strategy,
            "thread_id": threading.current_thread().ident,
        }


class BatchProcessor:
    """Handles batch processing of multiple tasks."""

    def __init__(self, module_class, **module_kwargs):
        self.module_class = module_class
        self.module_kwargs = module_kwargs

    def process_sequential(self, tasks: List[Any]) -> List[Dict]:
        """Process tasks sequentially (baseline)."""
        module = self.module_class(**self.module_kwargs)
        results = []

        start_time = time.time()
        for task in tasks:
            result = module(task)
            results.append(result)
        end_time = time.time()

        return {
            "results": results,
            "execution_time": end_time - start_time,
            "method": "sequential",
        }

    def process_parallel_threads(self, tasks: List[Any], max_workers: int = 4) -> Dict:
        """Process tasks using thread pool."""
        start_time = time.time()

        def process_single_task(task):
            # Create a new module instance for each thread
            module = self.module_class(**self.module_kwargs)
            return module(task)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(process_single_task, tasks))

        end_time = time.time()

        return {
            "results": results,
            "execution_time": end_time - start_time,
            "method": f"parallel_threads_{max_workers}",
            "speedup": None,  # Will be calculated later
        }

    def process_parallel_processes(
        self, tasks: List[Any], max_workers: int = 2
    ) -> Dict:
        """Process tasks using process pool (limited due to model serialization)."""
        start_time = time.time()

        def process_single_task(task):
            # Note: This may not work well with all DSPy modules due to serialization
            try:
                module = self.module_class(**self.module_kwargs)
                return module(task)
            except Exception as e:
                return {"error": str(e), "task": str(task)[:50]}

        try:
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                results = list(executor.map(process_single_task, tasks))
        except Exception as e:
            print(f"Process pool failed: {e}")
            return {
                "results": [],
                "execution_time": 0,
                "method": "process_pool_failed",
                "error": str(e),
            }

        end_time = time.time()

        return {
            "results": results,
            "execution_time": end_time - start_time,
            "method": f"parallel_processes_{max_workers}",
        }


async def async_qa_processor(questions: List[str], batch_size: int = 3):
    """Asynchronous question processing with batching."""

    async def process_batch(batch):
        """Process a batch of questions."""
        qa_module = ParallelQA("async_model")

        # Simulate async processing (DSPy doesn't have native async support yet)
        # In a real implementation, you'd use async HTTP calls to the model
        loop = asyncio.get_event_loop()

        def sync_process(question):
            return qa_module(question)

        # Run sync functions in thread pool
        with ThreadPoolExecutor() as executor:
            tasks = [
                loop.run_in_executor(executor, sync_process, question)
                for question in batch
            ]
            results = await asyncio.gather(*tasks)

        return results

    # Split questions into batches
    batches = [
        questions[i : i + batch_size] for i in range(0, len(questions), batch_size)
    ]

    start_time = time.time()

    # Process all batches concurrently
    batch_tasks = [process_batch(batch) for batch in batches]
    batch_results = await asyncio.gather(*batch_tasks)

    # Flatten results
    results = [result for batch_result in batch_results for result in batch_result]

    end_time = time.time()

    return {
        "results": results,
        "execution_time": end_time - start_time,
        "method": f"async_batched_{batch_size}",
        "num_batches": len(batches),
    }


def get_test_data():
    """Generate test data for parallel processing."""
    return {
        "questions": [
            "What is artificial intelligence?",
            "How does machine learning work?",
            "What are the benefits of cloud computing?",
            "Explain blockchain technology in simple terms.",
            "What is the difference between AI and ML?",
            "How does deep learning differ from machine learning?",
            "What are the main programming languages for data science?",
            "Explain the concept of neural networks.",
            "What is natural language processing?",
            "How do recommendation systems work?",
            "What is computer vision?",
            "Explain reinforcement learning.",
        ],
        "documents": [
            (
                "Artificial intelligence has revolutionized many industries by "
                "automating complex tasks and providing intelligent insights. "
                "Machine learning algorithms enable computers to learn from data "
                "without explicit programming. Deep learning uses neural networks "
                "to process vast amounts of information."
            ),
            (
                "Cloud computing offers scalable infrastructure and services over "
                "the internet. Organizations can reduce costs and improve "
                "flexibility by leveraging cloud platforms. Popular cloud "
                "providers include AWS, Azure, and Google Cloud Platform."
            ),
            (
                "Blockchain technology creates immutable, distributed ledgers "
                "that enable secure transactions without intermediaries. "
                "Cryptocurrencies like Bitcoin utilize blockchain for "
                "decentralized finance. Smart contracts automate agreement "
                "execution."
            ),
            (
                "Data science combines statistics, programming, and domain "
                "expertise to extract insights from data. Python and R are popular "
                "languages for data analysis. Machine learning models help "
                "predict future trends and behaviors."
            ),
            (
                "Cybersecurity protects digital assets from threats and "
                "vulnerabilities. Organizations implement multi-layered security "
                "strategies including firewalls, encryption, and access controls. "
                "Regular security audits help identify potential risks."
            ),
        ],
    }


def calculate_performance_metrics(results: List[Dict]) -> Dict:
    """Calculate performance metrics for different execution methods."""
    if not results:
        return {}

    # Find baseline (sequential) performance
    baseline_time = None
    for result in results:
        if result["method"] == "sequential":
            baseline_time = result["execution_time"]
            break

    if baseline_time is None:
        return {"error": "No baseline found"}

    metrics = {}
    for result in results:
        method = result["method"]
        execution_time = result["execution_time"]
        speedup = baseline_time / execution_time if execution_time > 0 else 0
        efficiency = (
            speedup / int(method.split("_")[-1])
            if method.count("_") > 1 and method.split("_")[-1].isdigit()
            else speedup
        )

        metrics[method] = {
            "execution_time": execution_time,
            "speedup": speedup,
            "efficiency": efficiency,
            "tasks_processed": len(result.get("results", [])),
        }

    return metrics


def main():
    """Run parallel execution examples."""
    print("⚡ DSPy Parallel and Asynchronous Execution")
    print("=" * 60)

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

    # Get test data
    test_data = get_test_data()

    print("\n📋 Testing Parallel Question Answering:")
    print(f"   Questions to process: {len(test_data['questions'])}")
    print("-" * 50)

    # Test question answering with different parallel strategies
    qa_processor = BatchProcessor(ParallelQA, model_name="qa_model")
    qa_results = []

    # Sequential baseline
    print("\n🔄 Sequential Processing:")
    sequential_result = qa_processor.process_sequential(
        test_data["questions"][:6]
    )  # Use fewer for demo
    qa_results.append(sequential_result)
    print(f"   Time: {sequential_result['execution_time']:.2f}s")
    print(f"   Questions processed: {len(sequential_result['results'])}")

    # Parallel with threads
    print("\n🚀 Parallel Processing (Threads):")
    for workers in [2, 4]:
        parallel_result = qa_processor.process_parallel_threads(
            test_data["questions"][:6], max_workers=workers
        )
        qa_results.append(parallel_result)
        speedup = (
            sequential_result["execution_time"] / parallel_result["execution_time"]
        )
        print(
            "   Workers: "
            f"{workers} | Time: {parallel_result['execution_time']:.2f}s | "
            f"Speedup: {speedup:.2f}x"
        )

    # Test document summarization
    print("\n📄 Testing Parallel Document Summarization:")
    print(f"   Documents to process: {len(test_data['documents'])}")
    print("-" * 50)

    summarizer_processor = BatchProcessor(ParallelSummarizer, strategy="comprehensive")

    # Sequential
    print("\n🔄 Sequential Summarization:")
    seq_summary = summarizer_processor.process_sequential(test_data["documents"])
    print(f"   Time: {seq_summary['execution_time']:.2f}s")

    # Parallel
    print("\n🚀 Parallel Summarization:")
    parallel_summary = summarizer_processor.process_parallel_threads(
        test_data["documents"], max_workers=3
    )
    speedup = seq_summary["execution_time"] / parallel_summary["execution_time"]
    print(
        f"   Time: {parallel_summary['execution_time']:.2f}s | Speedup: {speedup:.2f}x"
    )

    # Test async processing
    print("\n⚡ Testing Asynchronous Processing:")
    print("-" * 40)

    async def run_async_test():
        async_result = await async_qa_processor(
            test_data["questions"][:8], batch_size=3
        )
        return async_result

    print("\n🔄 Async Batch Processing:")
    async_result = asyncio.run(run_async_test())
    print(f"   Time: {async_result['execution_time']:.2f}s")
    print(f"   Batches: {async_result['num_batches']}")
    print(f"   Questions processed: {len(async_result['results'])}")

    # Performance Analysis
    all_results = qa_results + [async_result]
    metrics = calculate_performance_metrics(all_results)

    print("\n📊 Performance Analysis:")
    print("-" * 40)

    if metrics and "error" not in metrics:
        for method, stats in metrics.items():
            print(
                f"   {method:20s}: {stats['execution_time']:.2f}s "
                f"(Speedup: {stats['speedup']:.2f}x)"
            )

    print("\n🎯 Key Insights:")
    print("   ✅ Thread-based parallelism effective for I/O-bound tasks")
    print("   ✅ Async processing good for batching operations")
    print("   ⚠️  Process-based parallelism limited by model serialization")
    print("   ✅ Parallel execution reduces total processing time")
    print("   ⚠️  Overhead exists - not all tasks benefit equally")

    print("\n🚀 Production Recommendations:")
    print("   - Use ThreadPoolExecutor for DSPy parallel processing")
    print("   - Batch related operations for efficiency")
    print("   - Monitor resource usage and adjust worker count")
    print("   - Consider async patterns for high-throughput scenarios")
    print("   - Profile your specific use case for optimal configuration")


if __name__ == "__main__":
    main()
