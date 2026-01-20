#!/usr/bin/env python3
"""
Dataset Preparation for DSPy 0-to-1 Guide

This script creates sample datasets for training and evaluation.
"""

import os
import json
import csv
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime
import random

def create_qa_dataset():
    """Create question-answering dataset."""
    qa_data = [
        {
            "question": "What is machine learning?",
            "answer": "Machine learning is a subset of artificial intelligence that enables computers to learn and make decisions from data without explicit programming.",
            "category": "AI/ML",
            "difficulty": "beginner"
        },
        {
            "question": "How does a neural network work?",
            "answer": "A neural network is a computational model inspired by biological neural networks. It consists of interconnected nodes (neurons) that process information through weighted connections and activation functions.",
            "category": "AI/ML",
            "difficulty": "intermediate"
        },
        {
            "question": "What is the difference between supervised and unsupervised learning?",
            "answer": "Supervised learning uses labeled training data to learn patterns, while unsupervised learning finds hidden patterns in unlabeled data without specific target outputs.",
            "category": "AI/ML",
            "difficulty": "intermediate"
        },
        {
            "question": "What is cloud computing?",
            "answer": "Cloud computing is the delivery of computing services including servers, storage, databases, networking, software, analytics, and intelligence over the internet.",
            "category": "Technology",
            "difficulty": "beginner"
        },
        {
            "question": "How does blockchain technology work?",
            "answer": "Blockchain is a distributed ledger technology that maintains a continuously growing list of records (blocks) that are linked and secured using cryptography.",
            "category": "Technology",
            "difficulty": "intermediate"
        },
        {
            "question": "What is the capital of France?",
            "answer": "Paris",
            "category": "Geography",
            "difficulty": "beginner"
        },
        {
            "question": "Who invented the telephone?",
            "answer": "Alexander Graham Bell",
            "category": "History",
            "difficulty": "beginner"
        },
        {
            "question": "What is photosynthesis?",
            "answer": "Photosynthesis is the process by which plants use sunlight, water, and carbon dioxide to produce oxygen and energy in the form of sugar.",
            "category": "Science",
            "difficulty": "beginner"
        }
    ]

    return qa_data

def create_sentiment_dataset():
    """Create sentiment analysis dataset."""
    sentiment_data = [
        {
            "text": "I absolutely love this product! It exceeded all my expectations and works perfectly.",
            "sentiment": "positive",
            "confidence": 0.95
        },
        {
            "text": "This is the worst purchase I've ever made. Complete waste of money and terrible quality.",
            "sentiment": "negative",
            "confidence": 0.98
        },
        {
            "text": "The product is okay, nothing special but it does what it's supposed to do.",
            "sentiment": "neutral",
            "confidence": 0.85
        },
        {
            "text": "Outstanding customer service and amazing product quality! Highly recommend to everyone.",
            "sentiment": "positive",
            "confidence": 0.97
        },
        {
            "text": "Disappointed with the purchase. Poor build quality and doesn't work as advertised.",
            "sentiment": "negative",
            "confidence": 0.92
        },
        {
            "text": "Average product with standard features. Price seems reasonable for what you get.",
            "sentiment": "neutral",
            "confidence": 0.80
        },
        {
            "text": "Fantastic experience from start to finish! Will definitely buy again.",
            "sentiment": "positive",
            "confidence": 0.94
        },
        {
            "text": "Not impressed. Expected much better based on reviews. Would not recommend.",
            "sentiment": "negative",
            "confidence": 0.89
        }
    ]

    return sentiment_data

def create_math_dataset():
    """Create mathematical word problems dataset."""
    math_data = [
        {
            "problem": "Sarah has 24 apples. She gives 8 apples to her friend and then buys 15 more. How many apples does she have now?",
            "answer": 31,
            "solution_steps": ["Start: 24 apples", "Give away: 24 - 8 = 16 apples", "Buy more: 16 + 15 = 31 apples"],
            "category": "arithmetic"
        },
        {
            "problem": "A store sells shirts for $25 each. If John buys 3 shirts and pays with a $100 bill, how much change should he receive?",
            "answer": 25,
            "solution_steps": ["Cost: 3 × $25 = $75", "Change: $100 - $75 = $25"],
            "category": "arithmetic"
        },
        {
            "problem": "Lisa runs 2.5 miles every day for 6 days. How many total miles does she run?",
            "answer": 15.0,
            "solution_steps": ["Daily distance: 2.5 miles", "Total: 2.5 × 6 = 15.0 miles"],
            "category": "multiplication"
        },
        {
            "problem": "A rectangular garden is 12 feet long and 8 feet wide. What is its area in square feet?",
            "answer": 96,
            "solution_steps": ["Area = length × width", "Area = 12 × 8 = 96 square feet"],
            "category": "geometry"
        },
        {
            "problem": "Tom saves $50 per month for 8 months. How much money has he saved in total?",
            "answer": 400,
            "solution_steps": ["Monthly savings: $50", "Total: $50 × 8 = $400"],
            "category": "multiplication"
        }
    ]

    return math_data

def create_classification_dataset():
    """Create text classification dataset."""
    classification_data = [
        {
            "text": "Breaking: Stock market reaches all-time high as tech companies report strong quarterly earnings",
            "category": "business",
            "subcategory": "finance"
        },
        {
            "text": "New study reveals that regular exercise can reduce the risk of heart disease by up to 30%",
            "category": "health",
            "subcategory": "research"
        },
        {
            "text": "Scientists discover new exoplanet that could potentially support life",
            "category": "science",
            "subcategory": "astronomy"
        },
        {
            "text": "Local high school wins state championship in basketball after undefeated season",
            "category": "sports",
            "subcategory": "basketball"
        },
        {
            "text": "New artificial intelligence model achieves breakthrough in natural language understanding",
            "category": "technology",
            "subcategory": "ai"
        },
        {
            "text": "Celebrity couple announces engagement after two years of dating",
            "category": "entertainment",
            "subcategory": "celebrity"
        },
        {
            "text": "Government announces new climate change initiatives to reduce carbon emissions",
            "category": "politics",
            "subcategory": "environment"
        },
        {
            "text": "Chef opens innovative farm-to-table restaurant featuring local ingredients",
            "category": "lifestyle",
            "subcategory": "food"
        }
    ]

    return classification_data

def create_summarization_dataset():
    """Create document summarization dataset."""
    summarization_data = [
        {
            "document": """
            Artificial intelligence (AI) has become increasingly prevalent in modern society, transforming industries
            from healthcare to transportation. Machine learning, a subset of AI, enables computers to learn from data
            without explicit programming. Deep learning, which uses neural networks with multiple layers, has achieved
            remarkable breakthroughs in image recognition, natural language processing, and game playing. Companies
            are investing billions of dollars in AI research and development, while governments are creating policies
            to regulate its use. Despite the benefits, concerns about job displacement, privacy, and algorithmic bias
            remain significant challenges that society must address.
            """.strip(),
            "summary": "AI and machine learning are transforming industries with significant investment, but concerns about job displacement and bias need addressing.",
            "key_points": [
                "AI is transforming multiple industries",
                "Machine learning enables computers to learn from data",
                "Deep learning has achieved breakthroughs in various domains",
                "Massive investment in AI R&D",
                "Concerns about job displacement and bias exist"
            ]
        },
        {
            "document": """
            Climate change represents one of the most pressing challenges of our time. Global temperatures have risen
            by approximately 1.1 degrees Celsius since the late 19th century, primarily due to human activities such
            as burning fossil fuels and deforestation. The impacts are already visible: melting ice caps, rising sea
            levels, extreme weather events, and disruptions to ecosystems. The Intergovernmental Panel on Climate
            Change (IPCC) warns that limiting warming to 1.5°C requires rapid and far-reaching transitions in energy,
            land, urban infrastructure, and industrial systems. Solutions include renewable energy adoption, energy
            efficiency improvements, carbon pricing, and international cooperation through agreements like the Paris
            Climate Accord.
            """.strip(),
            "summary": "Climate change, driven by human activities, requires urgent action including renewable energy adoption and international cooperation to limit global warming.",
            "key_points": [
                "Global temperatures have risen 1.1°C since the 19th century",
                "Human activities are the primary cause",
                "Visible impacts include melting ice caps and extreme weather",
                "IPCC warns rapid transitions are needed",
                "Solutions include renewable energy and international cooperation"
            ]
        }
    ]

    return summarization_data

def save_dataset(data: List[Dict], filepath: Path, format: str = "json"):
    """Save dataset in specified format."""
    filepath.parent.mkdir(parents=True, exist_ok=True)

    if format == "json":
        with open(filepath, 'w') as f:
            json.dump({
                "metadata": {
                    "created": datetime.now().isoformat(),
                    "count": len(data),
                    "description": f"Dataset for {filepath.stem}"
                },
                "data": data
            }, f, indent=2)

    elif format == "jsonl":
        with open(filepath, 'w') as f:
            for item in data:
                f.write(json.dumps(item) + '\n')

    elif format == "csv":
        if data:
            with open(filepath, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=data[0].keys())
                writer.writeheader()
                writer.writerows(data)

def create_dspy_examples(data: List[Dict], input_fields: List[str], output_fields: List[str]):
    """Convert dataset to DSPy Example format."""
    dspy_examples = []

    for item in data:
        example_data = {}

        # Add input fields
        for field in input_fields:
            if field in item:
                example_data[field] = item[field]

        # Add output fields
        for field in output_fields:
            if field in item:
                example_data[field] = item[field]

        dspy_examples.append(example_data)

    return dspy_examples

def generate_train_test_split(data: List[Dict], train_ratio: float = 0.8):
    """Split data into training and test sets."""
    random.shuffle(data)
    split_idx = int(len(data) * train_ratio)

    return {
        "train": data[:split_idx],
        "test": data[split_idx:]
    }

def main():
    """Create all sample datasets."""
    print("📊 Preparing Sample Datasets for DSPy 0-to-1 Guide")
    print("=" * 60)

    datasets_dir = Path("datasets/sample_data")
    datasets_dir.mkdir(parents=True, exist_ok=True)

    # Create datasets
    datasets = {
        "qa": create_qa_dataset(),
        "sentiment": create_sentiment_dataset(),
        "math": create_math_dataset(),
        "classification": create_classification_dataset(),
        "summarization": create_summarization_dataset()
    }

    print(f"🏗️  Creating datasets in: {datasets_dir}")

    for name, data in datasets.items():
        print(f"\n📁 {name.title()} Dataset ({len(data)} examples)")

        # Save in multiple formats
        save_dataset(data, datasets_dir / f"{name}.json", "json")
        save_dataset(data, datasets_dir / f"{name}.jsonl", "jsonl")

        # Create train/test splits
        split_data = generate_train_test_split(data)
        save_dataset(split_data["train"], datasets_dir / f"{name}_train.json", "json")
        save_dataset(split_data["test"], datasets_dir / f"{name}_test.json", "json")

        print(f"   ✅ Saved {name}.json, {name}.jsonl, train/test splits")
        print(f"      Train: {len(split_data['train'])}, Test: {len(split_data['test'])}")

    # Create DSPy-specific examples
    print(f"\n🔧 Creating DSPy Example Files:")

    # QA examples
    qa_examples = create_dspy_examples(
        datasets["qa"],
        input_fields=["question"],
        output_fields=["answer"]
    )
    save_dataset(qa_examples, datasets_dir / "dspy_qa_examples.json", "json")
    print(f"   ✅ DSPy QA examples: {len(qa_examples)}")

    # Sentiment examples
    sentiment_examples = create_dspy_examples(
        datasets["sentiment"],
        input_fields=["text"],
        output_fields=["sentiment"]
    )
    save_dataset(sentiment_examples, datasets_dir / "dspy_sentiment_examples.json", "json")
    print(f"   ✅ DSPy Sentiment examples: {len(sentiment_examples)}")

    # Math examples
    math_examples = create_dspy_examples(
        datasets["math"],
        input_fields=["problem"],
        output_fields=["answer"]
    )
    save_dataset(math_examples, datasets_dir / "dspy_math_examples.json", "json")
    print(f"   ✅ DSPy Math examples: {len(math_examples)}")

    # Create summary statistics
    total_examples = sum(len(data) for data in datasets.values())

    summary = {
        "total_datasets": len(datasets),
        "total_examples": total_examples,
        "datasets": {
            name: {
                "count": len(data),
                "description": f"{name.title()} dataset for DSPy training and evaluation"
            }
            for name, data in datasets.items()
        },
        "formats_created": ["JSON", "JSONL", "Train/Test splits", "DSPy Examples"],
        "created_at": datetime.now().isoformat()
    }

    save_dataset(summary, datasets_dir / "dataset_summary.json", "json")

    print(f"\n📈 Dataset Creation Summary:")
    print(f"   Total Datasets: {summary['total_datasets']}")
    print(f"   Total Examples: {summary['total_examples']}")
    print(f"   Formats: {', '.join(summary['formats_created'])}")
    print(f"   Location: {datasets_dir}")

    print(f"\n🎯 Usage Examples:")
    print(f"   Load QA dataset: data = json.load(open('{datasets_dir}/qa.json'))['data']")
    print(f"   Load for DSPy: examples = json.load(open('{datasets_dir}/dspy_qa_examples.json'))['data']")
    print(f"   Train/test: train = json.load(open('{datasets_dir}/qa_train.json'))['data']")

    print(f"\n✅ Dataset preparation completed!")

if __name__ == "__main__":
    main()