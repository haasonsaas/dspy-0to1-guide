#!/usr/bin/env python3
"""
Document Summarization with DSPy

Demonstrates:
- Document summarization with different strategies
- Variable-length output control
- Multiple summarization approaches
- Quality assessment
"""

import os

import dspy
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class DocumentSummarizer(dspy.Module):
    """Flexible document summarization module."""

    def __init__(self, strategy: str = "comprehensive"):
        super().__init__()
        self.strategy = strategy

        if strategy == "comprehensive":
            self.summarize = dspy.ChainOfThought(
                "document, max_length -> summary, key_points"
            )
        elif strategy == "executive":
            self.summarize = dspy.ChainOfThought(
                "document, max_length -> executive_summary, action_items"
            )
        elif strategy == "technical":
            self.summarize = dspy.ChainOfThought(
                "document, max_length -> technical_summary, key_concepts, "
                "implications"
            )
        else:
            self.summarize = dspy.ChainOfThought("document, max_length -> summary")

    def forward(self, document: str, max_length: int = 200):
        """Summarize a document with specified maximum length."""
        return self.summarize(
            document=document, max_length=f"{max_length} words maximum"
        )


class AdaptiveSummarizer(dspy.Module):
    """Adaptive summarizer that chooses strategy based on document type."""

    def __init__(self):
        super().__init__()
        self.classifier = dspy.Predict("document -> document_type")
        self.comprehensive = DocumentSummarizer("comprehensive")
        self.executive = DocumentSummarizer("executive")
        self.technical = DocumentSummarizer("technical")

    def forward(self, document: str, max_length: int = 200):
        """Classify document type and use appropriate summarization strategy."""
        # First, classify the document
        doc_type = self.classifier(document=document[:500])  # Use first 500 chars

        # Choose appropriate summarizer
        if (
            "technical" in doc_type.document_type.lower()
            or "research" in doc_type.document_type.lower()
        ):
            result = self.technical(document, max_length)
            result.strategy_used = "technical"
        elif (
            "business" in doc_type.document_type.lower()
            or "report" in doc_type.document_type.lower()
        ):
            result = self.executive(document, max_length)
            result.strategy_used = "executive"
        else:
            result = self.comprehensive(document, max_length)
            result.strategy_used = "comprehensive"

        result.detected_type = doc_type.document_type
        return result


def get_sample_documents():
    """Get sample documents for testing."""
    return {
        "tech_article": (
            "Artificial Intelligence (AI) has revolutionized numerous industries "
            "through machine learning algorithms and neural networks. Deep learning, "
            "a subset of machine learning, utilizes multi-layered neural networks "
            "to process vast amounts of data and identify complex patterns. "
            "Convolutional Neural Networks (CNNs) excel in image recognition tasks, "
            "while Recurrent Neural Networks (RNNs) and their advanced variants like "
            "Long Short-Term Memory (LSTM) networks are particularly effective for "
            "sequential data processing, including natural language processing and "
            "time series analysis.\n\n"
            'The transformer architecture, introduced in the "Attention Is All You '
            'Need" paper, has become the foundation for modern large language models '
            "like GPT and BERT. These models demonstrate remarkable capabilities in "
            "text generation, translation, and comprehension. However, they also "
            "present challenges including computational requirements, potential "
            "biases in training data, and the need for responsible AI deployment "
            "practices.\n\n"
            "Current research focuses on improving model efficiency, reducing "
            "computational costs, and developing more interpretable AI systems. "
            "Techniques like model pruning, quantization, and knowledge distillation "
            "are being explored to make AI more accessible and sustainable."
        ),
        "business_report": (
            "Q3 2024 Financial Results and Strategic Update\n\n"
            "Our company achieved record quarterly revenue of $2.8 billion, "
            "representing a 15% year-over-year increase. This growth was primarily "
            "driven by strong performance in our cloud services division, which saw "
            "28% growth, and our mobile platform, which expanded by 12%. Operating "
            "margins improved to 22%, up from 19% in the previous quarter, "
            "demonstrating effective cost management and operational efficiency "
            "improvements.\n\n"
            "Key performance indicators show positive trends across all business "
            "units. Customer acquisition costs decreased by 8% while customer "
            "lifetime value increased by 18%, indicating improved marketing "
            "effectiveness and customer satisfaction. Our new product launches "
            "generated $340 million in revenue within their first 90 days, exceeding "
            "projections by 25%.\n\n"
            "Looking forward, we are investing $500 million in R&D for the next "
            "fiscal year, focusing on artificial intelligence integration and "
            "sustainability initiatives. We expect Q4 revenue to reach $3.1 billion, "
            "with full-year projections of $11.2 billion. Strategic partnerships "
            "with three major technology companies are expected to drive additional "
            "growth opportunities in 2025.\n\n"
            "Challenges include increased competition in the cloud market and "
            "regulatory changes in key international markets. We are proactively "
            "addressing these through enhanced product differentiation and "
            "compliance initiatives."
        ),
        "news_article": (
            "Local Community Garden Initiative Shows Remarkable Success\n\n"
            "The Riverside Community Garden project, launched just 18 months ago, "
            "has transformed a vacant lot into a thriving green space that serves "
            "over 200 local families. The initiative, spearheaded by neighborhood "
            "resident Maria Santos and supported by the city council, has become a "
            "model for urban agricultural programs.\n\n"
            "The garden features 150 individual plots, a shared composting system, "
            "and a greenhouse funded through community fundraising efforts. "
            "Participants grow a diverse range of vegetables, herbs, and flowers, "
            "with surplus produce donated to the local food bank. Last year, the "
            "garden donated over 3,000 pounds of fresh produce to families in need.\n\n"
            "Beyond food production, the garden has fostered community connections. "
            "Regular workshops on sustainable gardening practices, cooking classes, "
            "and seasonal celebrations have created a strong social network among "
            "participants. Children from nearby schools participate in educational "
            "programs, learning about plant biology, nutrition, and environmental "
            "stewardship.\n\n"
            "The success has inspired similar projects in three other neighborhoods, "
            "with the city allocating additional funding for urban agriculture "
            "initiatives. Plans for expansion include a tool library, rainwater "
            "harvesting system, and solar-powered irrigation."
        ),
        "research_paper": (
            "Abstract: This study investigates the effects of microplastic pollution "
            "on marine ecosystem biodiversity in coastal environments. Using a "
            "combination of field sampling, laboratory analysis, and statistical "
            "modeling, we examined microplastic concentrations and their correlation "
            "with species diversity indices across 15 sampling sites over a 2-year "
            "period.\n\n"
            "Methodology: Water and sediment samples were collected quarterly from "
            "sites with varying levels of human activity. Microplastic particles "
            "were isolated using density separation and identified through "
            "spectroscopic analysis. Biodiversity was assessed through standardized "
            "marine life surveys, measuring species richness, abundance, and "
            "community composition.\n\n"
            "Results: Microplastic concentrations ranged from 12 to 847 particles "
            "per liter, with highest levels near urban discharge points. Statistical "
            "analysis revealed a significant negative correlation (r = -0.73, "
            "p < 0.001) between microplastic density and biodiversity indices. Sites "
            "with concentrations above 200 particles per liter showed 35% lower "
            "species diversity compared to pristine locations.\n\n"
            "Discussion: The findings suggest that microplastic pollution poses a "
            "significant threat to marine biodiversity. Potential mechanisms include "
            "physical ingestion by marine organisms, chemical leaching of additives, "
            "and disruption of food chain dynamics. These results support the need "
            "for enhanced plastic waste management policies and continued monitoring "
            "of marine ecosystems.\n\n"
            "Conclusion: Immediate action is required to address microplastic "
            "pollution sources and implement conservation strategies for affected "
            "marine habitats."
        ),
    }


def evaluate_summary_quality(original: str, summary: str) -> dict:
    """Simple heuristic evaluation of summary quality."""
    original_words = len(original.split())
    summary_words = len(summary.split())
    compression_ratio = summary_words / original_words

    # Check for key information retention (very basic)
    original_lower = original.lower()
    summary_lower = summary.lower()

    # Count important terms that appear in both
    important_indicators = [
        "percent",
        "%",
        "million",
        "billion",
        "research",
        "study",
        "results",
        "growth",
    ]
    retained_terms = sum(
        1
        for term in important_indicators
        if term in original_lower and term in summary_lower
    )

    return {
        "compression_ratio": compression_ratio,
        "original_words": original_words,
        "summary_words": summary_words,
        "retained_key_terms": retained_terms,
        "quality_score": min(
            1.0,
            (
                retained_terms
                / max(1, len([t for t in important_indicators if t in original_lower]))
            )
            * 2,
        ),
    }


def main():
    """Run document summarization examples."""
    print("📄 DSPy Document Summarization")
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

    # Get sample documents
    documents = get_sample_documents()

    # Test different summarization approaches
    print("\n🔍 Testing Different Summarization Strategies:")
    print("-" * 50)

    # Create summarizers
    comprehensive_summarizer = DocumentSummarizer("comprehensive")
    adaptive_summarizer = AdaptiveSummarizer()

    for doc_name, document in documents.items():
        print(f"\n📑 Document: {doc_name.replace('_', ' ').title()}")
        print(f"   Original length: {len(document.split())} words")

        try:
            # Test comprehensive summarization
            print("\n   🔹 Comprehensive Summary:")
            comp_result = comprehensive_summarizer(document, max_length=100)
            if hasattr(comp_result, "summary"):
                print(f"      Summary: {comp_result.summary}")
                if hasattr(comp_result, "key_points"):
                    print(f"      Key Points: {comp_result.key_points}")

            # Test adaptive summarization
            print("\n   🔹 Adaptive Summary:")
            adaptive_result = adaptive_summarizer(document, max_length=100)

            if hasattr(adaptive_result, "detected_type"):
                print(f"      Detected Type: {adaptive_result.detected_type}")
            if hasattr(adaptive_result, "strategy_used"):
                print(f"      Strategy Used: {adaptive_result.strategy_used}")

            # Display appropriate summary based on strategy
            if hasattr(adaptive_result, "executive_summary"):
                print(f"      Executive Summary: {adaptive_result.executive_summary}")
            elif hasattr(adaptive_result, "technical_summary"):
                print(f"      Technical Summary: {adaptive_result.technical_summary}")
            elif hasattr(adaptive_result, "summary"):
                print(f"      Summary: {adaptive_result.summary}")

            # Evaluate quality
            if hasattr(comp_result, "summary"):
                quality = evaluate_summary_quality(document, comp_result.summary)
                print(f"      Quality Score: {quality['quality_score']:.2f}")
                print(f"      Compression: {quality['compression_ratio']:.1%}")

        except Exception as e:
            print(f"   ❌ Error processing {doc_name}: {e}")

    print("\n🎯 Summary Analysis:")
    print("   ✅ Different strategies adapt to document types")
    print("   ✅ Flexible length control")
    print("   ✅ Key information extraction")

    print("\n🚀 Next Steps:")
    print(
        "   - Try optimization: python examples/advanced/summarization_optimization.py"
    )
    print("   - Explore evaluation: python examples/evaluation/summary_metrics.py")
    print("   - Check personas: python examples/personas/legal_lucy.py")


if __name__ == "__main__":
    main()
