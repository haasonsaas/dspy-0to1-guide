#!/usr/bin/env python3
"""
Support-Sam: Customer Support with Knowledge Base

This persona demonstrates:
- RAG-based customer support
- Ticket classification and routing
- Response generation with knowledge base
- Customer satisfaction scoring
- Escalation detection
"""

import math
import os
from collections import Counter
from datetime import datetime
from typing import Dict, List

import dspy
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class TicketClassification(dspy.Signature):
    """Classify customer support tickets into categories and urgency levels."""

    ticket_content: str = dspy.InputField(desc="Full customer ticket content")
    customer_history: str = dspy.InputField(
        desc="Previous interactions summary", default="No prior history"
    )

    category: str = dspy.OutputField(
        desc="technical, billing, account, product, or general"
    )
    urgency: str = dspy.OutputField(desc="low, medium, high, or critical")
    sentiment: str = dspy.OutputField(desc="positive, neutral, negative, or frustrated")
    requires_escalation: bool = dspy.OutputField(desc="True if needs human agent")


class SupportResponse(dspy.Signature):
    """Generate customer support responses using knowledge base."""

    ticket_content: str = dspy.InputField(desc="Customer's issue description")
    category: str = dspy.InputField(desc="Ticket category")
    urgency: str = dspy.InputField(desc="Ticket urgency level")
    knowledge_base: str = dspy.InputField(desc="Relevant knowledge base articles")
    customer_history: str = dspy.InputField(
        desc="Customer interaction history", default=""
    )

    response: str = dspy.OutputField(desc="Professional, helpful customer response")
    solution_steps: str = dspy.OutputField(desc="Step-by-step solution if applicable")
    follow_up_needed: bool = dspy.OutputField(desc="Whether follow-up is required")


class SatisfactionPredictor(dspy.Signature):
    """Predict customer satisfaction based on response quality."""

    original_ticket: str = dspy.InputField(desc="Customer's original issue")
    support_response: str = dspy.InputField(desc="Support team's response")
    resolution_time: str = dspy.InputField(desc="Time to resolution")

    satisfaction_score: int = dspy.OutputField(desc="Predicted satisfaction (1-10)")
    improvement_suggestions: str = dspy.OutputField(desc="How to improve the response")


class SupportSam(dspy.Module):
    """Complete customer support agent with RAG and intelligence."""

    def __init__(self):
        super().__init__()
        self.classifier = dspy.ChainOfThought(TicketClassification)
        self.responder = dspy.ChainOfThought(SupportResponse)
        self.satisfaction = dspy.ChainOfThought(SatisfactionPredictor)

        # Load knowledge base with vector similarity
        self.knowledge_base = self._load_knowledge_base()
        self.kb_vectors = self._vectorize_knowledge_base()

    def _load_knowledge_base(self) -> Dict[str, List[str]]:
        """Load knowledge base articles by category."""
        return {
            "technical": [
                (
                    "Password Reset: Go to Settings > Account > Reset Password. "
                    "Check email within 5 minutes."
                ),
                (
                    "App Crashes: Clear app cache, restart device, update to "
                    "latest version."
                ),
                (
                    "Sync Issues: Ensure internet connection, force sync in "
                    "Settings > Data."
                ),
                (
                    "Login Problems: Verify credentials, check account status, "
                    "try password reset."
                ),
            ],
            "billing": [
                "Refund Policy: Refunds available within 30 days for premium features.",
                (
                    "Subscription Management: Manage subscriptions in Account > "
                    "Billing > Subscriptions."
                ),
                (
                    "Payment Failed: Update payment method in Account > Payment "
                    "Methods."
                ),
                (
                    "Billing Questions: Charges appear 1-2 business days after "
                    "subscription."
                ),
            ],
            "account": [
                (
                    "Account Deletion: Submit deletion request via Settings > "
                    "Privacy > Delete Account."
                ),
                (
                    "Email Change: Update email in Account Settings, verify with "
                    "confirmation email."
                ),
                (
                    "Profile Updates: Modify profile information in Settings > "
                    "Profile."
                ),
                "Privacy Settings: Control data sharing in Settings > Privacy.",
            ],
            "product": [
                "Feature Requests: Submit requests via Help > Feature Request form.",
                "Bug Reports: Report bugs with screenshots via Help > Report Bug.",
                (
                    "Product Updates: New features announced monthly via email "
                    "and app notifications."
                ),
                (
                    "Premium Features: Upgrade to premium for advanced analytics "
                    "and priority support."
                ),
            ],
        }

    def _vectorize_knowledge_base(self) -> Dict[str, List[tuple]]:
        """Create vector representations of knowledge base articles."""
        kb_vectors = {}

        # Simple TF-IDF vectorization
        all_articles = []
        for category, articles in self.knowledge_base.items():
            all_articles.extend(articles)

        # Calculate document frequency for each word
        word_doc_freq = Counter()
        for article in all_articles:
            words = set(article.lower().split())
            for word in words:
                if len(word) > 2:  # Skip short words
                    word_doc_freq[word] += 1

        # Vectorize each category's articles
        for category, articles in self.knowledge_base.items():
            category_vectors = []
            for article in articles:
                words = article.lower().split()
                word_freq = Counter(words)

                # Create TF-IDF vector
                vector = {}
                for word, freq in word_freq.items():
                    if len(word) > 2:
                        tf = freq / len(words)
                        idf = math.log(len(all_articles) / (word_doc_freq[word] + 1))
                        vector[word] = tf * idf

                category_vectors.append((article, vector))

            kb_vectors[category] = category_vectors

        return kb_vectors

    def _calculate_similarity(
        self, query_vector: Dict[str, float], doc_vector: Dict[str, float]
    ) -> float:
        """Calculate cosine similarity between query and document vectors."""
        # Calculate dot product
        dot_product = sum(
            query_vector.get(word, 0) * doc_vector.get(word, 0)
            for word in set(query_vector.keys()) | set(doc_vector.keys())
        )

        # Calculate magnitudes
        query_magnitude = math.sqrt(sum(val**2 for val in query_vector.values()))
        doc_magnitude = math.sqrt(sum(val**2 for val in doc_vector.values()))

        if query_magnitude == 0 or doc_magnitude == 0:
            return 0.0

        return dot_product / (query_magnitude * doc_magnitude)

    def _retrieve_knowledge(self, category: str, ticket_content: str) -> str:
        """Vector-based knowledge retrieval with TF-IDF similarity."""
        articles = self.kb_vectors.get(category, [])
        if not articles:
            return "No relevant knowledge base articles found."

        # Vectorize the query (ticket content)
        query_words = ticket_content.lower().split()
        query_freq = Counter(query_words)

        query_vector = {}
        for word, freq in query_freq.items():
            if len(word) > 2:
                tf = freq / len(query_words)
                # Use a simple IDF approximation for the query
                idf = 1.0  # Could be improved with proper IDF calculation
                query_vector[word] = tf * idf

        # Calculate similarities
        relevant_articles = []
        for article, doc_vector in articles:
            similarity = self._calculate_similarity(query_vector, doc_vector)
            if similarity > 0.01:  # Minimum threshold
                relevant_articles.append((article, similarity))

        # Sort by similarity and return top articles
        relevant_articles.sort(key=lambda x: x[1], reverse=True)

        if not relevant_articles:
            # Fallback to keyword matching if no vector similarities
            return self._fallback_keyword_matching(category, ticket_content)

        return " | ".join(article for article, _ in relevant_articles[:3])

    def _fallback_keyword_matching(self, category: str, ticket_content: str) -> str:
        """Fallback keyword matching when vector similarity fails."""
        articles = self.knowledge_base.get(category, [])
        relevant_articles = []
        ticket_lower = ticket_content.lower()

        for article in articles:
            article_words = article.lower().split()
            ticket_words = ticket_lower.split()

            # Count matching words
            matches = sum(
                1
                for word in ticket_words
                if any(word in article_word for article_word in article_words)
            )
            if matches > 0:
                relevant_articles.append((article, matches))

        # Sort by relevance and return top articles
        relevant_articles.sort(key=lambda x: x[1], reverse=True)
        return " | ".join(article for article, _ in relevant_articles[:3])

    def process_ticket(
        self,
        ticket_content: str,
        customer_id: str = "unknown",
        customer_history: str = "",
    ) -> Dict:
        """Process a customer support ticket end-to-end."""

        # Step 1: Classify the ticket
        classification = self.classifier(
            ticket_content=ticket_content,
            customer_history=customer_history,
        )

        # Step 2: Retrieve relevant knowledge
        knowledge = self._retrieve_knowledge(classification.category, ticket_content)

        # Step 3: Generate response
        response = self.responder(
            ticket_content=ticket_content,
            category=classification.category,
            urgency=classification.urgency,
            knowledge_base=knowledge,
            customer_history=customer_history,
        )

        # Step 4: Predict satisfaction
        satisfaction = self.satisfaction(
            original_ticket=ticket_content,
            support_response=response.response,
            resolution_time="< 1 hour (automated response)",
        )

        return {
            "ticket_id": f"TK-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            "customer_id": customer_id,
            "classification": {
                "category": classification.category,
                "urgency": classification.urgency,
                "sentiment": classification.sentiment,
                "requires_escalation": classification.requires_escalation,
            },
            "response": {
                "message": response.response,
                "solution_steps": response.solution_steps,
                "follow_up_needed": response.follow_up_needed,
            },
            "prediction": {
                "satisfaction_score": satisfaction.satisfaction_score,
                "improvement_suggestions": satisfaction.improvement_suggestions,
            },
            "knowledge_used": knowledge,
            "processed_at": datetime.now().isoformat(),
        }


def get_test_tickets():
    """Get sample customer support tickets for testing."""
    return [
        {
            "content": (
                "Hi, I can't log into my account. I keep getting an error message "
                "that says 'invalid credentials' but I know my password is "
                "correct. This is really frustrating because I need to access my "
                "data for an important meeting tomorrow."
            ),
            "customer_id": "CUST001",
            "history": (
                "Premium customer, usually tech-savvy, last contact 3 months ago "
                "about billing"
            ),
        },
        {
            "content": (
                "I was charged twice this month for my subscription. Can someone "
                "please explain why and process a refund? I only have one active "
                "subscription but see two charges on my credit card statement."
            ),
            "customer_id": "CUST002",
            "history": (
                "Free tier user upgraded to premium 2 months ago, no prior issues"
            ),
        },
        {
            "content": (
                "The mobile app keeps crashing every time I try to open the "
                "analytics dashboard. I've tried restarting my phone but it "
                "doesn't help. This is a critical issue as I need these reports "
                "for my team."
            ),
            "customer_id": "CUST003",
            "history": (
                "Business customer, frequently uses mobile app, reported minor "
                "bugs before"
            ),
        },
        {
            "content": (
                "I love the new update but I'm wondering if you'll be adding dark "
                "mode soon? Also, it would be great to have more export options "
                "for the reports."
            ),
            "customer_id": "CUST004",
            "history": (
                "Long-time user, active in community forums, generally positive"
            ),
        },
        {
            "content": (
                "THIS IS RIDICULOUS! Your service has been down for 3 hours and I "
                "haven't received any updates. I'm losing business because of "
                "this. I want a full refund and compensation for my losses!"
            ),
            "customer_id": "CUST005",
            "history": (
                "Business customer, previously escalated issues, high-value " "account"
            ),
        },
        {
            "content": (
                "How do I delete my account? I can't find the option anywhere in "
                "the settings."
            ),
            "customer_id": "CUST006",
            "history": "Recent signup, minimal usage, no prior contact",
        },
    ]


def main():
    """Run Support-Sam customer service examples."""
    print("🎧 Support-Sam: AI Customer Support Agent")
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

    # Create Support-Sam agent
    sam = SupportSam()

    # Process test tickets
    test_tickets = get_test_tickets()

    print(f"\n📋 Processing {len(test_tickets)} Customer Tickets:")
    print("-" * 60)

    results = []
    escalations = 0
    total_satisfaction = 0

    for i, ticket in enumerate(test_tickets, 1):
        print(f"\n🎫 Ticket #{i} (Customer: {ticket['customer_id']})")
        print(f"   Content: {ticket['content'][:100]}...")

        try:
            # Process the ticket
            result = sam.process_ticket(
                ticket_content=ticket["content"],
                customer_id=ticket["customer_id"],
                customer_history=ticket["history"],
            )

            results.append(result)

            # Display results
            classification = result["classification"]
            response = result["response"]
            prediction = result["prediction"]
            escalation_label = "Yes" if classification["requires_escalation"] else "No"

            print("   📊 Classification:")
            print(f"      Category: {classification['category'].title()}")
            print(f"      Urgency: {classification['urgency'].title()}")
            print(f"      Sentiment: {classification['sentiment'].title()}")
            print(f"      Escalation Needed: {escalation_label}")

            print("   💬 Response:")
            print(f"      {response['message'][:150]}...")

            print("   📈 Prediction:")
            print(f"      Satisfaction Score: {prediction['satisfaction_score']}/10")

            # Track metrics
            if classification["requires_escalation"]:
                escalations += 1

            try:
                total_satisfaction += int(prediction["satisfaction_score"])
            except (TypeError, ValueError):
                pass

        except Exception as e:
            print(f"   ❌ Error processing ticket: {e}")

    # Summary statistics
    print("\n📊 Support-Sam Performance Summary:")
    print("-" * 40)
    print(f"   Tickets Processed: {len(results)}")
    if results:
        escalation_rate = escalations / len(results) * 100
        print(f"   Escalations Required: {escalations} ({escalation_rate:.1f}%)")
    if total_satisfaction > 0 and results:
        avg_satisfaction = total_satisfaction / len(results)
        print(f"   Average Predicted Satisfaction: {avg_satisfaction:.1f}/10")

    # Category breakdown
    categories = {}
    urgencies = {}
    for result in results:
        cat = result["classification"]["category"]
        urg = result["classification"]["urgency"]
        categories[cat] = categories.get(cat, 0) + 1
        urgencies[urg] = urgencies.get(urg, 0) + 1

    print("\n   📋 Category Breakdown:")
    for category, count in sorted(categories.items()):
        print(f"      {category.title()}: {count}")

    print("\n   ⚡ Urgency Breakdown:")
    for urgency, count in sorted(urgencies.items()):
        print(f"      {urgency.title()}: {count}")

    print("\n🎯 Key Insights:")
    print("   ✅ Automated ticket classification and routing")
    print("   ✅ Knowledge base integration for consistent responses")
    print("   ✅ Satisfaction prediction for quality control")
    print("   ✅ Escalation detection for complex issues")

    print("\n🚀 Scaling & Integration:")
    print("   - This implementation uses TF-IDF vector similarity")
    print("   - For larger scale: replace with FAISS or Pinecone vector DB")
    print("   - CRM integration: add API calls to Salesforce/HubSpot")
    print("   - Multi-language: add translation layer with Google Translate API")
    print("   - Follow-up: integrate with email/SMS APIs for automated responses")


if __name__ == "__main__":
    main()
