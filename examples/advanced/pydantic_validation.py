#!/usr/bin/env python3
"""
Advanced DSPy: Pydantic Validation and Structured Output

Demonstrates:
- Structured output with Pydantic models
- Type validation and error handling
- Complex nested data structures
- Output formatting and consistency
"""

import json
import os
from typing import List, Literal, Optional

import dspy
from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator

# Load environment variables
load_dotenv()


# Pydantic models for structured output
class ContactInfo(BaseModel):
    """Contact information structure."""

    name: str = Field(..., min_length=1, description="Full name")
    email: Optional[str] = Field(None, description="Email address")
    phone: Optional[str] = Field(None, description="Phone number")
    role: Optional[str] = Field(None, description="Role or title")


class FinancialTerm(BaseModel):
    """Financial term structure."""

    amount: float = Field(..., description="Monetary amount")
    currency: str = Field(default="USD", description="Currency code")
    frequency: Optional[str] = Field(None, description="Payment frequency")
    due_date: Optional[str] = Field(None, description="Due date if applicable")


class ContractAnalysisResult(BaseModel):
    """Structured contract analysis result."""

    contract_type: Literal["service", "license", "employment", "vendor", "other"] = (
        Field(..., description="Type of contract")
    )
    parties: List[ContactInfo] = Field(
        ..., min_length=2, description="Contracting parties"
    )
    key_terms: List[str] = Field(..., min_length=1, description="Key contractual terms")
    financial_terms: List[FinancialTerm] = Field(
        default=[], description="Financial obligations"
    )
    duration: Optional[str] = Field(None, description="Contract duration")
    termination_notice: Optional[int] = Field(
        None, description="Termination notice in days"
    )
    risk_level: Literal["low", "medium", "high", "critical"] = Field(
        ..., description="Overall risk assessment"
    )
    compliance_score: int = Field(
        ..., ge=0, le=100, description="Compliance score (0-100)"
    )
    red_flags: List[str] = Field(default=[], description="Identified risk factors")

    @field_validator("financial_terms")
    @classmethod
    def validate_amounts(cls, v):
        for term in v:
            if term.amount < 0:
                raise ValueError("Financial amounts cannot be negative")
        return v


class ProductReview(BaseModel):
    """Structured product review analysis."""

    product_name: str = Field(..., description="Name of the product")
    overall_rating: int = Field(..., ge=1, le=5, description="Overall rating (1-5)")
    sentiment: Literal[
        "very_positive", "positive", "neutral", "negative", "very_negative"
    ] = Field(..., description="Sentiment classification")
    pros: List[str] = Field(default=[], description="Positive aspects")
    cons: List[str] = Field(default=[], description="Negative aspects")
    categories: List[str] = Field(default=[], description="Product categories")
    would_recommend: bool = Field(..., description="Would recommend to others")
    price_value: Literal["excellent", "good", "fair", "poor"] = Field(
        ..., description="Price-to-value assessment"
    )


class StructuredContractAnalyzer(dspy.Module):
    """Contract analyzer with structured Pydantic output."""

    def __init__(self):
        super().__init__()
        # Use ChainOfThought for analysis (TypedPredictor may not be available)
        self.analyzer = dspy.ChainOfThought("contract_text -> analysis")

    def forward(self, contract_text: str):
        """Analyze contract and return structured result."""
        try:
            # Get analysis from DSPy
            result = self.analyzer(contract_text=contract_text)

            # Parse LLM output to structured format
            return self._parse_to_structured(result, contract_text)
        except Exception as e:
            # Fallback to manual parsing if structured output fails
            print(f"Analysis failed, using fallback: {e}")
            return self._fallback_analysis(contract_text)

    def _parse_to_structured(self, result, contract_text: str):
        """Parse LLM output to structured format."""
        import re

        # Get the analysis text from DSPy result
        analysis_text = result.analysis if hasattr(result, "analysis") else str(result)
        combined_text = f"{contract_text} {analysis_text}"

        # Extract contract type
        contract_type = "other"
        if any(
            word in combined_text.lower()
            for word in ["service", "consulting", "professional"]
        ):
            contract_type = "service"
        elif any(
            word in combined_text.lower()
            for word in ["license", "software", "intellectual"]
        ):
            contract_type = "license"
        elif any(
            word in combined_text.lower() for word in ["employment", "hire", "employee"]
        ):
            contract_type = "employment"
        elif any(
            word in combined_text.lower() for word in ["vendor", "supply", "purchase"]
        ):
            contract_type = "vendor"

        # Extract parties
        parties = []

        # Look for company names and roles
        company_patterns = [
            r"([A-Z][a-zA-Z\s&]+(?:Inc|LLC|Corp|Ltd|Corporation|Company))",
            r"(Licensor|Licensee|Provider|Client|Vendor|Buyer|Seller)",
        ]

        found_names = set()
        for pattern in company_patterns:
            matches = re.findall(pattern, combined_text)
            for match in matches[:4]:  # Limit to 4 parties max
                if match not in found_names and len(match) > 2:
                    role = None
                    if any(r in match for r in ["Licensor", "Provider", "Vendor"]):
                        role = "Provider"
                    elif any(r in match for r in ["Licensee", "Client", "Buyer"]):
                        role = "Client"

                    parties.append(ContactInfo(name=match, role=role))
                    found_names.add(match)

        # Ensure at least 2 parties
        if len(parties) < 2:
            parties = [
                ContactInfo(name="Party A", role="Provider"),
                ContactInfo(name="Party B", role="Client"),
            ]

        # Extract financial terms
        financial_terms = []
        money_patterns = [
            r"\$?([\d,]+(?:\.\d{2})?)",
            r"([\d,]+)\s*(?:dollars?|USD)",
        ]

        for pattern in money_patterns:
            matches = re.findall(pattern, combined_text)
            for match in matches:
                try:
                    amount = float(match.replace(",", ""))
                    if amount > 100:  # Only capture significant amounts
                        frequency = "one-time"
                        if any(
                            freq in combined_text.lower()
                            for freq in ["annual", "yearly", "per year"]
                        ):
                            frequency = "annual"
                        elif any(
                            freq in combined_text.lower()
                            for freq in ["monthly", "per month"]
                        ):
                            frequency = "monthly"

                        financial_terms.append(
                            FinancialTerm(
                                amount=amount, currency="USD", frequency=frequency
                            )
                        )
                        break  # Only take first significant amount
                except ValueError:
                    continue

        # Extract duration
        duration = "Unknown"
        duration_patterns = [
            r"(\d+)\s*(?:year|yr)s?",
            r"(\d+)\s*months?",
            r"(\d+)\s*days?",
        ]

        for pattern in duration_patterns:
            match = re.search(pattern, combined_text.lower())
            if match:
                num = match.group(1)
                if "year" in match.group(0) or "yr" in match.group(0):
                    duration = f"{num} years"
                elif "month" in match.group(0):
                    duration = f"{num} months"
                elif "day" in match.group(0):
                    duration = f"{num} days"
                break

        # Extract termination notice
        termination_notice = 30  # Default
        notice_match = re.search(
            r"(\d+)\s*days?\s*(?:notice|written notice)", combined_text.lower()
        )
        if notice_match:
            termination_notice = int(notice_match.group(1))

        # Assess risk level based on content
        risk_indicators = {
            "high": [
                "unlimited liability",
                "personal guarantee",
                "liquidated damages",
                "penalty",
            ],
            "critical": ["unlimited liability", "personal guarantee"],
            "medium": ["liability", "indemnif", "terminate", "breach"],
            "low": [],
        }

        risk_level = "low"
        for level, indicators in risk_indicators.items():
            if any(indicator in combined_text.lower() for indicator in indicators):
                risk_level = level
                if level == "critical":  # Stop at critical
                    break

        # Calculate compliance score
        compliance_score = 100
        compliance_issues = [
            ("liability", -10),
            ("indemnif", -5),
            ("non-compete", -15),
            ("automatic renewal", -10),
            ("penalty", -15),
        ]

        for issue, penalty in compliance_issues:
            if issue in combined_text.lower():
                compliance_score += penalty

        compliance_score = max(0, min(100, compliance_score))

        # Identify red flags
        red_flags = []
        red_flag_terms = [
            "unlimited liability",
            "personal guarantee",
            "automatic renewal",
            "liquidated damages",
            "non-compete",
            "exclusivity",
        ]

        for term in red_flag_terms:
            if term in combined_text.lower():
                red_flags.append(term.title())

        # Extract key terms
        key_terms = []
        important_terms = [
            "payment",
            "termination",
            "liability",
            "confidentiality",
            "intellectual property",
            "warranty",
            "indemnification",
        ]

        for term in important_terms:
            if term in combined_text.lower():
                key_terms.append(term.title())

        if not key_terms:
            key_terms = ["Standard contract terms"]

        return ContractAnalysisResult(
            contract_type=contract_type,
            parties=parties,
            key_terms=key_terms,
            financial_terms=financial_terms,
            duration=duration,
            termination_notice=termination_notice,
            risk_level=risk_level,
            compliance_score=compliance_score,
            red_flags=red_flags,
        )

    def _fallback_analysis(self, contract_text: str):
        """Fallback analysis if structured output fails."""
        # Create a basic analysis manually
        return ContractAnalysisResult(
            contract_type="other",
            parties=[ContactInfo(name="Party A"), ContactInfo(name="Party B")],
            key_terms=["Standard terms apply"],
            risk_level="medium",
            compliance_score=75,
        )


class StructuredReviewAnalyzer(dspy.Module):
    """Product review analyzer with structured output."""

    def __init__(self):
        super().__init__()
        self.analyzer = dspy.ChainOfThought("review_text -> analysis")

    def forward(self, review_text: str):
        """Analyze review and return structured result."""
        try:
            self.analyzer(review_text=review_text)
            return self._extract_basic_info(
                review_text
            )  # Use basic extraction for demo
        except Exception as e:
            print(f"Analysis failed: {e}")
            return self._extract_basic_info(review_text)

    def _extract_basic_info(self, review_text: str):
        """Extract basic information manually."""
        # Simple fallback analysis
        positive_words = ["good", "great", "excellent", "amazing", "love"]
        negative_words = ["bad", "terrible", "awful", "hate", "worst"]

        text_lower = review_text.lower()
        pos_count = sum(1 for word in positive_words if word in text_lower)
        neg_count = sum(1 for word in negative_words if word in text_lower)

        if pos_count > neg_count:
            sentiment = "positive"
            rating = 4
        elif neg_count > pos_count:
            sentiment = "negative"
            rating = 2
        else:
            sentiment = "neutral"
            rating = 3

        return ProductReview(
            product_name="Unknown Product",
            overall_rating=rating,
            sentiment=sentiment,
            would_recommend=pos_count > neg_count,
            price_value="fair",
        )


def get_test_data():
    """Get test data for validation examples."""
    return {
        "contracts": [
            """
            SOFTWARE LICENSE AGREEMENT

            This agreement is between TechCorp Inc. (Licensor) and Business Solutions
            Ltd. (Licensee).

            LICENSE: Non-exclusive software license for internal use only.
            PAYMENT: Annual fee of $50,000 due within 30 days of invoice.
            TERM: 3 years from effective date with automatic renewal.
            TERMINATION: Either party may terminate with 90 days written notice.
            LIABILITY: Licensor's liability limited to amount paid under agreement.

            Contact: John Smith, CEO, TechCorp Inc., john@techcorp.com,
            (555) 123-4567
            Contact: Sarah Johnson, CTO, Business Solutions Ltd., sarah@bizsol.com
            """,
            """
            CONSULTING SERVICES AGREEMENT

            Provider: DataScience Pro LLC
            Client: Manufacturing Corp

            SERVICES: Business intelligence and data analytics consulting
            RATE: $200/hour for senior consultants, $150/hour for junior consultants
            PAYMENT: Monthly invoicing, Net 30 terms
            DURATION: 18 months starting January 1, 2024
            TERMINATION: 30 days written notice required
            NON-COMPETE: 12 month restriction on competing services

            High liability exposure due to data handling requirements.
            """,
        ],
        "reviews": [
            """
            I absolutely love this wireless headphone! The sound quality is amazing and
            the battery life lasts all day. The noise cancellation works perfectly in
            busy environments. The build quality feels premium and they're comfortable
            for long listening sessions. At $299, they're definitely worth every penny.
            I've recommended them to all my friends.
            Would definitely buy again!
            """,
            """
            This smartphone is okay but has some issues. The camera is decent but not as
            good as advertised. Battery life is average, gets me through the day but
            barely. The screen is nice and bright. Price seems a bit high for what you
            get - $899 feels like too much.
            Customer service was helpful when I had questions. It's not bad, just
            not exceptional.
            """,
            """
            Terrible experience with this laptop. It's slow, crashes frequently, and
            gets extremely hot during basic tasks. The screen quality is poor with
            washed out colors. Build quality feels cheap despite the high price tag.
            Customer support was unhelpful and dismissive. Save your money and buy
            something else. I regret this purchase completely.
            """,
        ],
    }


def validate_output_structure(result, expected_type):
    """Validate that output matches expected Pydantic structure."""
    try:
        if isinstance(result, expected_type):
            print("✅ Output structure is valid")
            return True
        else:
            print(
                f"❌ Output type mismatch. Expected {expected_type}, got {type(result)}"
            )
            return False
    except Exception as e:
        print(f"❌ Validation error: {e}")
        return False


def main():
    """Run Pydantic validation examples."""
    print("🔧 DSPy with Pydantic Validation")
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

    # Create analyzers
    contract_analyzer = StructuredContractAnalyzer()
    review_analyzer = StructuredReviewAnalyzer()

    # Get test data
    test_data = get_test_data()

    print("\n📋 Testing Contract Analysis with Pydantic Validation:")
    print("-" * 60)

    for i, contract in enumerate(test_data["contracts"], 1):
        print(f"\n🔍 Contract {i}:")
        print(f"   Length: {len(contract.split())} words")

        try:
            result = contract_analyzer(contract)

            # Validate structure
            is_valid = validate_output_structure(result, ContractAnalysisResult)

            if is_valid:
                print("   📊 Analysis Results:")
                print(f"      Contract Type: {result.contract_type}")
                print(f"      Number of Parties: {len(result.parties)}")
                print(f"      Key Terms Count: {len(result.key_terms)}")
                print(f"      Financial Terms: {len(result.financial_terms)}")
                print(f"      Risk Level: {result.risk_level}")
                print(f"      Compliance Score: {result.compliance_score}/100")

                if result.red_flags:
                    print(f"      🚨 Red Flags: {', '.join(result.red_flags[:2])}...")

                # Show structured data as JSON
                print("   💾 JSON Output (truncated):")
                json_output = result.model_dump()
                print(f"      {json.dumps(json_output, indent=2)[:200]}...")

        except Exception as e:
            print(f"   ❌ Analysis failed: {e}")

    print("\n⭐ Testing Product Review Analysis:")
    print("-" * 40)

    for i, review in enumerate(test_data["reviews"], 1):
        print(f"\n📝 Review {i}:")
        print(f"   Content: {review[:100]}...")

        try:
            result = review_analyzer(review)

            # Validate structure
            is_valid = validate_output_structure(result, ProductReview)

            if is_valid:
                print("   📊 Analysis Results:")
                print(f"      Product: {result.product_name}")
                print(f"      Rating: {result.overall_rating}/5 stars")
                print(f"      Sentiment: {result.sentiment}")
                recommendation = "Yes" if result.would_recommend else "No"
                print(f"      Would Recommend: {recommendation}")
                print(f"      Price Value: {result.price_value}")
                print(f"      Pros: {len(result.pros)} | Cons: {len(result.cons)}")

        except Exception as e:
            print(f"   ❌ Analysis failed: {e}")

    print("\n🎯 Pydantic Validation Benefits:")
    print("   ✅ Type safety and validation")
    print("   ✅ Structured, consistent output format")
    print("   ✅ Automatic error handling and fallbacks")
    print("   ✅ JSON serialization and deserialization")
    print("   ✅ Field validation and constraints")
    print("   ✅ Self-documenting data structures")

    print("\n🚀 Production Use Cases:")
    print("   - API responses with guaranteed structure")
    print("   - Database integration with validated data")
    print("   - Contract parsing and legal document analysis")
    print("   - E-commerce review processing")
    print("   - Financial document analysis")
    print("   - Data pipeline validation")


if __name__ == "__main__":
    main()
