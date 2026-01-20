#!/usr/bin/env python3
"""
Legal-Lucy: Contract Analysis and Legal Document Processing

This persona demonstrates:
- Legal document analysis and summarization
- Risk assessment and compliance checking
- Contract term extraction
- Legal language simplification
- Due diligence workflows
"""

import json
import os
from datetime import datetime
from typing import Dict, List

import dspy
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class ContractAnalysis(dspy.Signature):
    """Analyze contracts for key terms, risks, and compliance issues."""

    contract_text: str = dspy.InputField(desc="Full contract or legal document text")
    analysis_focus: str = dspy.InputField(
        desc="Specific areas to focus on", default="general"
    )

    document_type: str = dspy.OutputField(desc="Type of legal document")
    parties_involved: str = dspy.OutputField(desc="Key parties and their roles")
    key_terms: str = dspy.OutputField(desc="Important terms and conditions")
    payment_terms: str = dspy.OutputField(desc="Payment obligations and schedules")
    termination_clauses: str = dspy.OutputField(desc="Contract termination conditions")
    liability_provisions: str = dspy.OutputField(
        desc="Liability and indemnification terms"
    )


class RiskAssessment(dspy.Signature):
    """Assess legal and business risks in contracts and agreements."""

    contract_analysis: str = dspy.InputField(desc="Previous contract analysis results")
    business_context: str = dspy.InputField(desc="Business context and objectives")

    risk_level: str = dspy.OutputField(
        desc="Overall risk level: low, medium, high, critical"
    )
    primary_risks: str = dspy.OutputField(desc="Main risk factors identified")
    compliance_issues: str = dspy.OutputField(desc="Regulatory or compliance concerns")
    mitigation_strategies: str = dspy.OutputField(
        desc="Recommended risk mitigation approaches"
    )
    red_flags: str = dspy.OutputField(
        desc="Critical issues requiring immediate attention"
    )


class LegalSummary(dspy.Signature):
    """Create plain-English summaries of complex legal documents."""

    legal_document: str = dspy.InputField(desc="Complex legal text")
    target_audience: str = dspy.InputField(
        desc="Intended audience level", default="business professionals"
    )

    executive_summary: str = dspy.OutputField(desc="High-level summary for executives")
    key_obligations: str = dspy.OutputField(desc="Main obligations in plain English")
    important_dates: str = dspy.OutputField(desc="Critical dates and deadlines")
    action_items: str = dspy.OutputField(desc="Required actions and next steps")


class ComplianceChecker(dspy.Signature):
    """Check documents against regulatory requirements and standards."""

    document_content: str = dspy.InputField(desc="Document to check for compliance")
    regulation_type: str = dspy.InputField(desc="Type of regulation (GDPR, CCPA, etc.)")
    jurisdiction: str = dspy.InputField(desc="Legal jurisdiction", default="US")

    compliance_status: str = dspy.OutputField(desc="Overall compliance status")
    violations_found: str = dspy.OutputField(desc="Specific compliance violations")
    required_changes: str = dspy.OutputField(desc="Changes needed for compliance")
    compliance_score: int = dspy.OutputField(desc="Compliance score from 0-100")


class LegalLucy(dspy.Module):
    """Comprehensive legal document analysis and processing system."""

    def __init__(self):
        super().__init__()
        self.contract_analyzer = dspy.ChainOfThought(ContractAnalysis)
        self.risk_assessor = dspy.ChainOfThought(RiskAssessment)
        self.summarizer = dspy.ChainOfThought(LegalSummary)
        self.compliance_checker = dspy.ChainOfThought(ComplianceChecker)

        # Legal document templates and patterns
        self.document_patterns = self._load_legal_patterns()

    def _load_legal_patterns(self) -> Dict[str, List[str]]:
        """Load common legal document patterns and clauses."""
        return {
            "red_flag_terms": [
                "unlimited liability",
                "personal guarantee",
                "automatic renewal",
                "non-compete",
                "exclusivity",
                "penalty",
                "liquidated damages",
            ],
            "important_sections": [
                "termination",
                "liability",
                "indemnification",
                "payment",
                "confidentiality",
                "intellectual property",
                "governing law",
            ],
            "compliance_keywords": [
                "GDPR",
                "CCPA",
                "HIPAA",
                "SOX",
                "data protection",
                "privacy policy",
                "consent",
                "data subject rights",
            ],
        }

    def analyze_contract(
        self,
        contract_text: str,
        business_context: str = "",
        focus_areas: str = "general",
    ) -> Dict:
        """Perform comprehensive contract analysis."""

        # Step 1: Basic contract analysis
        analysis = self.contract_analyzer(
            contract_text=contract_text, analysis_focus=focus_areas
        )

        # Step 2: Risk assessment
        risk_assessment = self.risk_assessor(
            contract_analysis=json.dumps(
                {
                    "document_type": analysis.document_type,
                    "key_terms": analysis.key_terms,
                    "payment_terms": analysis.payment_terms,
                    "termination_clauses": analysis.termination_clauses,
                    "liability_provisions": analysis.liability_provisions,
                }
            ),
            business_context=business_context or "Standard business operations",
        )

        # Step 3: Plain English summary
        summary = self.summarizer(
            legal_document=contract_text, target_audience="business professionals"
        )

        # Step 4: Basic compliance check (GDPR focus)
        compliance = self.compliance_checker(
            document_content=contract_text,
            regulation_type="GDPR and general data protection",
            jurisdiction="US/EU",
        )

        return {
            "analysis_id": f"LA-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            "analysis": {
                "document_type": analysis.document_type,
                "parties_involved": analysis.parties_involved,
                "key_terms": analysis.key_terms,
                "payment_terms": analysis.payment_terms,
                "termination_clauses": analysis.termination_clauses,
                "liability_provisions": analysis.liability_provisions,
            },
            "risk_assessment": {
                "risk_level": risk_assessment.risk_level,
                "primary_risks": risk_assessment.primary_risks,
                "compliance_issues": risk_assessment.compliance_issues,
                "mitigation_strategies": risk_assessment.mitigation_strategies,
                "red_flags": risk_assessment.red_flags,
            },
            "summary": {
                "executive_summary": summary.executive_summary,
                "key_obligations": summary.key_obligations,
                "important_dates": summary.important_dates,
                "action_items": summary.action_items,
            },
            "compliance": {
                "status": compliance.compliance_status,
                "violations": compliance.violations_found,
                "required_changes": compliance.required_changes,
                "score": compliance.compliance_score,
            },
            "processed_at": datetime.now().isoformat(),
        }

    def batch_document_review(self, documents: List[Dict]) -> List[Dict]:
        """Process multiple legal documents efficiently."""
        results = []

        for doc in documents:
            try:
                result = self.analyze_contract(
                    contract_text=doc["content"],
                    business_context=doc.get("context", ""),
                    focus_areas=doc.get("focus", "general"),
                )
                result["document_name"] = doc.get("name", "Unknown")
                results.append(result)
            except Exception as e:
                results.append(
                    {
                        "document_name": doc.get("name", "Unknown"),
                        "error": str(e),
                        "status": "failed",
                    }
                )

        return results


def get_sample_legal_documents():
    """Get sample legal documents for testing."""
    return [
        {
            "name": "Software License Agreement",
            "content": (
                "SOFTWARE LICENSE AGREEMENT\n\n"
                'This Software License Agreement ("Agreement") is entered into '
                "on [DATE] between TechCorp Inc., a Delaware corporation "
                '("Licensor") and [CLIENT NAME] ("Licensee").\n\n'
                "1. GRANT OF LICENSE\n"
                "Subject to the terms and conditions of this Agreement, Licensor "
                "hereby grants to Licensee a non-exclusive, non-transferable "
                "license to use the licensed software.\n\n"
                "2. PAYMENT TERMS\n"
                "Licensee agrees to pay annual license fees of $50,000 within 30 "
                "days of invoice date. Late payments will incur a 2% monthly "
                "penalty. All fees are non-refundable.\n\n"
                "3. TERMINATION\n"
                "This Agreement may be terminated by either party with 90 days "
                "written notice. Upon termination, Licensee must immediately "
                "cease use and return all software copies.\n\n"
                "4. LIABILITY AND INDEMNIFICATION\n"
                "LICENSOR'S LIABILITY IS LIMITED TO THE AMOUNT PAID UNDER THIS "
                "AGREEMENT. Licensee agrees to indemnify and hold harmless "
                "Licensor from all claims arising from Licensee's use of the "
                "software.\n\n"
                "5. CONFIDENTIALITY\n"
                "Both parties agree to maintain confidentiality of proprietary "
                "information for 5 years.\n\n"
                "6. GOVERNING LAW\n"
                "This Agreement is governed by Delaware state law. Disputes will "
                "be resolved through binding arbitration in Delaware."
            ),
            "context": "Enterprise software licensing for internal business use",
            "focus": "payment terms and liability",
        },
        {
            "name": "Service Provider Agreement",
            "content": (
                "PROFESSIONAL SERVICES AGREEMENT\n\n"
                "This Professional Services Agreement is between ServicePro LLC "
                '("Provider") and Client Corporation ("Client") for consulting '
                "services.\n\n"
                "1. SCOPE OF SERVICES\n"
                "Provider will deliver business consulting services including "
                "strategy development, process optimization, and implementation "
                "support as detailed in Exhibit A.\n\n"
                "2. COMPENSATION\n"
                "Client will pay $200 per hour for consulting services. Monthly "
                "invoices due within 15 days. Provider may increase rates with 60 "
                "days notice.\n\n"
                "3. INTELLECTUAL PROPERTY\n"
                "All work product and deliverables become the exclusive property "
                "of Client. Provider retains rights to general methodologies and "
                "pre-existing IP.\n\n"
                "4. NON-COMPETE AND NON-SOLICITATION\n"
                "Provider agrees not to provide similar services to Client's "
                "direct competitors for 12 months after agreement termination. "
                "Provider will not solicit Client employees for 18 months.\n\n"
                "5. LIMITATION OF LIABILITY\n"
                "Provider's liability is limited to the fees paid in the 12 "
                "months preceding the claim. Provider is not liable for "
                "consequential or punitive damages.\n\n"
                "6. DATA PROTECTION\n"
                "Provider will implement reasonable security measures to protect "
                "Client data. Provider may not transfer data outside the US "
                "without written consent."
            ),
            "context": "Professional consulting services for business transformation",
            "focus": "IP rights and non-compete clauses",
        },
        {
            "name": "Vendor Agreement",
            "content": (
                "VENDOR SUPPLY AGREEMENT\n\n"
                'This Supply Agreement is between Manufacturing Corp ("Buyer") '
                'and SupplyCo ("Vendor") for the supply of industrial components.\n\n'
                "1. SUPPLY OBLIGATIONS\n"
                "Vendor agrees to supply components according to specifications in "
                "Schedule A. Minimum order quantities apply. Vendor guarantees 99% "
                "on-time delivery.\n\n"
                "2. PRICING AND PAYMENT\n"
                "Prices are fixed for 24 months. Payment terms are Net 45 days. "
                "Volume discounts apply for orders exceeding $100,000 quarterly.\n\n"
                "3. QUALITY ASSURANCE\n"
                "All components must meet ISO 9001 standards. Defective products "
                "will be replaced at Vendor's expense within 48 hours.\n\n"
                "4. FORCE MAJEURE\n"
                "Neither party is liable for delays due to acts of God, government "
                "actions, or other circumstances beyond reasonable control.\n\n"
                "5. TERMINATION\n"
                "Agreement may be terminated for cause with 30 days notice or "
                "without cause with 180 days notice. Buyer may terminate "
                "immediately for quality failures.\n\n"
                "6. DISPUTE RESOLUTION\n"
                "All disputes will be resolved through mediation, then binding "
                "arbitration under AAA Commercial Rules."
            ),
            "context": "Manufacturing supply chain partnership",
            "focus": "quality assurance and termination rights",
        },
    ]


def main():
    """Run Legal-Lucy contract analysis examples."""
    print("⚖️  Legal-Lucy: AI Legal Document Analyst")
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

    # Create Legal-Lucy agent
    lucy = LegalLucy()

    # Analyze sample documents
    sample_docs = get_sample_legal_documents()

    print(f"\n📄 Analyzing {len(sample_docs)} Legal Documents:")
    print("-" * 60)

    results = lucy.batch_document_review(sample_docs)

    # Process results
    risk_levels = {"low": 0, "medium": 0, "high": 0, "critical": 0}
    compliance_scores = []

    for i, result in enumerate(results, 1):
        if "error" in result:
            print(
                f"\n❌ Document {i}: {result['document_name']} - Error: "
                f"{result['error']}"
            )
            continue

        doc_name = result["document_name"]
        analysis = result["analysis"]
        risks = result["risk_assessment"]
        summary = result["summary"]
        compliance = result["compliance"]

        print(f"\n📋 Document {i}: {doc_name}")

        # Document analysis
        print("   📑 Analysis:")
        print(f"      Type: {analysis['document_type']}")
        print(f"      Parties: {analysis['parties_involved'][:100]}...")

        # Risk assessment
        print("   ⚠️  Risk Assessment:")
        print(f"      Level: {risks['risk_level'].upper()}")
        print(f"      Primary Risks: {risks['primary_risks'][:100]}...")
        if risks["red_flags"]:
            print(f"      🚨 Red Flags: {risks['red_flags'][:100]}...")

        # Executive summary
        print("   📊 Executive Summary:")
        print(f"      {summary['executive_summary'][:150]}...")

        # Compliance
        print("   ✅ Compliance:")
        print(f"      Status: {compliance['status']}")
        try:
            score = int(compliance["score"])
            print(f"      Score: {score}/100")
            compliance_scores.append(score)
        except (TypeError, ValueError):
            print("      Score: Not available")

        # Track risk levels
        risk_level = risks["risk_level"].lower()
        if risk_level in risk_levels:
            risk_levels[risk_level] += 1

    # Summary statistics
    print("\n📊 Legal Analysis Summary:")
    print("-" * 40)
    print(f"   Documents Processed: {len([r for r in results if 'error' not in r])}")
    print(f"   Processing Errors: {len([r for r in results if 'error' in r])}")

    print("\n   ⚠️  Risk Level Distribution:")
    for level, count in risk_levels.items():
        if count > 0:
            print(f"      {level.title()}: {count} documents")

    if compliance_scores:
        avg_compliance = sum(compliance_scores) / len(compliance_scores)
        print("\n   ✅ Compliance Summary:")
        print(f"      Average Compliance Score: {avg_compliance:.1f}/100")
        print(f"      Highest Score: {max(compliance_scores)}/100")
        print(f"      Lowest Score: {min(compliance_scores)}/100")

    print("\n🎯 Key Insights:")
    print("   ✅ Automated contract term extraction")
    print("   ✅ Risk assessment and red flag detection")
    print("   ✅ Plain English summaries for stakeholders")
    print("   ✅ Compliance checking against regulations")
    print("   ✅ Batch processing for due diligence")

    print("\n🚀 Production Enhancements:")
    print("   - Integration with document management systems")
    print("   - Custom risk scoring models by industry")
    print("   - Multi-jurisdiction compliance checking")
    print("   - Automated contract comparison and redlining")
    print("   - Legal precedent and case law integration")
    print("   - Workflow automation for legal review processes")


if __name__ == "__main__":
    main()
