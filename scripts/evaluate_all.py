#!/usr/bin/env python3
"""
Comprehensive Evaluation Runner for DSPy 0-to-1 Guide

This script runs evaluations across all examples and generates reports.
"""

import os
import sys
import time
import json
from pathlib import Path
from typing import Dict, List, Any
import subprocess
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
import dspy
from dspy import Example
from dspy.evaluate import Evaluate, metrics

# Load environment variables
load_dotenv()

class EvaluationRunner:
    """Runs comprehensive evaluations across all examples."""

    def __init__(self):
        self.results = {}
        self.start_time = time.time()

        # Configure DSPy
        self._configure_dspy()

    def _configure_dspy(self):
        """Configure DSPy with available model."""
        try:
            if os.getenv('OPENAI_API_KEY'):
                dspy.configure(lm=dspy.LM('openai/gpt-4o-mini'))
                self.model_name = "OpenAI GPT-4o-mini"
            else:
                dspy.configure(lm=dspy.LM('ollama_chat/llama3'))
                self.model_name = "Ollama Llama3"
        except Exception as e:
            raise RuntimeError(f"Failed to configure DSPy: {e}")

    def run_basic_examples(self) -> Dict:
        """Run and evaluate basic examples."""
        print("📚 Evaluating Basic Examples")
        print("-" * 40)

        basic_results = {}

        # Test hello world
        try:
            result = subprocess.run([
                sys.executable, "examples/basic/hello_world.py"
            ], capture_output=True, text=True, timeout=60)

            basic_results["hello_world"] = {
                "success": result.returncode == 0,
                "execution_time": "< 60s",
                "output_length": len(result.stdout) if result.returncode == 0 else 0,
                "error": result.stderr if result.returncode != 0 else None
            }
            print(f"   ✅ Hello World: {'✅ Pass' if result.returncode == 0 else '❌ Fail'}")

        except subprocess.TimeoutExpired:
            basic_results["hello_world"] = {"success": False, "error": "Timeout"}
            print("   ❌ Hello World: Timeout")
        except Exception as e:
            basic_results["hello_world"] = {"success": False, "error": str(e)}
            print(f"   ❌ Hello World: {e}")

        # Test math QA
        try:
            result = subprocess.run([
                sys.executable, "examples/basic/math_qa.py"
            ], capture_output=True, text=True, timeout=120)

            # Parse accuracy from output
            accuracy = self._extract_accuracy_from_output(result.stdout)

            basic_results["math_qa"] = {
                "success": result.returncode == 0,
                "accuracy": accuracy,
                "execution_time": "< 120s",
                "error": result.stderr if result.returncode != 0 else None
            }
            print(f"   📊 Math QA: {'✅ Pass' if result.returncode == 0 else '❌ Fail'} (Accuracy: {accuracy}%)")

        except subprocess.TimeoutExpired:
            basic_results["math_qa"] = {"success": False, "error": "Timeout"}
            print("   ❌ Math QA: Timeout")
        except Exception as e:
            basic_results["math_qa"] = {"success": False, "error": str(e)}
            print(f"   ❌ Math QA: {e}")

        # Test summarizer
        try:
            result = subprocess.run([
                sys.executable, "examples/basic/summarizer.py"
            ], capture_output=True, text=True, timeout=180)

            basic_results["summarizer"] = {
                "success": result.returncode == 0,
                "execution_time": "< 180s",
                "documents_processed": self._count_documents_processed(result.stdout),
                "error": result.stderr if result.returncode != 0 else None
            }
            print(f"   📄 Summarizer: {'✅ Pass' if result.returncode == 0 else '❌ Fail'}")

        except subprocess.TimeoutExpired:
            basic_results["summarizer"] = {"success": False, "error": "Timeout"}
            print("   ❌ Summarizer: Timeout")
        except Exception as e:
            basic_results["summarizer"] = {"success": False, "error": str(e)}
            print(f"   ❌ Summarizer: {e}")

        return basic_results

    def run_persona_examples(self) -> Dict:
        """Run and evaluate persona examples."""
        print("\n🎭 Evaluating Persona Examples")
        print("-" * 40)

        persona_results = {}

        # Test Support-Sam
        try:
            result = subprocess.run([
                sys.executable, "examples/personas/support_sam.py"
            ], capture_output=True, text=True, timeout=300)

            tickets_processed = self._count_tickets_processed(result.stdout)
            avg_satisfaction = self._extract_satisfaction_score(result.stdout)

            persona_results["support_sam"] = {
                "success": result.returncode == 0,
                "tickets_processed": tickets_processed,
                "avg_satisfaction": avg_satisfaction,
                "execution_time": "< 300s",
                "error": result.stderr if result.returncode != 0 else None
            }
            print(f"   🎧 Support-Sam: {'✅ Pass' if result.returncode == 0 else '❌ Fail'}")
            print(f"      Tickets: {tickets_processed}, Satisfaction: {avg_satisfaction}/10")

        except subprocess.TimeoutExpired:
            persona_results["support_sam"] = {"success": False, "error": "Timeout"}
            print("   ❌ Support-Sam: Timeout")
        except Exception as e:
            persona_results["support_sam"] = {"success": False, "error": str(e)}
            print(f"   ❌ Support-Sam: {e}")

        # Test Legal-Lucy (with shorter timeout)
        try:
            result = subprocess.run([
                sys.executable, "examples/personas/legal_lucy.py"
            ], capture_output=True, text=True, timeout=180)

            documents_analyzed = self._count_legal_documents(result.stdout)

            persona_results["legal_lucy"] = {
                "success": result.returncode == 0,
                "documents_analyzed": documents_analyzed,
                "execution_time": "< 180s",
                "error": result.stderr if result.returncode != 0 else None
            }
            print(f"   ⚖️  Legal-Lucy: {'✅ Pass' if result.returncode == 0 else '❌ Fail'}")
            print(f"      Documents: {documents_analyzed}")

        except subprocess.TimeoutExpired:
            persona_results["legal_lucy"] = {"success": False, "error": "Timeout (expected for complex analysis)"}
            print("   ⚠️  Legal-Lucy: Timeout (expected for complex documents)")
        except Exception as e:
            persona_results["legal_lucy"] = {"success": False, "error": str(e)}
            print(f"   ❌ Legal-Lucy: {e}")

        return persona_results

    def run_advanced_examples(self) -> Dict:
        """Run and evaluate advanced examples."""
        print("\n🚀 Evaluating Advanced Examples")
        print("-" * 40)

        advanced_results = {}

        # Test Pydantic validation
        try:
            result = subprocess.run([
                sys.executable, "examples/advanced/pydantic_validation.py"
            ], capture_output=True, text=True, timeout=120)

            contracts_analyzed = self._count_contracts_analyzed(result.stdout)

            advanced_results["pydantic_validation"] = {
                "success": result.returncode == 0,
                "contracts_analyzed": contracts_analyzed,
                "structured_output": "✅ Valid" if "Output structure is valid" in result.stdout else "❌ Invalid",
                "execution_time": "< 120s",
                "error": result.stderr if result.returncode != 0 else None
            }
            print(f"   🔧 Pydantic Validation: {'✅ Pass' if result.returncode == 0 else '❌ Fail'}")

        except subprocess.TimeoutExpired:
            advanced_results["pydantic_validation"] = {"success": False, "error": "Timeout"}
            print("   ❌ Pydantic Validation: Timeout")
        except Exception as e:
            advanced_results["pydantic_validation"] = {"success": False, "error": str(e)}
            print(f"   ❌ Pydantic Validation: {e}")

        return advanced_results

    def _extract_accuracy_from_output(self, output: str) -> float:
        """Extract accuracy percentage from math QA output."""
        try:
            lines = output.split('\n')
            for line in lines:
                if "Correct:" in line and "/" in line:
                    # Extract "Correct: 2/6 (33.3%)"
                    parts = line.split('(')
                    if len(parts) > 1:
                        percent = parts[1].split('%')[0]
                        return float(percent)
            return 0.0
        except:
            return 0.0

    def _count_documents_processed(self, output: str) -> int:
        """Count documents processed in summarizer output."""
        return output.count("📑 Document:")

    def _count_tickets_processed(self, output: str) -> int:
        """Count tickets processed in Support-Sam output."""
        return output.count("🎫 Ticket #")

    def _extract_satisfaction_score(self, output: str) -> float:
        """Extract average satisfaction score."""
        try:
            lines = output.split('\n')
            for line in lines:
                if "Average Predicted Satisfaction:" in line:
                    score = line.split(':')[-1].strip().split('/')[0]
                    return float(score)
            return 0.0
        except:
            return 0.0

    def _count_legal_documents(self, output: str) -> int:
        """Count legal documents analyzed."""
        return output.count("📋 Document")

    def _count_contracts_analyzed(self, output: str) -> int:
        """Count contracts analyzed in Pydantic example."""
        return output.count("🔍 Contract")

    def generate_report(self) -> Dict:
        """Generate comprehensive evaluation report."""
        total_time = time.time() - self.start_time

        # Count successes and failures
        all_results = {}
        all_results.update(self.results.get('basic', {}))
        all_results.update(self.results.get('persona', {}))
        all_results.update(self.results.get('advanced', {}))

        successful = sum(1 for r in all_results.values() if r.get('success', False))
        total = len(all_results)

        report = {
            "evaluation_summary": {
                "timestamp": datetime.now().isoformat(),
                "model_used": self.model_name,
                "total_examples": total,
                "successful": successful,
                "failed": total - successful,
                "success_rate": f"{successful/total*100:.1f}%" if total > 0 else "0%",
                "total_execution_time": f"{total_time:.1f}s"
            },
            "detailed_results": self.results,
            "recommendations": self._generate_recommendations()
        }

        return report

    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on evaluation results."""
        recommendations = []

        # Basic recommendations
        basic_results = self.results.get('basic', {})
        if any(not r.get('success', False) for r in basic_results.values()):
            recommendations.append("Some basic examples failed - check model configuration and dependencies")

        # Math QA accuracy
        math_result = basic_results.get('math_qa', {})
        if math_result.get('accuracy', 0) < 50:
            recommendations.append("Math QA accuracy is low - consider using a more capable model or adding optimization")

        # Timeout issues
        all_results = {}
        all_results.update(self.results.get('basic', {}))
        all_results.update(self.results.get('persona', {}))
        all_results.update(self.results.get('advanced', {}))

        timeouts = sum(1 for r in all_results.values() if r.get('error') == 'Timeout')
        if timeouts > 0:
            recommendations.append(f"{timeouts} examples timed out - consider increasing timeout limits or using faster models")

        return recommendations

def main():
    """Run comprehensive evaluation."""
    print("🧪 DSPy 0-to-1 Guide Comprehensive Evaluation")
    print("=" * 60)
    print(f"📅 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    try:
        runner = EvaluationRunner()
        print(f"🤖 Using model: {runner.model_name}")

        # Run evaluations
        runner.results['basic'] = runner.run_basic_examples()
        runner.results['persona'] = runner.run_persona_examples()
        runner.results['advanced'] = runner.run_advanced_examples()

        # Generate report
        report = runner.generate_report()

        # Save report
        report_path = Path("evaluation_report.json")
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)

        # Print summary
        print("\n" + "="*60)
        print("📊 Evaluation Summary")
        print("-" * 30)
        summary = report['evaluation_summary']
        print(f"   Total Examples: {summary['total_examples']}")
        print(f"   Successful: {summary['successful']}")
        print(f"   Failed: {summary['failed']}")
        print(f"   Success Rate: {summary['success_rate']}")
        print(f"   Total Time: {summary['total_execution_time']}")

        if report['recommendations']:
            print(f"\n💡 Recommendations:")
            for rec in report['recommendations']:
                print(f"   - {rec}")

        print(f"\n📄 Detailed report saved to: {report_path}")

        if summary['success_rate'].rstrip('%') == '100.0':
            print("\n🎉 All examples passed! The DSPy 0-to-1 Guide is working perfectly.")
        else:
            print(f"\n⚠️  Some examples failed. Check the detailed report for troubleshooting guidance.")

    except KeyboardInterrupt:
        print("\n⚠️  Evaluation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Evaluation failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()