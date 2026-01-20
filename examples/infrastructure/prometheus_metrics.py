#!/usr/bin/env python3
"""
DSPy Infrastructure Integration: Prometheus Metrics

Demonstrates:
- Prometheus metrics collection for DSPy applications
- Performance monitoring and observability
- Custom metrics for LLM operations
- HTTP server for metrics scraping
"""

import os
import threading
import time
from datetime import datetime
from typing import Any, Dict, Optional

import dspy
from dotenv import load_dotenv

# Prometheus client
from prometheus_client import (
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    Info,
    generate_latest,
    start_http_server,
)

# Load environment variables
load_dotenv()


class DSPyMetricsCollector:
    """Collects and exposes DSPy application metrics."""

    def __init__(self, registry: Optional[CollectorRegistry] = None):
        self.registry = registry or CollectorRegistry()

        # Define metrics
        self.llm_requests = Counter(
            "dspy_llm_requests_total",
            "Total number of LLM requests",
            ["model", "module_type", "status"],
            registry=self.registry,
        )

        self.llm_duration = Histogram(
            "dspy_llm_request_duration_seconds",
            "Duration of LLM requests",
            ["model", "module_type"],
            buckets=(0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0),
            registry=self.registry,
        )

        self.llm_tokens = Counter(
            "dspy_llm_tokens_total",
            "Total number of tokens processed",
            ["model", "module_type", "token_type"],  # token_type: input, output
            registry=self.registry,
        )

        self.active_requests = Gauge(
            "dspy_active_requests",
            "Number of active LLM requests",
            registry=self.registry,
        )

        self.optimization_runs = Counter(
            "dspy_optimization_runs_total",
            "Total optimization runs",
            ["optimizer_type", "status"],
            registry=self.registry,
        )

        self.optimization_duration = Histogram(
            "dspy_optimization_duration_seconds",
            "Duration of optimization runs",
            ["optimizer_type"],
            buckets=(30.0, 60.0, 300.0, 600.0, 1800.0, 3600.0),
            registry=self.registry,
        )

        self.performance_score = Gauge(
            "dspy_performance_score",
            "Performance score of DSPy modules",
            ["module_name", "metric_type"],
            registry=self.registry,
        )

        self.system_info = Info(
            "dspy_system_info", "System information", registry=self.registry
        )

        # Initialize system info
        self.system_info.info(
            {
                "version": "1.0.0",
                "python_version": os.sys.version.split()[0],
                "dspy_version": getattr(dspy, "__version__", "unknown"),
                "started_at": datetime.now().isoformat(),
            }
        )

    def record_llm_request(
        self,
        model: str,
        module_type: str,
        duration: float,
        status: str = "success",
        input_tokens: int = 0,
        output_tokens: int = 0,
    ):
        """Record an LLM request."""
        self.llm_requests.labels(
            model=model, module_type=module_type, status=status
        ).inc()
        self.llm_duration.labels(model=model, module_type=module_type).observe(duration)

        if input_tokens > 0:
            self.llm_tokens.labels(
                model=model, module_type=module_type, token_type="input"
            ).inc(input_tokens)
        if output_tokens > 0:
            self.llm_tokens.labels(
                model=model, module_type=module_type, token_type="output"
            ).inc(output_tokens)

    def record_optimization(
        self, optimizer_type: str, duration: float, status: str = "success"
    ):
        """Record an optimization run."""
        self.optimization_runs.labels(
            optimizer_type=optimizer_type, status=status
        ).inc()
        self.optimization_duration.labels(optimizer_type=optimizer_type).observe(
            duration
        )

    def set_performance_score(self, module_name: str, metric_type: str, score: float):
        """Set performance score for a module."""
        self.performance_score.labels(
            module_name=module_name, metric_type=metric_type
        ).set(score)

    def inc_active_requests(self):
        """Increment active requests."""
        self.active_requests.inc()

    def dec_active_requests(self):
        """Decrement active requests."""
        self.active_requests.dec()


class InstrumentedModule(dspy.Module):
    """Base class for DSPy modules with metrics collection."""

    def __init__(
        self, metrics_collector: DSPyMetricsCollector, module_type: str = "unknown"
    ):
        super().__init__()
        self.metrics = metrics_collector
        self.module_type = module_type
        self.model_name = "unknown"

    def _get_model_name(self):
        """Extract model name from DSPy configuration."""
        try:
            # This is a simplified way to get model name
            if hasattr(dspy.settings, "lm"):
                return (
                    str(dspy.settings.lm).split("/")[-1]
                    if "/" in str(dspy.settings.lm)
                    else str(dspy.settings.lm)
                )
            return "unknown"
        except:
            return "unknown"

    def forward(self, *args, **kwargs):
        """Instrumented forward method."""
        self.model_name = self._get_model_name()

        start_time = time.time()
        self.metrics.inc_active_requests()

        try:
            result = self._forward_impl(*args, **kwargs)
            duration = time.time() - start_time

            # Estimate tokens (simplified)
            input_text = " ".join(str(arg) for arg in args) + " ".join(
                str(v) for v in kwargs.values()
            )
            output_text = str(result)
            input_tokens = len(input_text.split()) * 1.3  # Rough approximation
            output_tokens = len(output_text.split()) * 1.3

            self.metrics.record_llm_request(
                model=self.model_name,
                module_type=self.module_type,
                duration=duration,
                status="success",
                input_tokens=int(input_tokens),
                output_tokens=int(output_tokens),
            )

            return result

        except Exception as e:
            duration = time.time() - start_time
            self.metrics.record_llm_request(
                model=self.model_name,
                module_type=self.module_type,
                duration=duration,
                status="error",
            )
            raise e
        finally:
            self.metrics.dec_active_requests()

    def _forward_impl(self, *args, **kwargs):
        """Override this method instead of forward."""
        raise NotImplementedError("Subclasses must implement _forward_impl")


class MonitoredQAModule(InstrumentedModule):
    """Question answering module with monitoring."""

    def __init__(self, metrics_collector: DSPyMetricsCollector):
        super().__init__(metrics_collector, "qa")
        self.qa = dspy.ChainOfThought("question -> answer")

    def _forward_impl(self, question: str):
        return self.qa(question=question)


class MonitoredSummarizer(InstrumentedModule):
    """Summarizer module with monitoring."""

    def __init__(self, metrics_collector: DSPyMetricsCollector):
        super().__init__(metrics_collector, "summarizer")
        self.summarizer = dspy.ChainOfThought("document -> summary")

    def _forward_impl(self, document: str):
        return self.summarizer(document=document)


class MonitoredClassifier(InstrumentedModule):
    """Classification module with monitoring."""

    def __init__(self, metrics_collector: DSPyMetricsCollector):
        super().__init__(metrics_collector, "classifier")
        self.classifier = dspy.ChainOfThought("text -> category, confidence")

    def _forward_impl(self, text: str):
        return self.classifier(text=text)


def run_demo_workload(metrics_collector: DSPyMetricsCollector):
    """Run demo workload to generate metrics."""
    print("🔄 Running demo workload to generate metrics...")

    # Create monitored modules
    qa_module = MonitoredQAModule(metrics_collector)
    summarizer = MonitoredSummarizer(metrics_collector)
    classifier = MonitoredClassifier(metrics_collector)

    # Sample data
    questions = [
        "What is machine learning?",
        "How does cloud computing work?",
        "What are the benefits of renewable energy?",
    ]

    documents = [
        "Machine learning is a subset of artificial intelligence that enables computers to learn from data.",
        "Cloud computing provides on-demand access to computing resources over the internet.",
    ]

    texts = [
        "This product is amazing! I love it.",
        "Terrible experience, would not recommend.",
        "It's okay, nothing special but works fine.",
    ]

    # Generate some traffic
    for i in range(3):
        print(f"   Batch {i+1}/3...")

        # QA requests
        for question in questions:
            try:
                result = qa_module(question)
                print(f"     Q: {question[:50]}... -> A: {str(result)[:30]}...")
            except Exception as e:
                print(f"     QA Error: {e}")

        # Summarization requests
        for doc in documents:
            try:
                result = summarizer(doc)
                print(f"     Doc: {doc[:30]}... -> Summary: {str(result)[:30]}...")
            except Exception as e:
                print(f"     Summarization Error: {e}")

        # Classification requests
        for text in texts:
            try:
                result = classifier(text)
                print(f"     Text: {text[:30]}... -> Category: {str(result)[:30]}...")
            except Exception as e:
                print(f"     Classification Error: {e}")

        time.sleep(1)  # Small delay between batches

    # Set some performance scores
    metrics_collector.set_performance_score("qa_module", "accuracy", 0.85)
    metrics_collector.set_performance_score("summarizer", "rouge_score", 0.72)
    metrics_collector.set_performance_score("classifier", "f1_score", 0.91)

    print("✅ Demo workload completed")


def create_grafana_config():
    """Create basic Grafana configuration files."""
    grafana_dir = Path("examples/infrastructure/grafana")

    # Create directories
    (grafana_dir / "dashboards").mkdir(parents=True, exist_ok=True)
    (grafana_dir / "datasources").mkdir(parents=True, exist_ok=True)

    # Datasource configuration
    datasource_config = {
        "apiVersion": 1,
        "datasources": [
            {
                "name": "Prometheus",
                "type": "prometheus",
                "access": "proxy",
                "url": "http://prometheus:9090",
                "isDefault": True,
            }
        ],
    }

    with open(grafana_dir / "datasources" / "prometheus.yml", "w") as f:
        import yaml

        yaml.dump(datasource_config, f)

    # Dashboard configuration
    dashboard_config = {
        "dashboard": {
            "id": None,
            "title": "DSPy Application Metrics",
            "panels": [
                {
                    "title": "LLM Requests per Second",
                    "type": "stat",
                    "targets": [
                        {
                            "expr": "rate(dspy_llm_requests_total[5m])",
                            "legendFormat": "{{model}} - {{module_type}}",
                        }
                    ],
                },
                {
                    "title": "Request Duration",
                    "type": "graph",
                    "targets": [
                        {
                            "expr": "histogram_quantile(0.95, rate(dspy_llm_request_duration_seconds_bucket[5m]))",
                            "legendFormat": "95th percentile",
                        }
                    ],
                },
            ],
        }
    }

    with open(grafana_dir / "dashboards" / "dspy-dashboard.json", "w") as f:
        json.dump(dashboard_config, f, indent=2)


def main():
    """Run Prometheus metrics collection example."""
    print("📊 DSPy Infrastructure Integration: Prometheus Metrics")
    print("=" * 70)

    # Configure DSPy
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

    # Create metrics collector
    metrics_collector = DSPyMetricsCollector()

    # Start Prometheus HTTP server
    port = int(os.getenv("PROMETHEUS_PORT", 8000))
    print(f"\n🚀 Starting Prometheus metrics server on port {port}...")

    try:
        start_http_server(port, registry=metrics_collector.registry)
        print(f"✅ Metrics server started: http://localhost:{port}/metrics")
    except Exception as e:
        print(f"❌ Failed to start metrics server: {e}")
        return

    print(f"\n📈 Available Metrics:")
    print(f"   - dspy_llm_requests_total: Total LLM requests")
    print(f"   - dspy_llm_request_duration_seconds: Request duration")
    print(f"   - dspy_llm_tokens_total: Token counts")
    print(f"   - dspy_active_requests: Active requests")
    print(f"   - dspy_performance_score: Module performance scores")
    print(f"   - dspy_system_info: System information")

    try:
        # Run demo workload
        run_demo_workload(metrics_collector)

        print(f"\n🔍 Sample Metrics Output:")
        print("-" * 40)

        # Show sample metrics
        metrics_output = generate_latest(metrics_collector.registry).decode("utf-8")

        # Show first few lines of metrics
        lines = metrics_output.split("\n")
        for line in lines[:20]:
            if line.strip() and not line.startswith("#"):
                print(f"   {line}")
        print("   ...")

        print(f"\n🌐 Access your metrics:")
        print(f"   Metrics endpoint: http://localhost:{port}/metrics")
        print(f"   Curl command: curl http://localhost:{port}/metrics")

        print(f"\n🔧 Production Setup:")
        print(f"   1. Configure Prometheus to scrape http://localhost:{port}/metrics")
        print(f"   2. Set up Grafana dashboards using the metrics")
        print(f"   3. Create alerts for high error rates or latency")
        print(f"   4. Use Docker Compose for full monitoring stack")

        print(f"\n⏰ Server will run for 60 seconds...")
        print(f"   Visit http://localhost:{port}/metrics in your browser")
        print(f"   Press Ctrl+C to stop the server")

        # Keep server running
        for i in range(60):
            time.sleep(1)
            if i % 10 == 0 and i > 0:
                print(f"   {60-i} seconds remaining...")

    except KeyboardInterrupt:
        print(f"\n⚠️  Server stopped by user")
    except Exception as e:
        print(f"\n❌ Error during execution: {e}")

    print(f"\n✅ Prometheus metrics example completed!")
    print(f"   For production deployment, see docker-compose.yml")


if __name__ == "__main__":
    main()
