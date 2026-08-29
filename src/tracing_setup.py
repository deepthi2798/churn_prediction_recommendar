"""
OpenTelemetry setup for the churn pipeline.

Default: exports spans to the console, so it works with zero extra
infrastructure. To point traces at a real backend instead (Jaeger, Grafana
Tempo, Honeycomb, etc.), swap CONSOLE for the OTLP exporter -- see the
commented block below. That's the only change needed; every span in the
pipeline (data load, train, predict, LLM call) stays the same.

Optional nicer local UI (needs Docker, which you already use for bET):
    docker run -d --name jaeger -p 16686:16686 -p 4317:4317 \
        jaegertracing/all-in-one:latest
    -> then open http://localhost:16686 to see traces visually
    -> and switch EXPORTER below to "otlp"
"""

import os
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    ConsoleSpanExporter,
    SimpleSpanProcessor,
)
from opentelemetry.sdk.resources import Resource

EXPORTER = os.environ.get("OTEL_EXPORTER", "console")  # "console" or "otlp"
# when running via Docker Compose, this becomes "tempo:4317" (the service
# name on the compose network) instead of "localhost:4317"
OTLP_ENDPOINT = os.environ.get("OTEL_EXPORTER_ENDPOINT", "localhost:4317")
_configured = False


def get_tracer(service_name: str = "churn-pipeline"):
    global _configured
    provider = trace.get_tracer_provider()

    if not _configured:
        resource = Resource.create({"service.name": service_name})
        provider = TracerProvider(resource=resource)

        if EXPORTER == "otlp":
            # requires: pip install opentelemetry-exporter-otlp
            # requires: a collector running (e.g. Jaeger, see docstring above)
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
                OTLPSpanExporter,
            )
            exporter = OTLPSpanExporter(endpoint=OTLP_ENDPOINT, insecure=True)
            provider.add_span_processor(BatchSpanProcessor(exporter))
        else:
            provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))

        trace.set_tracer_provider(provider)
        _configured = True

    return trace.get_tracer(service_name)
