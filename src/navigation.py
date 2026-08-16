"""Navigation registry for CompositeAI Streamlit app."""

from __future__ import annotations

from collections import OrderedDict
from typing import Callable

PageRenderer = Callable[[], None]

_NAVIGATION = OrderedDict(
    [
        ("PROJECT", ["Dashboard", "About Project", "Workflow"]),
        (
            "DATA",
            [
                "Dataset Explorer",
                "Dataset Profile",
                "EDA",
                "Feature Engineering",
                "Preprocessing",
            ],
        ),
        ("AI", ["Model Performance", "Strength Prediction"]),
        (
            "ENGINEERING ANALYSIS",
            [
                "CLT Analysis",
                "Stacking Optimizer",
                "AI vs CLT Comparison",
                "Composite vs Aerospace Metals",
            ],
        ),
        (
            "ENGINEERING VISUALIZATIONS",
            [
                "Strength Surface",
                "Optimization Landscape",
                "Ply Failure Map",
                "Material Benchmark",
            ],
        ),
        ("REPORTS", ["Report Generator"]),
        ("TOOLS", ["Gemini Assistant"]),
    ]
)


def get_available_pages() -> OrderedDict[str, list[str]]:
    """Return ordered sidebar page registry."""
    return OrderedDict((section, list(items)) for section, items in _NAVIGATION.items())


def render_page(selected_page: str, renderers: dict[str, PageRenderer]) -> None:
    """Render selected page from single source of truth."""
    page_name = selected_page if selected_page in renderers else "Dashboard"
    renderers[page_name]()


def validate_navigation(
    renderers: dict[str, PageRenderer] | None = None,
) -> dict[str, list[str] | bool]:
    """Validate registry for duplicates and missing renderers."""
    pages = [page for items in _NAVIGATION.values() for page in items]
    duplicates = sorted({page for page in pages if pages.count(page) > 1})
    missing_renderers = []
    unreachable_renderers = []

    if renderers is not None:
        missing_renderers = sorted(page for page in pages if page not in renderers)
        unreachable_renderers = sorted(
            page for page in renderers if page not in pages and page != "Dashboard"
        )

    return {
        "valid": not duplicates and not missing_renderers,
        "duplicates": duplicates,
        "missing_renderers": missing_renderers,
        "unreachable_renderers": unreachable_renderers,
    }
