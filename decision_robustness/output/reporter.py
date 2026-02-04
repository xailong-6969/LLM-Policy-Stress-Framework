"""
Report generation for decision diagnostics.

Produces formatted reports in multiple formats (text, JSON, HTML).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from decision_robustness.output.diagnostics import DecisionDiagnostics


class ReportFormat(Enum):
    """Available report formats."""
    TEXT = auto()
    JSON = auto()
    HTML = auto()
    MARKDOWN = auto()


class Reporter:
    """
    Generates formatted reports from diagnostics.
    """
    
    def __init__(self, diagnostics: "DecisionDiagnostics"):
        """
        Initialize reporter.
        
        Args:
            diagnostics: Decision diagnostics to report on
        """
        self.diagnostics = diagnostics
    
    def generate(self, format: ReportFormat = ReportFormat.TEXT) -> str:
        """
        Generate report in specified format.
        
        Args:
            format: Output format
            
        Returns:
            Formatted report string
        """
        if format == ReportFormat.TEXT:
            return self._generate_text()
        elif format == ReportFormat.JSON:
            return self._generate_json()
        elif format == ReportFormat.HTML:
            return self._generate_html()
        elif format == ReportFormat.MARKDOWN:
            return self._generate_markdown()
        else:
            raise ValueError(f"Unknown format: {format}")
    
    def _generate_text(self) -> str:
        """Generate plain text report."""
        return self.diagnostics.describe()
    
    def _generate_json(self) -> str:
        """Generate JSON report."""
        data = self.diagnostics.to_dict()
        return json.dumps(data, indent=2, default=str)
    
    def _generate_markdown(self) -> str:
        """Generate Markdown report."""
        d = self.diagnostics
        rp = d.risk_profile
        
        lines = [
            "# Decision Robustness Report",
            "",
            "## Risk Profile",
            "",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Overall Risk Level | **{rp.overall_risk_level.upper()}** |" if rp else "",
            f"| Stability Grade | **{rp.stability_grade}** |" if rp else "",
            f"| Failure Probability | {rp.failure_probability:.1%} |" if rp else "",
            f"| Brittleness Score | {rp.brittleness:.3f} |" if rp else "",
            "",
            "## Summary Statistics",
            "",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Total Runs | {d.summary.get('total_runs', 0)} |",
            f"| Success Rate | {d.summary.get('success_rate', 0):.1%} |",
            f"| Failure Rate | {d.summary.get('failure_rate', 0):.1%} |",
            f"| Mean Score | {d.summary.get('mean_score', 0):.3f} |",
            "",
        ]
        
        if rp and rp.key_risks:
            lines.extend([
                "## Key Risks",
                "",
            ])
            for risk in rp.key_risks:
                lines.append(f"- ⚠️ {risk}")
            lines.append("")
        
        # Survival analysis
        if d.survival_curve:
            lines.extend([
                "## Survival Analysis",
                "",
                f"- Final survival rate: {d.survival_curve.survival_prob[-1]:.1%}" if d.survival_curve.survival_prob else "",
                f"- Median survival time: {d.survival_curve.median_survival() or 'N/A'}",
                "",
            ])
        
        # Collapse analysis  
        if d.collapse_metrics:
            cm = d.collapse_metrics
            lines.extend([
                "## Collapse Analysis",
                "",
                f"- Total collapses: {cm.collapse_count}/{cm.total_runs}",
                f"- Early collapse rate: {cm.early_collapse_rate:.1%}",
                f"- Late collapse rate: {cm.late_collapse_rate:.1%}",
                "",
            ])
        
        return "\n".join(lines)
    
    def _generate_html(self) -> str:
        """Generate HTML report."""
        d = self.diagnostics
        rp = d.risk_profile
        
        # Risk level colors
        risk_colors = {
            "low": "#22c55e",
            "moderate": "#eab308",
            "high": "#f97316",
            "critical": "#ef4444",
        }
        risk_color = risk_colors.get(rp.overall_risk_level, "#gray") if rp else "#gray"
        
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Decision Robustness Report</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; 
               max-width: 800px; margin: 0 auto; padding: 20px; background: #0f172a; color: #e2e8f0; }}
        h1 {{ color: #f8fafc; }}
        h2 {{ color: #cbd5e1; border-bottom: 1px solid #334155; padding-bottom: 8px; }}
        .risk-badge {{ display: inline-block; padding: 8px 16px; border-radius: 6px; 
                       font-weight: bold; font-size: 1.2em; background: {risk_color}; color: white; }}
        .grade {{ font-size: 2em; font-weight: bold; color: #38bdf8; }}
        .metric-grid {{ display: grid; grid-template-columns: repeat(2, 1fr); gap: 16px; }}
        .metric-card {{ background: #1e293b; padding: 16px; border-radius: 8px; }}
        .metric-label {{ color: #94a3b8; font-size: 0.9em; }}
        .metric-value {{ font-size: 1.5em; font-weight: bold; color: #f8fafc; }}
        .risk-list {{ list-style: none; padding: 0; }}
        .risk-item {{ padding: 8px 12px; margin: 4px 0; background: #7f1d1d; border-radius: 4px; }}
        table {{ width: 100%; border-collapse: collapse; }}
        th, td {{ padding: 8px 12px; text-align: left; border-bottom: 1px solid #334155; }}
        th {{ color: #94a3b8; }}
    </style>
</head>
<body>
    <h1>🎯 Decision Robustness Report</h1>
    
    <div style="display: flex; align-items: center; gap: 24px; margin: 24px 0;">
        <div>
            <span class="metric-label">Overall Risk</span><br>
            <span class="risk-badge">{rp.overall_risk_level.upper() if rp else 'N/A'}</span>
        </div>
        <div>
            <span class="metric-label">Stability Grade</span><br>
            <span class="grade">{rp.stability_grade if rp else 'N/A'}</span>
        </div>
    </div>
    
    <h2>Summary</h2>
    <div class="metric-grid">
        <div class="metric-card">
            <div class="metric-label">Total Runs</div>
            <div class="metric-value">{d.summary.get('total_runs', 0)}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Success Rate</div>
            <div class="metric-value">{d.summary.get('success_rate', 0):.1%}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Failure Rate</div>
            <div class="metric-value">{d.summary.get('failure_rate', 0):.1%}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Mean Score</div>
            <div class="metric-value">{d.summary.get('mean_score', 0):.3f}</div>
        </div>
    </div>
    
    {"<h2>⚠️ Key Risks</h2><ul class='risk-list'>" + "".join(f"<li class='risk-item'>{r}</li>" for r in rp.key_risks) + "</ul>" if rp and rp.key_risks else ""}
    
    <h2>Detailed Metrics</h2>
    <table>
        <tr><th>Metric</th><th>Value</th></tr>
        <tr><td>Failure Probability</td><td>{rp.failure_probability:.1%}</td></tr>
        <tr><td>Tail Risk</td><td>{rp.tail_risk:.1%}</td></tr>
        <tr><td>Brittleness Score</td><td>{rp.brittleness:.3f}</td></tr>
        <tr><td>Mean Time to Failure</td><td>{rp.time_to_failure:.1f if rp.time_to_failure else 'N/A'}</td></tr>
    </table>
    
    <p style="color: #64748b; font-size: 0.8em; margin-top: 40px;">
        Generated by Decision Robustness Framework
    </p>
</body>
</html>
"""
        return html
    
    def save(self, filepath: str, format: Optional[ReportFormat] = None) -> None:
        """
        Save report to file.
        
        Args:
            filepath: Output file path
            format: Format (inferred from extension if not specified)
        """
        if format is None:
            if filepath.endswith('.json'):
                format = ReportFormat.JSON
            elif filepath.endswith('.html'):
                format = ReportFormat.HTML
            elif filepath.endswith('.md'):
                format = ReportFormat.MARKDOWN
            else:
                format = ReportFormat.TEXT
        
        content = self.generate(format)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
