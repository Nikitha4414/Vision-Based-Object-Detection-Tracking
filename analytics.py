"""
Analytics and Visualization Module for Vision-Based Object Detection & Tracking Using YOLO.
Provides real-time trajectory logging, class distribution charts, and interactive Plotly metrics.
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


class AnalyticsEngine:
    """
    Tracks session telemetry and generates interactive Plotly analytics charts.
    """

    def __init__(self, max_history=300):
        self.max_history = max_history
        self.fps_history = []
        self.object_count_history = []
        self.frame_indices = []
        self.frame_counter = 0

    def log_frame(self, fps, active_objects_count):
        """Log frame telemetry data."""
        self.frame_counter += 1
        self.frame_indices.append(self.frame_counter)
        self.fps_history.append(fps)
        self.object_count_history.append(active_objects_count)

        # Trim history buffer to max_history length
        if len(self.frame_indices) > self.max_history:
            self.frame_indices.pop(0)
            self.fps_history.pop(0)
            self.object_count_history.pop(0)

    def reset(self):
        """Reset historical logs."""
        self.fps_history.clear()
        self.object_count_history.clear()
        self.frame_indices.clear()
        self.frame_counter = 0

    def create_performance_chart(self):
        """
        Generate interactive Plotly line chart for FPS and Active Object count.
        """
        if not self.frame_indices:
            # Return empty figure placeholder
            fig = go.Figure()
            fig.update_layout(
                title="Performance Telemetry (Awaiting Feed)",
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#94a3b8")
            )
            return fig

        df = pd.DataFrame({
            "Frame": self.frame_indices,
            "FPS": self.fps_history,
            "Active Objects": self.object_count_history
        })

        fig = go.Figure()
        
        # FPS trace
        fig.add_trace(go.Scatter(
            x=df["Frame"],
            y=df["FPS"],
            mode="lines",
            name="FPS Rate",
            line=dict(color="#00f2fe", width=2.5),
            fill="tozeroy",
            fillcolor="rgba(0, 242, 254, 0.08)"
        ))

        # Active Objects trace (secondary y-axis)
        fig.add_trace(go.Scatter(
            x=df["Frame"],
            y=df["Active Objects"],
            mode="lines+markers",
            name="Detected Objects",
            line=dict(color="#10b981", width=2, dash="dot"),
            yaxis="y2"
        ))

        fig.update_layout(
            title=dict(text="⚡ Real-Time Processing Throughput & Object Density", font=dict(size=14, color="#f8fafc")),
            xaxis=dict(title="Frame Sequence", showgrid=True, gridcolor="rgba(255,255,255,0.05)"),
            yaxis=dict(title="Frames Per Second (FPS)", showgrid=True, gridcolor="rgba(255,255,255,0.05)"),
            yaxis2=dict(title="Active Objects Count", overlaying="y", side="right", showgrid=False),
            template="plotly_dark",
            paper_bgcolor="rgba(15, 23, 42, 0.4)",
            plot_bgcolor="rgba(0, 0, 0, 0)",
            height=320,
            margin=dict(l=20, r=20, t=40, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )

        return fig

    def create_class_distribution_chart(self, class_counts):
        """
        Generate interactive Plotly bar chart for Object Class Frequency Distribution.
        """
        if not class_counts:
            fig = go.Figure()
            fig.update_layout(
                title="Class Breakdown (No Objects Detected)",
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#94a3b8")
            )
            return fig

        df = pd.DataFrame(list(class_counts.items()), columns=["Class", "Count"]).sort_values(by="Count", ascending=True)

        fig = px.bar(
            df,
            x="Count",
            y="Class",
            orientation="h",
            text="Count",
            color="Count",
            color_continuous_scale=["#1e293b", "#00f2fe", "#4facfe"]
        )

        fig.update_layout(
            title=dict(text="📊 Active Object Class Frequency", font=dict(size=14, color="#f8fafc")),
            xaxis=dict(title="Detected Count", showgrid=True, gridcolor="rgba(255,255,255,0.05)"),
            yaxis=dict(title=""),
            template="plotly_dark",
            paper_bgcolor="rgba(15, 23, 42, 0.4)",
            plot_bgcolor="rgba(0, 0, 0, 0)",
            height=320,
            margin=dict(l=20, r=20, t=40, b=20),
            coloraxis_showscale=False
        )

        fig.update_traces(textposition="outside", marker_line_width=0)
        return fig
