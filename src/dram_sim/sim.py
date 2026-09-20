import marimo

__generated_with = "0.24.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import anywidget
    import json
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import re
    import statistics as stats
    import traitlets
    from dataclasses import dataclass, field
    from datetime import datetime
    from pathlib import Path
    from typing import Callable, Sequence
    from simpy import Environment, Resource
    from simpy.core import SimTime

    return (
        Callable,
        Environment,
        Path,
        Resource,
        Sequence,
        SimTime,
        anywidget,
        dataclass,
        datetime,
        field,
        json,
        mo,
        np,
        pd,
        plt,
        re,
        stats,
        traitlets,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # DRAM MESS high-level simulation

    Draw absolute DRAM latency as a function of queue depth, then run a sweep
    over CPU counts. Curve values are interpreted directly as nanoseconds.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Environment
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### System Setup
    """)
    return


@app.cell
def _(SimTime):
    TICK: SimTime = 10
    NS = TICK
    US = 1_000 * NS
    MS = 1_000 * US
    SECOND = 1_000 * MS
    GHZ = 1_000_000_000
    return NS, SECOND, US


@app.cell
def _(NS, SimTime, dataclass, field):
    @dataclass(kw_only=True)
    class DramParameters:
        base_latency: SimTime = 70 * NS
        channels: int = 1

    @dataclass(kw_only=True)
    class CpuParameters:
        frequency: float = 4.0 * 1_000_000_000
        instructions_per_loop: int = 1_000

    @dataclass(kw_only=True)
    class Parameters:
        running_time: SimTime = 1_000 * 1_000 * NS
        dram: DramParameters = field(default_factory=DramParameters)
        cpu: CpuParameters = field(default_factory=CpuParameters)

    return CpuParameters, DramParameters, Parameters


@app.cell
def _(Callable, Environment, Parameters, Resource, SECOND, SimTime, np, stats):
    class DramDevice:
        def __init__(
            self,
            dram_env: Environment,
            latency_calculator: Callable[[int], SimTime],
            channels: int,
        ):
            self.env, self.resource, self.latency_calculator = (
                dram_env,
                Resource(dram_env, capacity=channels),
                latency_calculator,
            )
            self.latencies: list[float] = []
            self.max_queue = 0
            self.time_series: list[dict[str, float]] = []

        def read(self):
            with self.resource.request() as request:
                started = self.env.now
                queue_depth = len(self.resource.queue)
                self.max_queue = max(self.max_queue, queue_depth)
                yield request
                yield self.env.timeout(self.latency_calculator(queue_depth))
                self.latencies.append(self.env.now - started)

    class CpuSource:
        def __init__(
            self, dram_device: DramDevice, rng: np.random.Generator, interval: float
        ):
            self.dram, self.rng, self.interval = dram_device, rng, interval

        def run(self, cpu_env: Environment):
            while True:
                cpu_env.process(self.dram.read())
                yield cpu_env.timeout(self.rng.poisson(lam=self.interval))

    def simulate(
        cpu_count: int, params: Parameters, latency_calculator: Callable[[int], SimTime]
    ) -> tuple[dict[str, float], list[dict[str, float]]]:
        sim_env = Environment()
        sim_dram = DramDevice(sim_env, latency_calculator, params.dram.channels)

        def sample_metrics():
            sample_interval = max(params.running_time / 200, 1)
            while True:
                sim_dram.time_series.append(
                    {
                        "time": float(sim_env.now),
                        "lat_avg": (
                            stats.mean(sim_dram.latencies)
                            if sim_dram.latencies
                            else float("nan")
                        ),
                        "queue_size": len(sim_dram.resource.queue),
                    }
                )
                yield sim_env.timeout(sample_interval)

        sim_env.process(sample_metrics())
        for cpu_number in range(cpu_count):
            sim_env.process(
                CpuSource(
                    sim_dram,
                    np.random.default_rng(cpu_number + 1),
                    SECOND / params.cpu.frequency * params.cpu.instructions_per_loop,
                ).run(sim_env)
            )
        sim_env.run(until=params.running_time)
        summary = {
            "n_cpu": cpu_count,
            "lat_avg": stats.mean(sim_dram.latencies) if sim_dram.latencies else 0.0,
            "tput": len(sim_dram.latencies) / params.running_time,
            "max_queue": sim_dram.max_queue,
        }
        time_series = [dict(sample, n_cpu=cpu_count) for sample in sim_dram.time_series]
        return summary, time_series

    return (simulate,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Latency-Capacity Curve Setup
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Utility class for interactive function drowing widget.
    """)
    return


@app.cell(hide_code=True)
def _(anywidget, traitlets):
    class HandDrawnCurve(anywidget.AnyWidget):
        """Canvas widget that synchronizes hand-drawn strokes to Python."""

        _esm = r"""
        function render({ model, el }) {
          const canvas = document.createElement("canvas");
          canvas.width = 800; canvas.height = 360;
          const clear = document.createElement("button"); clear.textContent = "Clear curve";
          const randomToggle = document.createElement("input"); randomToggle.type = "checkbox"; randomToggle.checked = model.get("random_colors");
          const randomLabel = document.createElement("label"); randomLabel.append(randomToggle, " Random stroke colors");
          const controls = document.createElement("div"); controls.className = "controls"; controls.append(clear, randomLabel);
          const root = document.createElement("div"); root.className = "curve";
          root.append(canvas, controls); el.appendChild(root);
          const ctx = canvas.getContext("2d"); let drawing = false; let stroke = [];
          const cloneStrokes = strokes => (strokes || []).map(saved => saved.map(point => [...point]));
          let nextHue = Math.random() * 360;
          const randomColor = () => { const color = `hsl(${Math.round(nextHue)} 75% 42%)`; nextHue = (nextHue + 137.508) % 360; return color; };
          let draftStrokes = cloneStrokes(model.get("strokes"));
          let draftColors = [...(model.get("stroke_colors") || [])];
          while (draftColors.length < draftStrokes.length) draftColors.push(randomColor());
          const pad = { left: 72, right: 16, top: 18, bottom: 50 };
          const plotWidth = () => canvas.width - pad.left - pad.right;
          const plotHeight = () => canvas.height - pad.top - pad.bottom;
          const xp = x => pad.left + (x - model.get("x_min")) / (model.get("x_max") - model.get("x_min")) * plotWidth();
          const yp = y => canvas.height - pad.bottom - (y - model.get("y_min")) / (model.get("y_max") - model.get("y_min")) * plotHeight();
          const point = event => {
            const rect = canvas.getBoundingClientRect();
            const px = Math.min(Math.max(event.clientX - rect.left, pad.left), canvas.width - pad.right);
            const py = Math.min(Math.max(event.clientY - rect.top, pad.top), canvas.height - pad.bottom);
            return [
              (px - pad.left) / plotWidth() * (model.get("x_max") - model.get("x_min")) + model.get("x_min"),
              (canvas.height - pad.bottom - py) / plotHeight() * (model.get("y_max") - model.get("y_min")) + model.get("y_min"),
            ];
          };
          function redraw() {
            ctx.clearRect(0, 0, canvas.width, canvas.height); ctx.strokeStyle = "#e5e7eb"; ctx.lineWidth = 1;
            const xmin = model.get("x_min"), xmax = model.get("x_max"), ymin = model.get("y_min"), ymax = model.get("y_max");
            const xStep = (xmax - xmin) / 8, yStep = (ymax - ymin) / 6;
            ctx.font = "12px system-ui"; ctx.fillStyle = "#475569"; ctx.textAlign = "center";
            for (let index = 0; index <= 8; index++) { const x = xmin + index * xStep; const px = xp(x); ctx.beginPath(); ctx.moveTo(px, pad.top); ctx.lineTo(px, canvas.height - pad.bottom); ctx.stroke(); ctx.fillText(x.toPrecision(3), px, canvas.height - pad.bottom + 18); }
            ctx.textAlign = "right";
            for (let index = 0; index <= 6; index++) { const y = ymin + index * yStep; const py = yp(y); ctx.beginPath(); ctx.moveTo(pad.left, py); ctx.lineTo(canvas.width - pad.right, py); ctx.stroke(); ctx.fillText(y.toPrecision(3), pad.left - 8, py + 4); }
            ctx.strokeStyle = "#64748b"; ctx.lineWidth = 1.5; ctx.beginPath(); ctx.moveTo(pad.left, pad.top); ctx.lineTo(pad.left, canvas.height - pad.bottom); ctx.lineTo(canvas.width - pad.right, canvas.height - pad.bottom); ctx.stroke();
            ctx.fillStyle = "#0f172a"; ctx.textAlign = "center"; ctx.fillText("Queue depth (requests)", pad.left + plotWidth() / 2, canvas.height - 10);
            ctx.save(); ctx.translate(16, pad.top + plotHeight() / 2); ctx.rotate(-Math.PI / 2); ctx.fillText("Latency (ns)", 0, 0); ctx.restore();
            ctx.lineWidth = 2; ctx.lineCap = "round";
            for (const [index, saved] of draftStrokes.entries()) { if (!saved.length) continue; ctx.strokeStyle = model.get("random_colors") ? draftColors[index] : "#2563eb"; ctx.beginPath(); ctx.moveTo(xp(saved[0][0]), yp(saved[0][1])); for (const [x, y] of saved.slice(1)) ctx.lineTo(xp(x), yp(y)); ctx.stroke(); }
          }
          canvas.addEventListener("pointerdown", event => {
            drawing = true; canvas.setPointerCapture(event.pointerId); stroke = [point(event)];
            draftStrokes = [...draftStrokes, stroke]; draftColors = [...draftColors, randomColor()]; redraw();
          });
          canvas.addEventListener("pointermove", event => {
            if (!drawing) return; stroke.push(point(event)); redraw();
          });
          const stop = event => {
            if (!drawing) return;
            if (event.type === "pointerup") stroke.push(point(event));
            drawing = false; model.set("strokes", cloneStrokes(draftStrokes)); model.set("stroke_colors", [...draftColors]); model.save_changes(); stroke = []; redraw();
          };
          canvas.addEventListener("pointerup", stop); canvas.addEventListener("pointercancel", stop);
          clear.addEventListener("click", () => { drawing = false; stroke = []; draftStrokes = []; draftColors = []; model.set("strokes", []); model.set("stroke_colors", []); model.save_changes(); redraw(); });
          randomToggle.addEventListener("change", () => { model.set("random_colors", randomToggle.checked); model.save_changes(); redraw(); });
          model.on("change:strokes", () => { if (!drawing) { draftStrokes = cloneStrokes(model.get("strokes")); while (draftColors.length < draftStrokes.length) draftColors.push(randomColor()); } redraw(); });
          model.on("change:stroke_colors", () => { if (!drawing) draftColors = [...(model.get("stroke_colors") || [])]; redraw(); });
          model.on("change:random_colors", () => { randomToggle.checked = model.get("random_colors"); redraw(); });
          model.on("change:x_min change:x_max change:y_min change:y_max", redraw); redraw();
        }
        export default { render };
        """
        _css = ".curve { display:grid; gap:8px; width:fit-content } .curve canvas { border:1px solid #cbd5e1; background:white; cursor:crosshair; touch-action:none } .curve .controls { display:flex; align-items:center; gap:16px } .curve button { width:fit-content; padding:4px 10px } .curve label { display:flex; align-items:center; cursor:pointer }"
        x_min = traitlets.Float(0.0).tag(sync=True)
        x_max = traitlets.Float(3_000.0).tag(sync=True)
        y_min = traitlets.Float(20.0).tag(sync=True)
        y_max = traitlets.Float(500.0).tag(sync=True)
        strokes = traitlets.List(default_value=[]).tag(sync=True)
        stroke_colors = traitlets.List(trait=traitlets.Unicode(), default_value=[]).tag(sync=True)
        random_colors = traitlets.Bool(True).tag(sync=True)

    return (HandDrawnCurve,)


@app.cell(hide_code=True)
def _(mo):
    base_latency_input = mo.ui.slider(
        20, 200, value=70, step=5, label="Fallback DRAM latency (ns)"
    )
    channel_input = mo.ui.slider(1, 8, value=1, step=1, label="DRAM channels")
    queue_min_input = mo.ui.number(value=0, step=100, label="Queue-depth x minimum")
    queue_max_input = mo.ui.number(value=3_000, step=100, label="Queue-depth x maximum")
    latency_min_input = mo.ui.number(
        value=5, step=1, label="Latency y minimum (ns)"
    )
    latency_max_input = mo.ui.number(
        value=100, step=1, label="Latency y maximum (ns)"
    )
    frequency_input = mo.ui.slider(
        1.0, 6.0, value=4.0, step=0.1, label="CPU frequency (GHz)"
    )
    instructions_input = mo.ui.slider(
        100, 5_000, value=1_000, step=50, label="Instructions per loop"
    )
    runtime_input = mo.ui.slider(
        10, 10_000, value=1_000, step=10, label="Simulation duration (µs)"
    )
    simulation_name_input = mo.ui.text(value="dram-mess", label="Simulation name")
    cpu_counts_input = mo.ui.text(value="1,2,3,4,5,6,8,10,12,16,20", label="CPU counts")
    return (
        base_latency_input,
        channel_input,
        cpu_counts_input,
        frequency_input,
        instructions_input,
        latency_max_input,
        latency_min_input,
        queue_max_input,
        queue_min_input,
        runtime_input,
        simulation_name_input,
    )


@app.cell(hide_code=True)
def _(latency_max_input, latency_min_input, queue_max_input, queue_min_input):
    curve_x_min = float(queue_min_input.value or 0)
    curve_x_max = max(float(queue_max_input.value or curve_x_min + 1), curve_x_min + 1)
    curve_y_min = float(latency_min_input.value or 0)
    curve_y_max = max(
        float(latency_max_input.value or curve_y_min + 1), curve_y_min + 1
    )
    return curve_x_max, curve_x_min, curve_y_max, curve_y_min


@app.cell(hide_code=True)
def _(
    base_latency_input,
    channel_input,
    cpu_counts_input,
    frequency_input,
    instructions_input,
    latency_max_input,
    latency_min_input,
    mo,
    queue_max_input,
    queue_min_input,
    runtime_input,
    simulation_name_input,
):
    mo.vstack(
        [
            mo.md("#### Configure and draw"),
            mo.hstack([base_latency_input, channel_input, frequency_input]),
            mo.hstack([instructions_input, runtime_input, cpu_counts_input]),
            simulation_name_input,
            mo.md(
                "#### Drawing range\nSet the axis bounds before drawing; changing a bound creates a fresh canvas."
            ),
            mo.hstack([queue_min_input, queue_max_input]),
            mo.hstack([latency_min_input, latency_max_input]),
        ]
    )
    return


@app.cell(hide_code=True)
def _(HandDrawnCurve, curve_x_max, curve_x_min, curve_y_max, curve_y_min):
    curve_model = HandDrawnCurve(
        x_min=curve_x_min, x_max=curve_x_max, y_min=curve_y_min, y_max=curve_y_max
    )
    return (curve_model,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Draw your `capacity -> latency` function plot:
    """)
    return


@app.cell(hide_code=True)
def _(curve_model, mo):
    curve_widget = mo.ui.anywidget(curve_model)
    curve_widget
    return (curve_widget,)


@app.cell(hide_code=True)
def _(Sequence, np):
    def normalized_curve(
        strokes: Sequence[Sequence[Sequence[float]]],
    ) -> tuple[np.ndarray, np.ndarray]:
        valid_strokes = [stroke for stroke in strokes if len(stroke) >= 2]
        if not valid_strokes:
            return np.array([], dtype=float), np.array([], dtype=float)
        samples = np.asarray(valid_strokes[-1], dtype=float)
        ordered = samples[np.argsort(samples[:, 0], kind="stable")]
        curve_x, bucket = np.unique(ordered[:, 0], return_inverse=True)
        curve_y = np.zeros_like(curve_x)
        np.add.at(curve_y, bucket, ordered[:, 1])
        return curve_x, curve_y / np.bincount(bucket)

    return (normalized_curve,)


@app.cell(hide_code=True)
def _(curve_widget, normalized_curve, np, queue_max_input):
    drawn_x, drawn_y = normalized_curve(curve_widget.value.get("strokes", []))
    preview_x = (
        np.linspace(0, queue_max_input.value, 400) if drawn_x.size else np.array([])
    )
    preview_y = np.interp(preview_x, drawn_x, drawn_y) if drawn_x.size else np.array([])
    return drawn_x, drawn_y, preview_x, preview_y


@app.cell(hide_code=True)
def _(drawn_x, drawn_y, plt, preview_x, preview_y):


    curve_figure, curve_axis = plt.subplots(figsize=(8, 3))
    if drawn_x.size:
        curve_axis.plot(drawn_x, drawn_y, ".", alpha=0.35, label="drawn samples")
        curve_axis.plot(preview_x, preview_y, label="interpolated latency")
        curve_axis.legend()
    else:
        curve_axis.text(
            0.5,
            0.5,
            "Draw a curve to preview it",
            transform=curve_axis.transAxes,
            ha="center",
        )
    curve_axis.set(
        xlabel="Queue depth (requests)",
        ylabel="Latency (ns)",
        title="Latency curve",
    )
    curve_axis.grid()
    curve_figure.tight_layout()
    curve_figure
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Run Simulation
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    run_button = mo.ui.run_button(label="Run simulation", kind="success")
    run_button
    return (run_button,)


@app.cell(hide_code=True)
def _(
    CpuParameters,
    DramParameters,
    NS,
    Parameters,
    base_latency_input,
    channel_input,
    cpu_counts_input,
    drawn_x,
    drawn_y,
    frequency_input,
    instructions_input,
    mo,
    np,
    pd,
    run_button,
    runtime_input,
    simulate,
):
    result_table = pd.DataFrame(columns=["n_cpu", "lat_avg", "tput", "max_queue"])
    time_series_table = pd.DataFrame(
        columns=["time", "lat_avg", "queue_size", "n_cpu"]
    )
    if not run_button.value:
        run_status = mo.md("Press run simulation button")
    else:
        requested_cpu_counts = [
            int(item.strip()) for item in cpu_counts_input.value.split(",") if item.strip()
        ]
        active_parameters = Parameters(
            running_time=runtime_input.value * 1_000 * NS,
            dram=DramParameters(
                base_latency=base_latency_input.value * NS, channels=channel_input.value
            ),
            cpu=CpuParameters(
                frequency=frequency_input.value * 1_000_000_000,
                instructions_per_loop=instructions_input.value,
            ),
        )
        def hand_drawn_latency(queue_depth: int) -> float:
            latency_ns = (
                float(np.interp(queue_depth, drawn_x, drawn_y))
                if drawn_x.size
                else active_parameters.dram.base_latency / NS
            )
            return float(max(latency_ns, 0.0) * NS)

        simulation_results = [
            simulate(cpu_count, active_parameters, hand_drawn_latency)
            for cpu_count in requested_cpu_counts
        ]
        result_table = pd.DataFrame([summary for summary, _trace in simulation_results])
        time_series_table = pd.DataFrame(
            sample
            for _summary, trace in simulation_results
            for sample in trace
        )
        run_status = mo.callout(
            "Simulation complete",
            kind="success",
        )

    run_status
    return result_table, time_series_table


@app.cell(hide_code=True)
def _(US, plt, result_table, time_series_table):
    results_figure, result_axes = plt.subplots(2, 2, figsize=(11, 7))
    latency_axis, queue_axis = result_axes[0]
    latency_time_axis, queue_time_axis = result_axes[1]
    bandwidth = result_table["tput"] * US
    latency = result_table["lat_avg"] / US
    latency_axis.plot(bandwidth, latency, "o-")
    for row_index, row in result_table.iterrows():
        latency_axis.annotate(
            str(int(row["n_cpu"])),
            (bandwidth[row_index], latency[row_index]),
            xytext=(6, 5),
            textcoords="offset points",
        )
    latency_axis.set(
        xlabel="Bandwidth (requests / µs)",
        ylabel="Average latency (µs)",
        title="Latency vs bandwidth",
    )
    queue_axis.plot(result_table["n_cpu"], result_table["max_queue"], "o-")
    queue_axis.set(
        xlabel="CPU count", ylabel="Maximum queue length", title="Queue depth"
    )

    for cpu_count, samples in time_series_table.groupby("n_cpu"):
        simulation_time = samples["time"] / US
        latency_time_axis.plot(
            simulation_time,
            samples["lat_avg"] / US,
            label=f"{int(cpu_count)} CPUs",
        )
        queue_time_axis.plot(
            simulation_time,
            samples["queue_size"],
            label=f"{int(cpu_count)} CPUs",
        )
    latency_time_axis.set(
        xlabel="Simulation time (µs)",
        ylabel="Cumulative average latency (µs)",
        title="Average latency over time",
    )
    queue_time_axis.set(
        xlabel="Simulation time (µs)",
        ylabel="Waiting requests",
        title="Queue size over time",
    )
    if not time_series_table.empty:
        latency_time_axis.legend(fontsize="small", ncol=2)
        queue_time_axis.legend(fontsize="small", ncol=2)
    for result_axis in result_axes.flat:
        result_axis.grid()
    results_figure.tight_layout()
    results_figure
    return (results_figure,)


@app.cell(hide_code=True)
def _(mo, result_table):
    mo.vstack([mo.md("## Results"), result_table])
    return


@app.cell(hide_code=True)
def _(mo):
    save_results_button = mo.ui.run_button(label="Save plots and configuration")
    save_results_button
    return (save_results_button,)


@app.cell(hide_code=True)
def _(
    Path,
    base_latency_input,
    channel_input,
    cpu_counts_input,
    curve_model,
    datetime,
    frequency_input,
    instructions_input,
    json,
    latency_max_input,
    latency_min_input,
    mo,
    queue_max_input,
    queue_min_input,
    re,
    result_table,
    results_figure,
    runtime_input,
    save_results_button,
    simulation_name_input,
):
    if not save_results_button.value:
        save_status = mo.md("Save the current plots and configuration to `out/`.")
    else:
        safe_name = re.sub(
            r"[^A-Za-z0-9._-]+", "-", simulation_name_input.value.strip()
        )
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        artifact_stem = f"{safe_name or 'simulation'}-{timestamp}"
        output_directory = Path("out")
        output_directory.mkdir(exist_ok=True)
        plot_path = output_directory / f"{artifact_stem}.png"
        config_path = output_directory / f"{artifact_stem}.json"
        results_figure.savefig(plot_path, dpi=160, bbox_inches="tight")
        configuration = {
            "simulation_name": simulation_name_input.value,
            "saved_at": timestamp,
            "dram": {
                "base_latency_ns": base_latency_input.value,
                "channels": channel_input.value,
            },
            "cpu": {
                "frequency_ghz": frequency_input.value,
                "instructions_per_loop": instructions_input.value,
                "counts": cpu_counts_input.value,
            },
            "running_time_us": runtime_input.value,
            "drawing_range": {
                "queue_min": queue_min_input.value,
                "queue_max": queue_max_input.value,
                "latency_min_ns": latency_min_input.value,
                "latency_max_ns": latency_max_input.value,
            },
            "drawn_strokes": curve_model.strokes,
            "results": result_table.to_dict(orient="records"),
        }
        config_path.write_text(json.dumps(configuration, indent=2), encoding="utf-8")
        save_status = mo.callout(
            f"Saved [{plot_path}]({plot_path}) and [{config_path}]({config_path}).",
            kind="success",
        )
    save_status
    return


if __name__ == "__main__":
    app.run()
