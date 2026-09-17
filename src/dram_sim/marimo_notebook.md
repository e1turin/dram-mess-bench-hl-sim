Отлично, тогда виджет должен работать как «графический планшет»: пользователь рисует кривую мышью/тачем, а Python получает набор точек и может строить аппроксимацию, интерполяцию или анализировать форму. Ниже — минимальный рабочий пример для marimo, где график рисуется от руки, а не задаётся формулой.

## Виджет: рисование кривой от руки

```python
import anywidget
import traitlets
import marimo as mo


class HandDrawnCurve(anywidget.AnyWidget):
    _esm = """
    function render({ model, el }) {
      const root = document.createElement("div");
      root.className = "hand-drawn-curve";

      const canvas = document.createElement("canvas");
      canvas.width = 800;
      canvas.height = 400;

      const controls = document.createElement("div");
      controls.className = "controls";

      const clearButton = document.createElement("button");
      clearButton.textContent = "Очистить";

      controls.appendChild(clearButton);
      root.appendChild(canvas);
      root.appendChild(controls);
      el.appendChild(root);

      const ctx = canvas.getContext("2d");

      let isDrawing = false;
      let currentStroke = [];

      function xToPixel(x) {
        const xMin = model.get("x_min");
        const xMax = model.get("x_max");
        return (x - xMin) / (xMax - xMin) * canvas.width;
      }

      function yToPixel(y) {
        const yMin = model.get("y_min");
        const yMax = model.get("y_max");
        return canvas.height - (y - yMin) / (yMax - yMin) * canvas.height;
      }

      function pixelToX(px) {
        const xMin = model.get("x_min");
        const xMax = model.get("x_max");
        return px / canvas.width * (xMax - xMin) + xMin;
      }

      function pixelToY(py) {
        const yMin = model.get("y_min");
        const yMax = model.get("y_max");
        return (canvas.height - py) / canvas.height * (yMax - yMin) + yMin;
      }

      function drawAxes() {
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        ctx.strokeStyle = "#e5e7eb";
        ctx.lineWidth = 1;

        // Вертикальные линии
        const xMin = model.get("x_min");
        const xMax = model.get("x_max");
        const yMin = model.get("y_min");
        const yMax = model.get("y_max");

        for (let x = Math.ceil(xMin); x <= Math.floor(xMax); x++) {
          const px = xToPixel(x);
          ctx.beginPath();
          ctx.moveTo(px, 0);
          ctx.lineTo(px, canvas.height);
          ctx.stroke();
        }

        // Горизонтальные линии
        for (let y = Math.ceil(yMin); y <= Math.floor(yMax); y++) {
          const py = yToPixel(y);
          ctx.beginPath();
          ctx.moveTo(0, py);
          ctx.lineTo(canvas.width, py);
          ctx.stroke();
        }

        // Оси
        ctx.strokeStyle = "#9ca3af";
        ctx.lineWidth = 2;

        // Ось X
        const y0 = yToPixel(0);
        if (y0 >= 0 && y0 <= canvas.height) {
          ctx.beginPath();
          ctx.moveTo(0, y0);
          ctx.lineTo(canvas.width, y0);
          ctx.stroke();
        }

        // Ось Y
        const x0 = xToPixel(0);
        if (x0 >= 0 && x0 <= canvas.width) {
          ctx.beginPath();
          ctx.moveTo(x0, 0);
          ctx.lineTo(x0, canvas.height);
          ctx.stroke();
        }
      }

      function drawCurve() {
        const strokes = model.get("strokes") || [];

        ctx.strokeStyle = "#2563eb";
        ctx.lineWidth = 2;
        ctx.lineCap = "round";
        ctx.lineJoin = "round";

        for (const stroke of strokes) {
          if (stroke.length === 0) continue;

          ctx.beginPath();
          const [x0, y0] = stroke[0];
          ctx.moveTo(xToPixel(x0), yToPixel(y0));

          for (let i = 1; i < stroke.length; i++) {
            const [x, y] = stroke[i];
            ctx.lineTo(xToPixel(x), yToPixel(y));
          }

          ctx.stroke();
        }
      }

      function redraw() {
        drawAxes();
        drawCurve();
      }

      function getPointFromEvent(event) {
        const rect = canvas.getBoundingClientRect();
        const px = event.clientX - rect.left;
        const py = event.clientY - rect.top;
        return [pixelToX(px), pixelToY(py)];
      }

      canvas.addEventListener("pointerdown", (event) => {
        isDrawing = true;
        currentStroke = [getPointFromEvent(event)];

        const strokes = [...(model.get("strokes") || [])];
        strokes.push(currentStroke);
        model.set("strokes", strokes);
        model.save_changes();

        redraw();
      });

      canvas.addEventListener("pointermove", (event) => {
        if (!isDrawing) return;

        const point = getPointFromEvent(event);
        currentStroke.push(point);

        const strokes = model.get("strokes") || [];
        strokes[strokes.length - 1] = currentStroke;
        model.set("strokes", strokes);
        model.save_changes();

        redraw();
      });

      canvas.addEventListener("pointerup", () => {
        isDrawing = false;
        currentStroke = [];
      });

      canvas.addEventListener("pointerleave", () => {
        isDrawing = false;
        currentStroke = [];
      });

      clearButton.addEventListener("click", () => {
        model.set("strokes", []);
        model.save_changes();
        redraw();
      });

      model.on("change:x_min", redraw);
      model.on("change:x_max", redraw);
      model.on("change:y_min", redraw);
      model.on("change:y_max", redraw);
      model.on("change:strokes", redraw);

      redraw();
    }

    export default { render };
    """

    _css = """
    .hand-drawn-curve {
      display: flex;
      flex-direction: column;
      gap: 8px;
      width: fit-content;
    }

    canvas {
      border: 1px solid #ccc;
      background: white;
      cursor: crosshair;
      touch-action: none;
    }

    button {
      width: fit-content;
      padding: 4px 10px;
    }
    """

    x_min = traitlets.Float(-10.0).tag(sync=True)
    x_max = traitlets.Float(10.0).tag(sync=True)
    y_min = traitlets.Float(-10.0).tag(sync=True)
    y_max = traitlets.Float(10.0).tag(sync=True)

    strokes = traitlets.List(
        trait=traitlets.List(
            trait=traitlets.List(
                trait=traitlets.Float(),
                minlen=2,
                maxlen=2,
            ),
        ),
    ).tag(sync=True)
```

Создайте и отобразите виджет:

```python
curve = HandDrawnCurve(
    x_min=-10,
    x_max=10,
    y_min=-10,
    y_max=10,
    strokes=[],
)

widget = mo.ui.anywidget(curve)
widget
```

Теперь можно рисовать кривую прямо в браузере.

## Получение нарисованных данных

В следующей ячейке:

```python
strokes = curve.strokes
strokes
```

`strokes` — это список «штрихов», где каждый штрих — список точек `[[x1, y1], [x2, y2], ...]`.

Если вам нужна одна кривая (например, первый штрих):

```python
if strokes:
    first = strokes[0]
    xs = [p[0] for p in first]
    ys = [p[1] for p in first]
else:
    xs, ys = [], []
```

## Построение аппроксимации или сглаживание

Например, полиномиальная аппроксимация:

```python
import numpy as np
import matplotlib.pyplot as plt

if len(xs) >= 2:
    x_arr = np.array(xs)
    y_arr = np.array(ys)

    deg = 3  # степень полинома
    coeffs = np.polyfit(x_arr, y_arr, deg)
    poly = np.poly1d(coeffs)

    x_fit = np.linspace(min(xs), max(xs), 400)
    y_fit = poly(x_fit)

    plt.figure(figsize=(6, 4))
    plt.plot(x_arr, y_arr, "o", label="Нарисованные точки")
    plt.plot(x_fit, y_fit, "-", label=f"Аппроксимация (deg={deg})")
    plt.grid()
    plt.legend()
    plt.show()
```

Или сглаживание сплайном:

```python
from scipy.interpolate import UnivariateSpline

if len(xs) >= 4:
    x_arr = np.array(xs)
    y_arr = np.array(ys)

    order = np.argsort(x_arr)
    x_sorted = x_arr[order]
    y_sorted = y_arr[order]

    spline = UnivariateSpline(x_sorted, y_sorted, s=1.0)

    x_fit = np.linspace(x_sorted.min(), x_sorted.max(), 400)
    y_fit = spline(x_fit)

    plt.figure(figsize=(6, 4))
    plt.plot(x_sorted, y_sorted, "o", label="Точки")
    plt.plot(x_fit, y_fit, "-", label="Сплайн")
    plt.grid()
    plt.legend()
    plt.show()
```

## Как это использовать дальше

- Рисовать несколько кривых и сравнивать их.
- Вычислять производную/интеграл нарисованной функции численно.
- Подбирать параметры модели под форму нарисованной кривой.
- Экспортировать точки в CSV для дальнейшего анализа.

Если нужно, могу показать вариант, где виджет сразу возвращает одну «усреднённую» кривую (например, по первому штриху) и автоматически строит сглаженную версию в Python.
