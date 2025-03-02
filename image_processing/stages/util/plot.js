import fs from "fs"
import { ChartJSNodeCanvas } from "chartjs-node-canvas"


const width = 800 // width of the chart
const height = 600 // height of the chart
const chartJSNodeCanvas = new ChartJSNodeCanvas({ width, height })


async function plotValues(values) {
  const configuration = {
    type: "bar",
    data: {
      labels: values.map((_, index) => index),
      datasets: [
        {
          label: "Character Splitter Values",
          data: values,
          backgroundColor: "rgba(75, 192, 192, 0.2)",
          borderColor: "rgba(75, 192, 192, 1)",
          borderWidth: 1,
        },
      ],
    },
    options: {
      scales: {
        y: {
          beginAtZero: true,
        },
      },
    },
  }

  // const imageBuffer = await chartJSNodeCanvas.renderToBuffer(configuration)
  // fs.writeFileSync("chart.png", imageBuffer)
}

export { plotValues }