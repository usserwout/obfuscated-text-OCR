const tf = require("@tensorflow/tfjs-node")
const fs = require("fs")
const jimp = require("jimp")
const path = require("path")

async function loadImages(dataDir) {
  const classes = fs.readdirSync(dataDir).filter(file => file !== ".DS_Store")
  const images = []
  const labels = []

  for (const label of classes) {
    const files = fs.readdirSync(path.join(dataDir, label)).filter((file) => file !== ".DS_Store")
    for (const file of files) {
      const imagePath = path.join(dataDir, label, file)
      const image = await jimp.read(imagePath)
      const tensor = tf.browser
        .fromPixels(image.bitmap)
        .resizeNearestNeighbor([32, 32])
        .toFloat()
        .div(tf.scalar(255.0))
        .expandDims()
      images.push(tensor)
      labels.push(classes.indexOf(label))
    }
  }

  // Save the labels to a file
  fs.writeFileSync(path.join(__dirname, "./model/labels.json"), JSON.stringify(classes))

  return {
    images: tf.concat(images),
    labels: tf.tensor1d(labels, "int32").toFloat(),
  }
}

function buildCharacterModel(numClasses) {
  const model = tf.sequential()
  model.add(tf.layers.conv2d({ inputShape: [32, 32, 3], filters: 32, kernelSize: 3, activation: "relu" }))
  model.add(tf.layers.maxPooling2d({ poolSize: 2 }))
  model.add(tf.layers.conv2d({ filters: 64, kernelSize: 3, activation: "relu" }))
  model.add(tf.layers.maxPooling2d({ poolSize: 2 }))
  model.add(tf.layers.flatten())
  model.add(tf.layers.dense({ units: 128, activation: "relu" }))
  model.add(tf.layers.dense({ units: numClasses, activation: "softmax" }))
  model.compile({ optimizer: "adam", loss: "sparseCategoricalCrossentropy", metrics: ["accuracy"] })
  return model
}


async function trainModel(dataDir, modelPath) {
  const { images, labels } = await loadImages(dataDir)
  const numClasses = labels.max().dataSync()[0] + 1
  const model = buildCharacterModel(numClasses)
  await model.fit(images, labels, { epochs: 15 })
  await model.save(`file://${modelPath}`)
}

const dataDir = path.join(__dirname, "../../images/characters")
const modelPath = path.join(__dirname, "./model")
trainModel(dataDir, modelPath)
