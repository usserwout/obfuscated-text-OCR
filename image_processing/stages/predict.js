const tf = require("@tensorflow/tfjs-node")
const path = require("path")
const jimp = require("jimp")
const fs = require("fs")

let model
let characters

async function loadModel() {
  model = await tf.loadLayersModel(`file://${path.join(__dirname, "./model/model.json")}`)
  const labelsPath = path.join(__dirname, "./model/labels.json")
  characters = JSON.parse(fs.readFileSync(labelsPath))
}

async function preprocessImage(image) {
  const tensor = tf.browser
    .fromPixels(image.bitmap)
    .resizeNearestNeighbor([32, 32])
    .toFloat()
    .div(tf.scalar(255.0))
    .expandDims()
  return tensor
}

async function predictCharacter(image) {
  if (!model) {
    await loadModel()
  }
  const tensor = await preprocessImage(image)
  const prediction = model.predict(tensor)
  const predictedIndex = prediction.argMax(1).dataSync()[0]
  return characters[predictedIndex]
}

async function test() {
 // 88HD-2R54-T5Z
 let output = []
  for(let i = 0; i < 11; i++) {
    await jimp.read(path.join(__dirname, `../processed/split/split-${i}.png`)).then(async (image) => {
      const character = await predictCharacter(image)
      output.push(character)
    })
  }

  console.log(output.join(""))
  console.log("88HD2R54T5Z")

}

test()

module.exports = { predictCharacter }
