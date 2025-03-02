const fs = require("fs")
const Jimp = require("jimp")
const path = require("path")


const dataDir = path.join(__dirname, "../../images/characters2")

async function processImage(imgPath, output) {
  const image = await Jimp.read(imgPath)

  for (let y = 0; y < image.bitmap.height; y++) {
    for (let x = 0; x < image.bitmap.width; x++) {
      const color = image.getPixelColor(x, y)
      if (color === 0xffffffff) output[y][x] -= 1
      else {
        output[y][x] += 1
        image.setPixelColor(0x000000ff, x, y)
      }
    }
  }


  image.write(imgPath)

}

async function train(dir) {
  let result = {}
  const directories = fs.readdirSync(dir).filter((file) => file !== ".DS_Store")

  for (const label of directories) {
    let promisses = []
    result[label] = Array.from({ length: 32 }, () => Array(32).fill(0))
    const files = fs.readdirSync(path.join(dir, label)).filter((file) => file !== ".DS_Store")
    for (const file of files) {
      const imagePath = path.join(dir, label, file)
      promisses.push(processImage(imagePath, result[label]))
    }
    await Promise.all(promisses)

    // divide entire array by files.length
    for (let y = 0; y < 32; y++) {
      for (let x = 0; x < 32; x++) {
        result[label][y][x] /= files.length
      }
    }
  }

  fs.writeFileSync(path.join(__dirname, "./model/characters2.json"), JSON.stringify(result))
}

train(dataDir)
