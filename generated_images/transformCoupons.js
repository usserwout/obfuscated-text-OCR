import { processImage } from "../image_processing/fitCharacters.js"
import fs from "fs"
import { Jimp } from "jimp"

const IMAGE_WIDTH = 256
const IMAGE_HEIGHT = 50

async function main() {
  const couponPositions = JSON.parse(fs.readFileSync("./generated/metadata.json"))
  let images = fs.readdirSync("./generated").filter((f) => f.endsWith(".png") && !f.endsWith("_transformed.png"))

  for (const imagePath of images) {
    console.log(`Processing: ${imagePath}`)
    let img = couponPositions[imagePath]
    const newImageName = imagePath.replace(".png", "") + "_transformed.png"
    if (!img) throw new Error(`No position found for ${imagePath}`)
    couponPositions[newImageName] = img
    delete couponPositions[imagePath]

    const result = await processImage(`./generated/${imagePath}`, true)
    if (result === null) throw new Error("Characters not detected for image: " + imagePath)
    const { image /* Jimp image */, cropInfo } = result

    let scaleX = IMAGE_WIDTH / image.bitmap.width
    let scaleY = IMAGE_HEIGHT / image.bitmap.height

    img.char_positions.forEach((e) => {
      e[1][0] = (e[1][0] - cropInfo.x) * scaleX
      e[1][1] = (e[1][1] - cropInfo.y) * scaleY
      e[1][2] = (e[1][2] - cropInfo.x) * scaleX
      e[1][3] = (e[1][3] - cropInfo.y) * scaleY
      e[1] = e[1].map((d) => Math.round(d))
    })

    // Resize to image to 256x50
    image.resize({ w: IMAGE_WIDTH, h: IMAGE_HEIGHT })

    // Add a rectangle around every character
    // img.char_positions.forEach((e) => {
    //   let [x0, y0, x1, y1] = e[1]

    //   for(let x = x0; x <= x1; x++) {
    //     image.setPixelColor(0xff0000ff, x, y0)
    //     image.setPixelColor(0xff0000ff, x, y1)
    //   }

    //   for (let y = y0; y <= y1; y++) {
    //     image.setPixelColor(0xff0000ff, x0, y)
    //     image.setPixelColor(0xff0000ff, x1, y)

    //   }
    // })

    image.write(`./generated/${newImageName}`)
  }

  fs.writeFileSync("./generated/metadata_transformed.json", JSON.stringify(couponPositions))
}

main()
