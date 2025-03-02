import path from "path"
import { createCanvas, registerFont } from "canvas"
import { Jimp } from "jimp"
import { fileURLToPath } from "url"

const __filename = fileURLToPath(import.meta.url)
const __dirname = path.dirname(__filename)



registerFont(path.join(__dirname,"../burbankbigcondensed_bold.otf"), { family: "Burbank" })
const canvas = createCanvas(32, 40) // createCanvas(300, 40)
const context = canvas.getContext("2d")
context.textAlign = "center"
context.textBaseline = "middle"

// Fill the background with white color

context.fillRect(0, 0, canvas.width, canvas.height)

context.fillStyle = "black"

function getCharacterBorder(image) {
  let values = Array(image.bitmap.height).fill(0)
  let total = 0
  let startX = 1e10
  let endX = -1
  for (let h = 0; h < image.bitmap.height; h++) {
    for (let w = 0; w < image.bitmap.width; w++) {
      const color = image.getPixelColor(w, h)
      const red = (color >> 24) & 0xff
      const green = (color >> 16) & 0xff
      const blue = (color >> 8) & 0xff

      if (red + green + blue > 0xff * 3 - 0x66) {
        values[h]++
        total++

        if (w < startX) startX = w
        if (w > endX) endX = w
      }
    }
  }
  let threshold = total / (values.length * 2)
  const lastIndex = values.findLastIndex((v) => v > threshold)
  const firstIndex = values.findIndex((v) => v > threshold)
  // Draw the 2 lines
  // for (let w = 0; w < image.bitmap.width; w++) {
  //   image.setPixelColor(0xff0000ff, w, lastIndex)
  //   image.setPixelColor(0xff0000ff, w, firstIndex)
  // }

  // for(let h = firstIndex; h < lastIndex; h++) {
  //   image.setPixelColor(0xff0000ff, startX, h)
  //   image.setPixelColor(0xff0000ff, endX, h)
  // }

  return { y: firstIndex, height: lastIndex - firstIndex, x: startX, width: endX - startX }
}

async function createCharacters(character, fontSize) {
  context.clearRect(0, 0, canvas.width, canvas.height)
  context.fillStyle = "white"
  context.font = `${fontSize}px "Burbank"`
  context.fillText(character, fontSize / 4, fontSize / 2)

  // Convert the canvas to a Buffer
  const buffer = canvas.toBuffer()
  const image = await Jimp.read(buffer)

  let startX = image.bitmap.width
  let endX = 0
  let startY = image.bitmap.height
  let endY = 0

  for (let x = 0; x < image.bitmap.width; x++) {
    for (let y = 0; y < image.bitmap.height; y++) {
      const pixel = image.getPixelColor(x, y)
      if (pixel !== 0) {
        if (x < startX) startX = x
        if (x > endX) endX = x
        if (y < startY) startY = y
        if (y > endY) endY = y
      }
    }
  }

  image.crop({ x: startX, y: startY, w: endX - startX, h: endY - startY })

  return image
}

function isLetter(color) {
  return ((color >> 24) & 0xff) + ((color >> 16) & 0xff) + ((color >> 8) & 0xff) > 0xff * 3 - 0xf6
}

function getPixel(image, x, y) {
  if (x < 0 || x >= image.bitmap.width) return 0
  if (y < 0 || y >= image.bitmap.height) return 0
  return image.getPixelColor(x, y)
}

// Returns the loss
function evaluateCharacter(image, character, charX, charY, height, imageX, imageY, errorImg) {
  let error = 0

  let startX = Math.min(imageX, charX)
  let startY = Math.min(imageY, charY)
  let endY = Math.max(imageY + height, charY + height)
  let endX = charX + character.bitmap.width //Math.max(imageX + charWidth, charX + charWidth)
  // Draw lines of field we are checking

  // const color = rgbaToInt(Math.random() * 0xff, Math.random() * 0xff, Math.random() * 0xff, 0xFF)
  // for (let x = 0; x < endX - startX; x++) {
  //   image.setPixelColor(color, startX + x, startY)
  //   image.setPixelColor(color, startX + x, endY )
  // }

  // for (let y = 0; y < endY - startY; y++) {
  //   image.setPixelColor(color, startX, startY + y)
  //   image.setPixelColor(color, endX, startY + y)
  // }

  for (let dx = 0; dx < endX - startX; dx++) {
    for (let dy = 0; dy < endY - startY; dy++) {
      const imageColor = getPixel(image, startX + dx, startY + dy)

      const charColor = getPixel(character, dx, dy)

      // Goal:
      // 1) Minimize the left over white pixels
      // 2) Mininize the non overlapping pixels

      // const dr = Math.abs(((charColor >> 24) & 0xff) - ((imageColor >> 24) & 0xff))
      // const dg = Math.abs(((charColor >> 16) & 0xff) - ((imageColor >> 16) & 0xff))
      // const db = Math.abs(((charColor >> 8) & 0xff) - ((imageColor >> 8) & 0xff))

      // if (dr + dg + db < 0xff) {
      //   error -= 0xff
      // } else if (dr + dg + db > 0xff*2.5) {
      //   error += 0xff
      // }
      // error += dr + dg + db//Math.sqrt(dr * dr + dg * dg + db * db)
      // continue
      let isLetterChar = (charColor & 0xff) !== 0
      let isLetterImage = isLetter(imageColor)


      let errColor = errorImg.getPixelColor(startX + dx, startY + dy)
      const errRed = (errColor >> 24) & 0xff
      const errGreen = (errColor >> 16) & 0xff
      const errBlue = (errColor >> 8) & 0xff

      const charRed = (charColor >> 24) & 0xff
      const charGreen = (charColor >> 16) & 0xff
      const charBlue = (charColor >> 8) & 0xff

      errColor = errRed + errGreen + errBlue
      const errRatio = errColor / (3 * 0xff)

      if (isLetterImage) {
        // We do nothing if:
        // * The image is letter and the character is not letter
        // * They are both black, to ensure a fair comparison (since not every character is equal in size)
        //
        // We penalize if:
        // * Pixel is inside char but not inside char in image does not -> penalty is lower when pixel is closer to the edge of the character
        // * Both are letters: we give a negative reward. More reward if both pixels are close to the edge of the character
        //

        if (isLetterChar) {
          // Reward if both pixels are characters
          error -= errRatio * 50
        } else {
          // Punish that the image pixel doesn't match the character
          error += errRatio * 150
        }
      } else {
        if (isLetterChar) {
          // Punish that the character pixel doesn't match a letter from the image
          error += errRatio * 100
        } else {
        }
      }
    }
  }

  return error // / (character.bitmap.width)
}

async function getBestCharacter(image, char, height, x, y, errorImg) {
  let lowestError = 1e10
  let bestCharacter = null
  let best = 0
  for (let scale = 2; scale < 4; scale++) {
    const character = await createCharacters(char, Math.floor(height * 1.35) + scale)
    if (char === "-") character.write(`./processed/char-${char}.png`)

    const heightChar = character.bitmap.height

    let middle = Math.floor(Math.abs(height - heightChar) / 2)
    for (let dx = -3; dx < 4; dx++) {
      for (let dy = -5 + middle; dy < 2 + middle; dy++) {
        let v = evaluateCharacter(image, character, x + dx, y + dy, height, x, y, errorImg)

        if (v < lowestError) {
          lowestError = v
          best = { width: character.bitmap.width, x: x + dx, y: y + dy, scale }
          bestCharacter = character
        }
      }
    }
  }

  image.clone().composite(bestCharacter, best.x, best.y).write(`./processed/bestChar-${char}.png`)
  return { error: lowestError, endX: best.width + best.x, x: best.x, y: best.y, image: bestCharacter, char, scale: best.scale }
}

async function fitCharacters(image) {
  let couponCode = []
  const errorImg = image.clone().convolution([
    [1 / 9, 1 / 9, 1 / 9],
    [1 / 9, 1 / 9, 1 / 9],
    [1 / 9, 1 / 9, 1 / 9],
  ])

  // Darken the image
  errorImg.scan(0, 0, errorImg.bitmap.width, errorImg.bitmap.height, function (x, y, idx) {
    this.bitmap.data[idx] = this.bitmap.data[idx] * 0.8 // Red
    this.bitmap.data[idx + 1] = this.bitmap.data[idx + 1] * 0.8 // Green
    this.bitmap.data[idx + 2] = this.bitmap.data[idx + 2] * 0.8 // Blue
  })
  errorImg.write("./processed/errorImg.png")
  let { x, y, width, height } = getCharacterBorder(image)
  let maxWidth = x + width
  while (x + 10 < maxWidth) {
    let results = []
    for (const char of "ABCDEFGHJKMNPQRSTUVWXYZ23456789-") {
      results.push(await getBestCharacter(image, char, height, x, y, errorImg))
    }
    results.sort((a, b) => a.error - b.error)

    console.log(
      "Result: ",
      results.map((r) => `${r.char}: ${r.error}`)
    )
    const best = results[0]
    console.log("=>",best.scale)
    couponCode.push(best)
    image.composite(best.image.invert(), best.x, best.y)

  //  if (couponCode.length === 3) break
    x = best.endX
  }

  // couponCode.forEach((r) => {
  //   image.composite(r.image.invert(), r.x, r.y)
  // })

  console.log("Coupon code: ", couponCode.map((r) => r.char).join(""))
  return image
}

export { fitCharacters }
