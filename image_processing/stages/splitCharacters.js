import { Jimp } from "jimp"

function splitCharacters(image) {
  const uniqueColors = new Set()
  const width = image.bitmap.width
  const height = image.bitmap.height

  // Identify unique colors (excluding white)
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const color = image.getPixelColor(x, y)
      if (color !== 0xffffffff) {
        // Exclude white
        uniqueColors.add(color)
      }
    }
  }

  const characters = []

  // Create a new image for each unique color
  for (const color of uniqueColors) {
    let pixelCount = 0
    let minX = width,
      minY = height,
      maxX = 0,
      maxY = 0

    // Find the bounding box for the current color
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        if (image.getPixelColor(x, y) === color) {
          if (x < minX) minX = x
          if (y < minY) minY = y
          if (x > maxX) maxX = x
          if (y > maxY) maxY = y
          pixelCount++
        }
      }
    }

    if (pixelCount < 64) continue

    const charWidth = maxX - minX + 1
    const charHeight = maxY - minY + 1
    const charImage = new Jimp(charWidth, charHeight, 0xffffffff) // Create a white image

    // Copy the character pixels to the new image
    for (let y = minY; y <= maxY; y++) {
      for (let x = minX; x <= maxX; x++) {
        if (image.getPixelColor(x, y) === color) {
          charImage.setPixelColor(0x000000ff, x - minX, y - minY)
        }
      }
    }

    // Scale the character image to fit within 32x32 while maintaining aspect ratio
    charImage.scaleToFit(32, 32)

    // Create a 32x32 image and center the scaled character image
    const finalImage = new Jimp(32, 32, 0xffffffff) // Create a white 32x32 image
    const xOffset = (32 - charImage.bitmap.width) / 2
    const yOffset = (32 - charImage.bitmap.height) / 2
    finalImage.composite(charImage, xOffset, yOffset)

    for (let y = 0; y < finalImage.bitmap.height; y++) {
      for (let x = 0; x < finalImage.bitmap.width; x++) {
        const color = finalImage.getPixelColor(x, y)
        if (color !== 0xffffffff && color !== 0x000000ff) {
          finalImage.setPixelColor(0x000000ff, x, y)
        }
      }
    }

    characters.push({ image: finalImage, minX })
  }

  // Sort characters by their minX value
  characters.sort((a, b) => a.minX - b.minX)

  // Return only the images in the sorted order
  return characters.map((char) => char.image)
}

export {
  splitCharacters
}
