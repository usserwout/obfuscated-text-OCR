import { intToRGBA, rgbaToInt } from "jimp"
/**
 * A border is detected if pixel:
 * - is surrounded by at least 1 0xffffffff     = white
 * - is surrounded by at least 1 0x414E6DFF rgba(65, 78, 109, 1)    = border color            calc: -122
 * - in range of 0x414E6DFF < c < 0xffffffff
 */

function backgroundRemover(image) {
  // grayscale the image

  image.scan(0, 0, image.bitmap.width, image.bitmap.height, function (x, y, idx) {
    const red = this.bitmap.data[idx + 0]
    const green = this.bitmap.data[idx + 1]
    const blue = this.bitmap.data[idx + 2]
    const alpha = this.bitmap.data[idx + 3]
    const avg = (red + green + blue) / 3
    this.bitmap.data[idx + 0] = avg
    this.bitmap.data[idx + 1] = avg
    this.bitmap.data[idx + 2] = avg
    this.bitmap.data[idx + 3] = alpha
  })

  for (let x = 0; x < image.bitmap.width; x++) {
    for (let y = 0; y < image.bitmap.height; y++) {
      const color = (image.getPixelColor(x, y) >> 8) & 0xff
      if (color > 210) {
        image.setPixelColor(0xff0000ff, x, y)
      }
    }
  }

  // EVery blob that touches the border of the image should get removed

  for (let x = 0; x < image.bitmap.width; x++) {
    removeBlob(image, x, 0)
    removeBlob(image, x, image.bitmap.height - 1)
  }

  for (let y = 0; y < image.bitmap.height; y++) {
    removeBlob(image, 0, y)
    removeBlob(image, image.bitmap.width - 1, y)
  }

  // Remove red blobs that are less than 8 pixels large
  removeSmallRedBlobs(image, 5)

  // Change make red colors black, all other colors white

  for (let x = 0; x < image.bitmap.width; x++) {
    for (let y = 0; y < image.bitmap.height; y++) {
      const color = image.getPixelColor(x, y)
      if (color === 0xff0000ff) {
        image.setPixelColor(0x000000ff, x, y)
      } else {
        image.setPixelColor(0xffffffff, x, y)
      }
    }
  }
}

function removeBlob(image, x, y) {
  const blob = []
  floodFill(image, x, y, new Set(), blob)
  let count = 0
  let maxDist = 1 // max dist away from the border
  for (const [bx, by] of blob) {
    if (bx < maxDist || by < maxDist || image.bitmap.width - bx <= maxDist || image.bitmap.height - by <= maxDist) {
      count++
    }
  }

  if (count < 2) return

  for (const [bx, by] of blob) {
    image.setPixelColor(0xffffffff, bx, by)
  }
}

function floodFill(image, x, y, visited, blob) {
  const stack = [[x, y]]
  while (stack.length > 0) {
    const [cx, cy] = stack.pop()
    if (cx >= 0 && cx < image.bitmap.width && cy >= 0 && cy < image.bitmap.height && !visited.has(`${cx},${cy}`)) {
      const color = image.getPixelColor(cx, cy)
      if (color === 0xff0000ff) {
        visited.add(`${cx},${cy}`)
        blob.push([cx, cy])
        stack.push([cx + 1, cy])
        stack.push([cx - 1, cy])
        stack.push([cx, cy + 1])
        stack.push([cx, cy - 1])
      }
    }
  }
}

function removeSmallRedBlobs(image, maxSize) {
  const visited = new Set()

  for (let x = 0; x < image.bitmap.width; x++) {
    for (let y = 0; y < image.bitmap.height; y++) {
      const color = image.getPixelColor(x, y)
      if (color === 0xff0000ff && !visited.has(`${x},${y}`)) {
        const blob = []
        floodFill(image, x, y, visited, blob)
        if (blob.length < maxSize) {
          for (const [bx, by] of blob) {
            image.setPixelColor(0xffffffff, bx, by)
          }
        }
      }
    }
  }
}

function floodFill2(image, x, y, visited, blob) {
  const stack = [[x, y]]
  while (stack.length > 0) {
    const [cx, cy] = stack.pop()
    if (cx >= 0 && cx < image.bitmap.width && cy >= 0 && cy < image.bitmap.height && !visited.has(`${cx},${cy}`)) {
      const color = image.getPixelColor(cx, cy)
      if (color !== 0x0) {
        visited.add(`${cx},${cy}`)
        blob.push([cx, cy])
        stack.push([cx + 1, cy])
        stack.push([cx - 1, cy])
        stack.push([cx, cy + 1])
        stack.push([cx, cy - 1])
      }
    }
  }
}


function removeSmallBlobs(image, maxSize) {
  const visited = new Set()

  for (let x = 0; x < image.bitmap.width; x++) {
    for (let y = 0; y < image.bitmap.height; y++) {
      const color = image.getPixelColor(x, y)
      if (color !== 0x0 && !visited.has(`${x},${y}`)) {
        const blob = []
        floodFill2(image, x, y, visited, blob)
        if (blob.length < maxSize) {
          for (const [bx, by] of blob) {
            image.setPixelColor(0x0, bx, by)
          }
        }
      }
    }
  }
  
}

function removeBgColors(image) {
  for (let x = 0; x < image.bitmap.width; x++) {
    for (let y = 0; y < image.bitmap.height; y++) {
      const color = image.getPixelColor(x, y)
      const red = (color >> 24) & 0xff
      const green = (color >> 16) & 0xff
      const blue = (color >> 8) & 0xff
      if (red <= 0x11 || green <= 0x1e || blue <= 0x1d || green >= 0xee || blue >= 0xcc) {
        if (red + green + blue < 0xff * 3 - 0x88) {
          image.setPixelColor(rgbaToInt(red, green, blue, 0x01), x, y)
        }
      }
    }
  }
}

// Reduce the noise in the image
function bilateralFilter(image, radius, sigma) {
  const width = image.bitmap.width
  const height = image.bitmap.height
  const newImage = image.clone()

  const gaussian = (x, sigma) => Math.exp(-(x * x) / (2 * sigma * sigma))

  const spatialWeights = []
  for (let dx = -radius; dx <= radius; dx++) {
    for (let dy = -radius; dy <= radius; dy++) {
      spatialWeights.push(gaussian(Math.sqrt(dx * dx + dy * dy), sigma))
    }
  }

  for (let x = 0; x < width; x++) {
    for (let y = 0; y < height; y++) {
      let redSum = 0,
        greenSum = 0,
        blueSum = 0,
        weightSum = 0

      const centerColor = intToRGBA(image.getPixelColor(x, y))

      let index = 0
      for (let dx = -radius; dx <= radius; dx++) {
        for (let dy = -radius; dy <= radius; dy++) {
          const nx = x + dx
          const ny = y + dy

          if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
            const neighborColor = intToRGBA(image.getPixelColor(nx, ny))

            const colorWeight = gaussian(
              Math.sqrt(
                Math.pow(centerColor.r - neighborColor.r, 2) +
                  Math.pow(centerColor.g - neighborColor.g, 2) +
                  Math.pow(centerColor.b - neighborColor.b, 2)
              ),
              sigma
            )

            const weight = spatialWeights[index] * colorWeight

            redSum += neighborColor.r * weight
            greenSum += neighborColor.g * weight
            blueSum += neighborColor.b * weight
            weightSum += weight
          }
          index++
        }
      }

      const newRed = Math.round(redSum / weightSum)
      const newGreen = Math.round(greenSum / weightSum)
      const newBlue = Math.round(blueSum / weightSum)

      newImage.setPixelColor(rgbaToInt(newRed, newGreen, newBlue, centerColor.a), x, y)
    }
  }

  return newImage
}

function blobFinder(image, x, y, visited, blob, maxBlobSize) {
  const stack = [[x, y]]
  while (stack.length > 0 && blob.length <= maxBlobSize) {
    const [cx, cy] = stack.pop()
    if (cx >= 0 && cx < image.bitmap.width && cy >= 0 && cy < image.bitmap.height && !visited.has(`${cx},${cy}`)) {
      const color = image.getPixelColor(cx, cy)
      if ((color & 0xff) === 0x1) {
        visited.add(`${cx},${cy}`)
        blob.push([cx, cy])
        stack.push([cx + 1, cy])
        stack.push([cx - 1, cy])
        stack.push([cx, cy + 1])
        stack.push([cx, cy - 1])
      }
    }
  }
}

function removeLargeBlobs(image, minSize) {
  const visited = new Set()
  for (let x = 0; x < image.bitmap.width; x++) {
    for (let y = 0; y < image.bitmap.height; y++) {
      const color = image.getPixelColor(x, y)
      if ((color & 0xff) === 0x1) {
        const blob = []
        blobFinder(image, x, y, visited, blob, minSize)
        if (blob.length > minSize) {
          for (const [bx, by] of blob) {
            image.setPixelColor(0x0, bx, by)
          }
        } else {
          for (const [bx, by] of blob) {
            image.setPixelColor(
              rgbaToInt((color >> 24) & 0xff, (color >> 16) & 0xff, (color >> 8) & 0xff, 0xff),
              bx,
              by
            )
          }
        }
      }
    }
  }
}

function __removeBlob(image, x, y, newColor) {
  if (x < 0 || x >= image.bitmap.width || y < 0 || y >= image.bitmap.height) return
  const color = image.getPixelColor(x, y)
  if (color === newColor) return
  const red = (color >> 24) & 0xff
  const green = (color >> 16) & 0xff
  const blue = (color >> 8) & 0xff
  if (red + green + blue < 0xff * 3 - 0xee) return
  image.setPixelColor(newColor, x, y)
  __removeBlob(image, x + 1, y, newColor)
  __removeBlob(image, x - 1, y, newColor)
  __removeBlob(image, x, y + 1, newColor)
  __removeBlob(image, x, y - 1, newColor)
  
}

function removeWhiteBorders(image) {
  for (let w = 0; w < image.bitmap.width; w++) {
    for (let h of [0, image.bitmap.height - 1]) {
      const color = image.getPixelColor(w, h)
      if ((color & 0xff) === 0) continue

      __removeBlob(image, w, h, 0x0)
    }
  }

  for (let h = 0; h < image.bitmap.height; h++) {
    for (let w of [0, image.bitmap.width - 1]) {
      const color = image.getPixelColor(w, h)
      if ((color & 0xff) === 0) continue

      __removeBlob(image, w, h, 0x0)
      
    }
  }
}

function characterAbstractor(image) {
  image.contrast(0.25)
  const start = Date.now()
  image = bilateralFilter(image, 3, 100)
  console.log("Bilateral filter took", Date.now() - start)
  removeBgColors(image)
  removeLargeBlobs(image, 7)
  removeWhiteBorders(image)
  removeSmallBlobs(image, 15)
  image.threshold({ max: 255 })



  // image.convolution([
  //   [-0.5, 0.5, -0.5],
  //   [0.5, 1, 0.5],
  //   [-0.5, 0.5, -0.5],
  // ])





  return image
}

export { backgroundRemover, characterAbstractor }
