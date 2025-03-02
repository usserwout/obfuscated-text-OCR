import { walkingMeanFilter } from "./util/filters.js"
import { plotValues } from "./util/plot.js"

const COLOR_BORDER = 0x414e6dff

let bred = COLOR_BORDER >> 24
let bgreen = (COLOR_BORDER & 0x00ff0000) >> 16
let bblue = (COLOR_BORDER & 0x0000ff00) >> 8

function isBorder(color, tolerance) {
  return (
    Math.abs((color >> 24) - bred) < tolerance &&
    Math.abs(((color & 0x00ff0000) >> 16) - bgreen) < tolerance &&
    Math.abs(((color & 0x0000ff00) >> 8) - bblue) < tolerance
  )
}

function detect(image, tolerance = 32) {
  let startX = 0
  let startY = 0
  let endX = 0
  let endY = 0

  // Make borders more clear:

  for (let x = 0; x < image.bitmap.width; x++) {
    for (let y = 0; y < image.bitmap.height; y++) {
      if (isBorder(image.getPixelColor(x, y), tolerance)) {
        image.setPixelColor(COLOR_BORDER, x, y)
      }
    }
  }

  let xValues = Array(image.bitmap.width).fill(0)
  let yValues = Array(image.bitmap.height).fill(0)

  for (let x = 0; x < image.bitmap.width; x++) {
    for (let y = 0; y < image.bitmap.height; y++) {
      const color = image.getPixelColor(x, y)
      if (color === COLOR_BORDER) {
        xValues[x]++
        yValues[y]++
      }
    }
  }
  xValues = walkingMeanFilter(xValues, 4)
  yValues = walkingMeanFilter(yValues, 5)

  // Filter out the noise
  xValues = xValues.map((value) => (value > 5 ? value : 0))
  yValues = yValues.map((value) => (value > 15 ? value : 0))

  for (let i = 1; i < yValues.length && yValues[i] !== 0; i++) yValues[i - 1] = 0

  plotValues(yValues)

  let largestYIndex = yValues.indexOf(Math.max(...yValues))
  for (let i = largestYIndex + 1; i < yValues.length; i++) {
    if (yValues[i - 1] === 0) {
      yValues[i] = 0
    }
  }
  for (let i = largestYIndex - 1; i >= 0; i--) {
    if (yValues[i + 1] === 0) {
      yValues[i] = 0
    }
  }

  startX = xValues.findIndex((value) => value > 0)
  endX =
    xValues.length -
    xValues
      .slice()
      .reverse()
      .findIndex((value) => value > 0) -
    1

  startY = yValues.findIndex((value) => value > 0)
  endY =
    yValues.length -
    yValues
      .slice()
      .reverse()
      .findIndex((value) => value > 0) -
    1

  let isValid = true

  if (startX < 0 || Math.abs(endX - startX) < 64 || endX - startX > 302) {
    isValid = false
    startX = 107
    endX = 299
  }

  if (startY < 0 || Math.abs(endY - startY) < 32 || endY - startY > 127) {
    isValid = false
    startY = 140
    endY = 183
  }
  const toCrop = { x: startX-12, y: startY, w: endX - startX + 16, h: endY - startY }
  if (isValid) image.crop(toCrop)

  return { isValid, ...toCrop }
}

function cropper(image, color = 255) {
  // Crop image even more
  let yTop = -1
  let yBottom = -1
  // Get highest black pixel
  for (let y = 0; y < image.bitmap.height; y++) {
    for (let x = 0; x < image.bitmap.width; x++) {
      if (image.getPixelColor(x, y) === color) {
        yTop = y
        break
      }
    }
    if (yTop !== -1) break
  }

  // Get most left black pixel
  let xLeft = -1
  for (let x = 0; x < image.bitmap.width; x++) {
    for (let y = yTop; y < image.bitmap.height; y++) {
      if (image.getPixelColor(x, y) === color) {
        xLeft = x
        break
      }
    }
    if (xLeft !== -1) break
  }
  // Get most right black pixel
  let xRight = -1
  for (let x = image.bitmap.width; x > xLeft; x--) {
    for (let y = yTop; y < image.bitmap.height; y++) {
      if (image.getPixelColor(x, y) === color) {
        xRight = x
        break
      }
    }
    if (xRight !== -1) break
  }
  // Get lowest black pixel
  for (let y = image.bitmap.height; y > yTop; y--) {
    for (let x = xLeft; x < xRight; x++) {
      if (image.getPixelColor(x, y) === color) {
        yBottom = y
        break
      }
    }
    if (yBottom !== -1) break
  }
  //yBottom = image.bitmap.height

  // Crop image
  image.crop({ x: xLeft, y: yTop, w: xRight - xLeft, h: Math.min(yBottom - yTop + 1, image.bitmap.height - yTop) })
}

export { detect, cropper }
