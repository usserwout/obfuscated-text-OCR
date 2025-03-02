

function walkingMeanFilter(values, windowSize = 5) {
  const smoothed = []
  for (let i = 0; i < values.length; i++) {
    let sum = 0
    let count = 0
    for (let j = -windowSize; j <= windowSize; j++) {
      const idx = i + j
      if (idx >= 0 && idx < values.length) {
        sum += values[idx]
        count++
      }
    }
    smoothed.push(sum / count)
  }
  return smoothed
}

function gaussianFilter(values, windowSize = 5, sigma = 1) {
  const gauss = []
  const factor = 1 / (Math.sqrt(2 * Math.PI) * sigma)
  const denominator = 2 * sigma * sigma

  // Generate Gaussian kernel
  for (let i = -windowSize; i <= windowSize; i++) {
    gauss.push(factor * Math.exp(-(i * i) / denominator))
  }

  // Apply Gaussian filter
  const smoothed = []
  for (let i = 0; i < values.length; i++) {
    let sum = 0
    let weight = 0
    for (let j = -windowSize; j <= windowSize; j++) {
      const idx = i + j
      if (idx >= 0 && idx < values.length) {
        sum += values[idx] * gauss[j + windowSize]
        weight += gauss[j + windowSize]
      }
    }
    smoothed.push(sum / weight)
  }
  return smoothed
}

export {
  walkingMeanFilter,
  gaussianFilter
}