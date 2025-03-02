
import { kmeans } from "ml-kmeans"
import clustering from "density-clustering"
import assert from "assert"

const colors = [
  ...new Set([
    0xf00000ff, 0x00f000ff, 0x0000f0ff, 0xf0f000ff, 0x00f0f0ff, 0xf000f0ff, 0x0de0f0ff, 0x008888ff, 0x888800ff,
    0x880088ff, 0x008800ff, 0x880000ff, 0x000088ff, 0x808080ff, 0xffa500ff, 0x800080ff, 0x00ff00ff, 0xff00ffff,
    0x00ffffff, 0xff69b4ff, 0x8a2be2ff, 0x5f9ea0ff, 0xd2691eff, 0xff4500ff, 0x2e8b57ff, 0xdaa520ff, 0x4b0082ff,
    0x7fff00ff, 0xff6347ff, 0x4682b4ff, 0x9acd32ff, 0x6a5acdff, 0x20b2aaff, 0xff1493ff, 0x32cd32ff, 0x8b0000ff,
    0x00ced1ff, 0xffd700ff, 0xadff2fff, 0x1e90ffff, 0xff4500ff, 0x2e8b57ff, 0x8b4513ff, 0xff6347ff, 0x4682b4ff,
    0x9acd32ff, 0x6a5acdff, 0x20b2aaff, 0xff1493ff, 0x32cd32ff,
  ]),
]

function runKMeans(separators, image, numClusters = 4) {

  let clusterCount = 0
  for (let i = 0; i < separators.length; i++) {
    let samples = []

    // Extract sample points within the current section
    for (let x = separators[i][0]; x <= separators[i][1]; x++) {
      for (let y = 0; y < image.bitmap.height; y++) {
        const color = image.getPixelColor(x, y)
        if (color === 0x000000ff) {
          samples.push(x) // Use only the x-value for 1D representation
        }
      }
    }

    console.log(`Section ${i}: ${samples.length} samples`)

    // Use K-Means to cluster the points within the current section
    const clusters = kmeans(
      samples.map((x) => [x]),
      numClusters,
      {
        maxIterations: 10_000,
        tolerance: 1e-12,
        initialization: "kmeans++",
      }
    )

    console.log(`Section ${i}: ${clusters.clusters.length} clusters`)


    // Assign each pixel to the nearest cluster within the current section
    for (let x = separators[i][0]; x <= separators[i][1]; x++) {
      for (let y = 0; y < image.bitmap.height; y++) {
        const color = image.getPixelColor(x, y)
        if (color === 0x000000ff) {
          const clusterIndex = clusters.clusters[samples.indexOf(x)]
          if (clusterIndex !== undefined) {
            assert(clusterCount < colors.length, "Too many clusters")
            image.setPixelColor(colors[clusterCount], x, y) // Color each cluster differently
          }
        }
      }
      clusterCount++
    }
  }
}

function runDBSCAN(separators, image) {
  const epsilon = 2 // Maximum distance between two samples for them to be considered as in the same neighborhood
  const minPoints = 3 // Minimum number of points to generate a single cluster
  let clusterCount = 0
  // Clone the image and apply erosion to make the lines thinner
  const erodedImage = image.clone()
  applyHorizontalErosion(erodedImage)


  for (let i = 0; i < separators.length; i++) {
    let samples = []

    // Extract sample points within the current section from the eroded image
    for (let x = separators[i][0]; x <= separators[i][1]; x++) {
      for (let y = 0; y < erodedImage.bitmap.height; y++) {
        const color = erodedImage.getPixelColor(x, y)
        if (color !== 0xffffffff) {
          samples.push(x) // Use only the x-value for 1D representation
        }
      }
    }

    console.log(`Section ${i}: ${samples.length} samples`)

    // Use DBSCAN to cluster the points within the current section
    const dbscan = new clustering.DBSCAN()
    let clusters
    
    clusters = dbscan.run(
      samples.map((x) => [x]),
      epsilon,
      minPoints
    )
    if(clusters.length > 5) {
     samples = []
      for (let x = separators[i][0]; x <= separators[i][1]; x++) {
        for (let y = 0; y < image.bitmap.height; y++) {
          const color = image.getPixelColor(x, y)
          if (color !== 0xffffffff) {
            samples.push(x) // Use only the x-value for 1D representation
          }
        }
      }


      clusters = dbscan.run(
        samples.map((x) => [x]),
        epsilon,
        minPoints
      )
    }
    

    // Assign each pixel to the nearest cluster within the current section
    let clusterIndex = 0
    for (let cluster of clusters) {
      for (let point of cluster) {
        const x = samples[point]
        for (let y = 0; y < image.bitmap.height; y++) {
          const color = image.getPixelColor(x, y)
          if (color !== 0xffffffff) {
            assert(clusterCount < colors.length, "Too many clusters")
            image.setPixelColor(colors[clusterCount], x, y)
          }
        }
      }
      clusterCount++
      clusterIndex++
    }

  }
  // Color in the left over black points
  const queue = [];

  // Initialize the queue with all black pixels
  for (let y = 0; y < image.bitmap.height; y++) {
    for (let x = 0; x < image.bitmap.width; x++) {
      if (image.getPixelColor(x, y) !== 0xffffffff) {
        queue.push({ x, y });
      }
    }
  }


  let doNotRepeat = []
  // Process the queue
  while (queue.length > 0) {
    const { x, y } = queue.shift();
    const color = image.getPixelColor(x, y);

    if (color === 0x000000ff) {
      // Get the color of neighboring pixels that aren't black or white
      const neighbors = [];
      if (x > 0) {
        neighbors.push(image.getPixelColor(x - 1, y));
      }
      if (x < image.bitmap.width - 1) {
        neighbors.push(image.getPixelColor(x + 1, y));
      }
      if (y > 0) {
        neighbors.push(image.getPixelColor(x, y - 1));
      }
      if (y < image.bitmap.height - 1) {
        neighbors.push(image.getPixelColor(x, y + 1));
      }

      const newColor = neighbors.find((c) => c !== 0xffffffff && c !== 0x000000ff);
      if (newColor) {
        image.setPixelColor(newColor, x, y);
      } else if (queue.length > 0 && !doNotRepeat.find((p) => p.x === x && p.y === y)) {
        doNotRepeat.push({ x, y })
        queue.push({ x, y })
      }
    }
  }

  return clusterCount
}

function applyHorizontalErosion(image) {
  const width = image.bitmap.width
  const height = image.bitmap.height
  const newImage = image.clone()

  for (let x = 1; x < width - 1; x++) {
    for (let y = 0; y < height; y++) {
      const color = image.getPixelColor(x, y)
      if (color === 0x000000ff) {
        // Check the left and right neighbors
        let neighbors = 0
        if (image.getPixelColor(x - 1, y) === 0x000000ff) {
          neighbors++
        }
        if (image.getPixelColor(x + 1, y) === 0x000000ff) {
          neighbors++
        }
        // If fewer than 2 neighbors are black, set the pixel to white
        if (neighbors < 2) {
          newImage.setPixelColor(0xffffffff, x, y)
        }
      }
    }
  }

  // Copy the eroded image back to the original image
  image.scan(0, 0, width, height, function (x, y, idx) {
    this.bitmap.data[idx + 0] = newImage.bitmap.data[idx + 0]
    this.bitmap.data[idx + 1] = newImage.bitmap.data[idx + 1]
    this.bitmap.data[idx + 2] = newImage.bitmap.data[idx + 2]
    this.bitmap.data[idx + 3] = newImage.bitmap.data[idx + 3]
  })
}


export {
  runKMeans,
  runDBSCAN,
  applyHorizontalErosion
}