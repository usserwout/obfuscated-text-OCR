import fs from "fs"
import path from "path"
import { fileURLToPath } from "url"
import {Jimp} from "jimp"
import model from "./model/characters.json" assert { type: "json" }
import { applyHorizontalErosion } from "./util/clustering.js"

const __filename = fileURLToPath(import.meta.url)
const __dirname = path.dirname(__filename)

let input_images = null

async function loadImages(){
  input_images = {}
  const characters = fs
    .readdirSync(path.join(__dirname, "../../images/characters"))
    .filter((file) => file !== ".DS_Store")

  for (const character of characters) {
     const images = fs
       .readdirSync(path.join(__dirname, "../../images/characters", character))
       .filter((file) => file !== ".DS_Store")
    let promisses = []

     for (const image of images) {
        const imagePath = path.join(__dirname, "../../images/characters", character, image)
        promisses.push(Jimp.read(imagePath))
     }
     const res = await Promise.all(promisses)
     input_images[character] = res
      input_images[character]
      
  }
}





async function findBestMatch(image) {
  
  
  if(input_images === null){
    throw new Error("Images not loaded")
  }
  applyHorizontalErosion(image)
  let bestScore = Infinity
  let bestCharacter = ""
  let scores = {}

  // Create a 32x32 matrix of the image
  // let matrix = Array.from({ length: 32 }, () => Array(32).fill(0))

  let seen = new Set()
  for (let character in input_images) {
    
     for(let img of input_images[character]){

        let difference = 0

        for (let x = 0; x < image.bitmap.width; x++) {
          for (let y = 0; y < image.bitmap.height; y++) {
            const foundColor = image.getPixelColor(x, y)
            const expectedColor = img.getPixelColor(x, y)
            if (foundColor === 0x000000ff && expectedColor === 0xffffffff) {
              difference += 1 
              seen.add(`${x},${y}`)
              if (seen.has(`${x-1},${y}`)) difference += 1
              if (seen.has(`${x},${y-1}`)) difference += 1
            } else if (foundColor === 0x000000ff && expectedColor === 0x000000ff) {
  
            }
          }
        
        }

        scores[character] = difference
      
        //const difference = jimp.diff(image, img).percent
        // console.time("hash")
        // const hashCmp = jimp.compareHashes(image.pHash(), img.pHash())
        // console.timeEnd("hash")
        //  const difference = jimp.distance(image, img)

        //console.log(`${i} Character: ${character}, Difference: ${difference}, Distance: ${distance}`)
        if (difference < bestScore) {
          bestScore = difference
          bestCharacter = character
        }

      
     }
  }
//  if (bestCharacter === "X" || bestCharacter === "K") console.log(scores, bestCharacter)
  return bestCharacter
}


function predictCharacter(image) {

  let bestScore = -Infinity
  let bestCharacter = ""
  let scores = {}

  for (let character in model) {
    let matrix = model[character]
    let score = 0

    let charMatch = 0
    let totCharMatch = 0
    let expectedMatch = 0
    
    for (let y = 0; y < image.bitmap.height; y++) {
      for (let x = 0; x < image.bitmap.width; x++) {
        const color = image.getPixelColor(x, y)
        let val = matrix[y][x]

        if (color !== 0xffffffff) {
          if (val > 0) {
            // Black and found black
            charMatch++
            expectedMatch++
            score += val
          } else {
            // There's supposed to be a white pixel but found black
            score += val
          }
          
          totCharMatch++

        } else {
          if (val < 0) {
            // White and found white
            score -= val 
          } else {
            expectedMatch++
            // There's supposed to be a black pixel but found white
            score -= val / 4
          }
        }
      }

      if (score + (32 - y) * 32 < bestScore) {
        break
      }
    }

    // console.log(
    //   `Character: ${character}, Score: ${score}, CharMatch: ${charMatch}, TotCharMatch: ${totCharMatch}, ExpectedMatch: ${expectedMatch} | ${charMatch/totCharMatch} | ${charMatch/expectedMatch}`
    // )
    score = charMatch / totCharMatch + (charMatch / expectedMatch) 


    if (score > bestScore) {
      bestScore = score
      bestCharacter = character
    }

    scores[character] = score
  }
//console.log("-----------------------------------------");
  
  // console.log(Object.entries(scores).sort((a, b) => b[1] - a[1]), bestCharacter)
  return bestCharacter
}

async function test() {
  const files = fs.readdirSync(path.join(__dirname, "../../images/test-set")).filter((file) => file !== ".DS_Store")
  let correct = 0
  for (const file of files) {
    const imagePath = path.join(__dirname, "../../images/test-set", file)
    const correctSymbol = file[0]
    const image = await Jimp.read(imagePath)
    const character = predictCharacter(image)
    console.log(`(${correctSymbol} -> ${character})`)
    if (character === correctSymbol) correct++
  }

  console.log(`Accuracy: ${correct}/${files.length}`)
}


export {
  predictCharacter,
  loadImages,
  findBestMatch
}