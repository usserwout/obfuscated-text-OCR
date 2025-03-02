import { cropper, detect as detectCoupon } from "./stages/couponScanner.js"
import { backgroundRemover, characterAbstractor } from "./stages/backgroundRemover.js"
import { Jimp } from "jimp"
import path from "path"
import { fileURLToPath } from "url"
import { fitCharacters } from "./stages/detectCharacters.js"
import { splitCharacters } from "./stages/splitCharacters.js"
import { predictCharacter, loadImages, findBestMatch } from "./stages/predictSimple.js"

const __filename = fileURLToPath(import.meta.url)
const __dirname = path.dirname(__filename)
let i = 0
async function process(image, silent = false) {
  i++
  let croppedResult = detectCoupon(image, 24)
  if (!croppedResult.isValid) {
    console.log("Could not detect coupon code, trying again with lower tolerance")
    croppedResult = detectCoupon(image, 32)
  }

  if (!croppedResult.isValid) {
    console.log("Failed to detect coupon code")
    return null
  }

  
  if (!silent) image.write(path.join(__dirname, "./processed/cropped.png"))
    
    image = characterAbstractor(image)
    
  return { image, cropInfo: croppedResult }

  //backgroundRemover(image)
  //cropper(image)
  if (!silent) image.write(path.join(__dirname, "./processed/convolution" + i + ".png"))
  await fitCharacters(image)
  if (!silent) image.write(path.join(__dirname, "./processed/charactersBoxed.png"))

  //image.write(path.join(__dirname, "../generated_images/dataset/convolution" + i + ".png"))

  image.write(path.join(__dirname, "./processed/colored.png"))

  const characterImages = splitCharacters(image)

  // featureAbstractor(characterImages[10])

  return characterImages.length

  let promisses = []

  splitCharacters(image).map((img, i) => {
    img.write(path.join(__dirname, `./processed/split/split-${i}.png`))

    promisses.push(findBestMatch(img))
  })

  const characters = (await Promise.all(promisses)).join("")

  image.write(path.join(__dirname, "./processed/out.png"))
  const coupon = characters.slice(0, 4) + "-" + characters.slice(4, 8) + "-" + characters.slice(8, 12)
  console.log("Coupon code: " + coupon)

  return coupon
}

async function processImage(url, silent = false) {
  const isPath = !url.startsWith("https://")


  const file = isPath ? url : url + "width=800&height=450"
  const image = await Jimp.read(file)

  if (image.bitmap.width < 5) return // INVALID IMAGE
  if (image.bitmap.width !== 400 || image.bitmap.height !== 300) {
    image.scaleToFit({ w: 400, h: 300 })
    if (!silent) image.write(path.join(__dirname, "processed/resized.png"))
  }
  if (!silent) image.write(path.join(__dirname, "processed/original.png"))

  // await imageLoadPromise
  return process(image, silent)
}

// Jartex_Coupon_07, Jartex_Coupon_26(1), Jartex_Coupon_06, Jartex_Coupon_31, Jartex_Coupon_56, Jartex_Coupon_30,Jartex_Coupon_60, Jartex_Coupon_33, JartexCoupon_55, Jartex_Coupon_05, Coupon_35,Jartex_Coupon_27
function test() {
  // DA2H-WAE7-N2PY
  processImage(
    path.join(__dirname, "../images/JartexCoupon_55.png")
    //   "https://media.discordapp.net/attachments/1318230639660765234/1320454675333644359/Jartexnetwork_coupon66.png?ex=6769a8bf&is=6768573f&hm=66ffeac156b0bc34c80095a36158c2c4ffb4ce7d7b4555cfd394e06c29a41bc7&=&"
  )
}

export { processImage }
