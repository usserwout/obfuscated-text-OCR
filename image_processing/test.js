import { processImage } from "./fitCharacters.js"
import path from 'path'
import fs from 'fs'
import { fileURLToPath } from "url"

const __filename = fileURLToPath(import.meta.url)
const __dirname = path.dirname(__filename)


const data = [
  ["Coupon_04.png", "WQ77-RAUU-5AM3"],
  ["Coupon_35.png", "QS45-WTEQ-46FM"],
  ["Jartex_Coupon_01.png", "DA2H-WAE7-N2PY"],
  ["Jartex_Coupon_02.png", "6VN3-YKKV-HH33"],
  ["Jartex_Coupon_05.png", "T7AW-85KA-KYSS"],
  ["Jartex_Coupon_12.png", "9QXX-WAP9-ZDWS"],
  ["Jartex_Coupon_16.png", "-HC4C-7X47-SMSZ-"],
  ["Jartex_Coupon_24.png", "MC25-HJE6-7PKX"],
  ["JartexCoupon_55.png", "PEYM-MM9Q-YA4X"],
  ["Jartex_Coupon_25.png", "MA3V-P54R-4YVY"],
]


async function test() {

  let correct = 0
  let totDist = 0
  let dist = 0
  for (const [file, code] of data) {
    console.log(`Processing ${file}`)
   
    const coupon = await processImage(
      path.join(__dirname, `../images/${file}`)
    )

    console.log(`'${coupon}' === '${code}' `, coupon === code)
    if (coupon === code) {
      correct++
    } 

    // calculate ham distance
    for (let i = 0; i < coupon.length; i++) {
      if (coupon[i] === code[i]) {
        dist++
      }
      totDist++
    }

  }
  console.log(`Correct: ${correct}/${data.length}`)
  console.log("Accuracy: ", dist / totDist)
}



async function largeTest(){
  let successes = 0

  const folderPath = "/Users/usserwout/Documents/JavaScript_Projects/projects/december_giveaways"
  const folders = ["./images"].map(f => path.join(folderPath, f))
  let total = 0
  for(const folder of folders) {
    const files = fs.readdirSync(folder).filter(f => f.endsWith(".png"))

    for(const file of files) {
      total++
      console.log(`Processing: ${file}`)
      const chars = await processImage(
        path.join(folder, file)
      )
      console.log("Characters: ", chars)

      if (chars === 12 || chars === 9) successes++
    }
  }

  console.log(`Successes: ${successes}/${total} (${(successes / total) * 100}%)`);
}

async function testSingle() {
  const file =
    // "/Users/usserwout/Documents/JavaScript_Projects/projects/december_giveaways/images/jartexnetwork_coupon33 2.png"
    // "/Users/usserwout/Documents/JavaScript_Projects/projects/december_giveaways/images/Jartex_Coupon_07.png"
     "/Users/usserwout/Documents/JavaScript_Projects/projects/december_giveaways/images/jartexnetworkcoupon2024-17 2.png"
    // "/Users/usserwout/Documents/JavaScript_Projects/projects/december_giveaways/images/Jartex_Coupon_33.png"
    // "/Users/usserwout/Documents/JavaScript_Projects/projects/december_giveaways/images/Jartex_Coupon_26(1).png"
  const chars = await processImage(file)
  console.log("Characters: ", chars)
}

//largeTest()
 testSingle()