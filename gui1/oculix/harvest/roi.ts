import type { Page } from "@playwright/test"

export interface BoundingBox {
  x: number
  y: number
  width: number
  height: number
}

export function intersectBoxes(
  a: BoundingBox,
  b: BoundingBox
): BoundingBox | null {
  const left = Math.max(a.x, b.x)
  const top = Math.max(a.y, b.y)
  const right = Math.min(a.x + a.width, b.x + b.width)
  const bottom = Math.min(a.y + a.height, b.y + b.height)

  if (left >= right || top >= bottom) {
    return null
  }

  return {
    x: left,
    y: top,
    width: right - left,
    height: bottom - top,
  }
}

export function clampToViewport(
  box: BoundingBox,
  viewportWidth: number,
  viewportHeight: number
): BoundingBox | null {
  const clamped = intersectBoxes(box, {
    x: 0,
    y: 0,
    width: viewportWidth,
    height: viewportHeight,
  })
  if (!clamped || clamped.width <= 0 || clamped.height <= 0) {
    return null
  }
  return clamped
}

export function expandBox(box: BoundingBox, padding: number): BoundingBox {
  return {
    x: box.x - padding,
    y: box.y - padding,
    width: box.width + 2 * padding,
    height: box.height + 2 * padding,
  }
}

export async function computeRoi(
  page: Page,
  idleBuf: Buffer,
  activeBuf: Buffer,
  elementBBox: BoundingBox,
  padding = 2
): Promise<BoundingBox> {
  const idleB64 = idleBuf.toString("base64")
  const activeB64 = activeBuf.toString("base64")

  const { diffBox, viewportWidth, viewportHeight } = await page.evaluate(
    async ([idleDataUrl, activeDataUrl]) => {
      const img1 = new Image()
      const img2 = new Image()

      await Promise.all([
        new Promise<void>((resolve) => {
          img1.onload = () => resolve()
          img1.src = idleDataUrl
        }),
        new Promise<void>((resolve) => {
          img2.onload = () => resolve()
          img2.src = activeDataUrl
        }),
      ])

      const width = img1.width
      const height = img1.height

      const canvas1 = document.createElement("canvas")
      canvas1.width = width
      canvas1.height = height
      const ctx1 = canvas1.getContext("2d")!
      ctx1.drawImage(img1, 0, 0)

      const canvas2 = document.createElement("canvas")
      canvas2.width = width
      canvas2.height = height
      const ctx2 = canvas2.getContext("2d")!
      ctx2.drawImage(img2, 0, 0)

      const data1 = ctx1.getImageData(0, 0, width, height).data
      const data2 = ctx2.getImageData(0, 0, width, height).data

      let minX = width
      let minY = height
      let maxX = 0
      let maxY = 0
      let hasDiff = false

      for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
          const idx = (y * width + x) * 4
          const r1 = data1[idx]
          const g1 = data1[idx + 1]
          const b1 = data1[idx + 2]
          const a1 = data1[idx + 3]
          const r2 = data2[idx]
          const g2 = data2[idx + 1]
          const b2 = data2[idx + 2]
          const a2 = data2[idx + 3]

          if (r1 !== r2 || g1 !== g2 || b1 !== b2 || a1 !== a2) {
            hasDiff = true
            if (x < minX) minX = x
            if (y < minY) minY = y
            if (x > maxX) maxX = x
            if (y > maxY) maxY = y
          }
        }
      }

      if (!hasDiff) {
        return { diffBox: null, viewportWidth: width, viewportHeight: height }
      }

      return {
        diffBox: {
          x: minX,
          y: minY,
          width: maxX - minX + 1,
          height: maxY - minY + 1,
        },
        viewportWidth: width,
        viewportHeight: height,
      }
    },
    [`data:image/png;base64,${idleB64}`, `data:image/png;base64,${activeB64}`]
  )

  let roi: BoundingBox | null

  if (!diffBox) {
    roi = expandBox(elementBBox, padding)
  } else {
    const intersected = intersectBoxes(diffBox, elementBBox)
    if (!intersected) {
      roi = expandBox(elementBBox, padding)
    } else {
      roi = expandBox(intersected, padding)
    }
  }

  const clamped = clampToViewport(roi!, viewportWidth, viewportHeight)
  if (!clamped) {
    return { x: 0, y: 0, width: 1, height: 1 }
  }

  return clamped
}

export function buffersEqual(a: Buffer, b: Buffer): boolean {
  if (a.length !== b.length) return false
  for (let i = 0; i < a.length; i++) {
    if (a[i] !== b[i]) return false
  }
  return true
}
