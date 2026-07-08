import UIKit

enum GarmentTextureProcessor {
    static func makeTexture(from image: UIImage, size: CGFloat = 768) -> UIImage {
        let targetSize = CGSize(width: size, height: size)
        let prepared = image.aspectFitOnCanvas(size: targetSize, fill: .clear)
        guard let cgImage = prepared.cgImage else { return prepared }

        let width = cgImage.width
        let height = cgImage.height
        let bytesPerPixel = 4
        let bytesPerRow = width * bytesPerPixel
        var pixels = [UInt8](repeating: 0, count: height * bytesPerRow)

        guard let context = CGContext(
            data: &pixels,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: bytesPerRow,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        ) else {
            return prepared
        }

        context.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))
        let background = estimatedBackgroundColor(pixels: pixels, width: width, height: height, bytesPerRow: bytesPerRow)

        var minX = width
        var minY = height
        var maxX = 0
        var maxY = 0

        for y in 0..<height {
            for x in 0..<width {
                let index = y * bytesPerRow + x * bytesPerPixel
                let r = pixels[index]
                let g = pixels[index + 1]
                let b = pixels[index + 2]
                let alpha = foregroundAlpha(r: r, g: g, b: b, background: background)
                pixels[index + 3] = alpha

                if alpha > 28 {
                    minX = min(minX, x)
                    minY = min(minY, y)
                    maxX = max(maxX, x)
                    maxY = max(maxY, y)
                }
            }
        }

        guard minX < maxX, minY < maxY else { return prepared }

        let padding = Int(size * 0.06)
        minX = max(0, minX - padding)
        minY = max(0, minY - padding)
        maxX = min(width - 1, maxX + padding)
        maxY = min(height - 1, maxY + padding)

        guard let maskedCG = context.makeImage(),
              let croppedCG = maskedCG.cropping(to: CGRect(x: minX, y: minY, width: maxX - minX + 1, height: maxY - minY + 1))
        else {
            return prepared
        }

        return UIImage(cgImage: croppedCG).aspectFitOnCanvas(size: targetSize, fill: .clear)
    }

    private static func estimatedBackgroundColor(
        pixels: [UInt8],
        width: Int,
        height: Int,
        bytesPerRow: Int
    ) -> RGB {
        var samples: [RGB] = []
        let strideX = max(1, width / 12)
        let strideY = max(1, height / 12)

        for x in stride(from: 0, to: width, by: strideX) {
            samples.append(sample(pixels, x: x, y: 0, bytesPerRow: bytesPerRow))
            samples.append(sample(pixels, x: x, y: height - 1, bytesPerRow: bytesPerRow))
        }

        for y in stride(from: 0, to: height, by: strideY) {
            samples.append(sample(pixels, x: 0, y: y, bytesPerRow: bytesPerRow))
            samples.append(sample(pixels, x: width - 1, y: y, bytesPerRow: bytesPerRow))
        }

        let count = max(1, samples.count)
        let r = samples.reduce(0) { $0 + Int($1.r) } / count
        let g = samples.reduce(0) { $0 + Int($1.g) } / count
        let b = samples.reduce(0) { $0 + Int($1.b) } / count
        return RGB(UInt8(r), UInt8(g), UInt8(b))
    }

    private static func sample(_ pixels: [UInt8], x: Int, y: Int, bytesPerRow: Int) -> RGB {
        let index = y * bytesPerRow + x * 4
        return RGB(pixels[index], pixels[index + 1], pixels[index + 2])
    }

    private static func foregroundAlpha(r: UInt8, g: UInt8, b: UInt8, background: RGB) -> UInt8 {
        let dr = abs(Int(r) - Int(background.r))
        let dg = abs(Int(g) - Int(background.g))
        let db = abs(Int(b) - Int(background.b))
        let distance = Double(dr + dg + db) / 3.0
        let saturation = Double(max(r, g, b) - min(r, g, b))
        let threshold = max(22.0, min(58.0, saturation * 0.35 + 26.0))

        if distance < threshold {
            return 0
        }

        let alpha = min(255.0, (distance - threshold) * 5.2)
        return UInt8(max(0, alpha))
    }
}

private struct RGB {
    let r: UInt8
    let g: UInt8
    let b: UInt8

    init(_ r: UInt8, _ g: UInt8, _ b: UInt8) {
        self.r = r
        self.g = g
        self.b = b
    }
}

private extension UIImage {
    func aspectFitOnCanvas(size targetSize: CGSize, fill color: UIColor) -> UIImage {
        let renderer = UIGraphicsImageRenderer(size: targetSize)
        return renderer.image { context in
            color.setFill()
            context.fill(CGRect(origin: .zero, size: targetSize))

            let imageAspect = size.width / max(1, size.height)
            let canvasAspect = targetSize.width / max(1, targetSize.height)
            let drawSize: CGSize

            if imageAspect > canvasAspect {
                drawSize = CGSize(width: targetSize.width, height: targetSize.width / imageAspect)
            } else {
                drawSize = CGSize(width: targetSize.height * imageAspect, height: targetSize.height)
            }

            let drawRect = CGRect(
                x: (targetSize.width - drawSize.width) / 2,
                y: (targetSize.height - drawSize.height) / 2,
                width: drawSize.width,
                height: drawSize.height
            )
            draw(in: drawRect)
        }
    }
}
