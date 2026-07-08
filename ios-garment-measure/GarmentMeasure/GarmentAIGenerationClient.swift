import Foundation
import UIKit

enum GarmentAIGenerationError: LocalizedError {
    case missingImage
    case invalidURL
    case invalidImageData
    case server(String)

    var errorDescription: String? {
        switch self {
        case .missingImage:
            "请先添加 T 恤照片"
        case .invalidURL:
            "AI 服务地址无效"
        case .invalidImageData:
            "生成结果不是有效图片"
        case .server(let message):
            message
        }
    }
}

struct GarmentAIGenerationClient {
    let baseURL: URL

    func generatePreview(photo: UIImage, measurements: [GarmentDimension: Double]) async throws -> UIImage {
        guard let imageData = photo.resizedForUpload(maxSide: 1280).jpegData(compressionQuality: 0.82) else {
            throw GarmentAIGenerationError.invalidImageData
        }

        var request = URLRequest(url: baseURL.appendingPathComponent("/api/v1/garment/ai-preview"))
        request.httpMethod = "POST"
        request.timeoutInterval = 240

        let boundary = "Boundary-\(UUID().uuidString)"
        request.setValue("multipart/form-data; boundary=\(boundary)", forHTTPHeaderField: "Content-Type")

        let measurementsPayload = Dictionary(uniqueKeysWithValues: measurements.map { ($0.key.rawValue, $0.value) })
        let measurementsData = try JSONSerialization.data(withJSONObject: measurementsPayload)
        let measurementsText = String(data: measurementsData, encoding: .utf8) ?? "{}"

        var body = Data()
        body.appendMultipartField(name: "measurements", value: measurementsText, boundary: boundary)
        body.appendMultipartFile(
            name: "file",
            filename: "tshirt.jpg",
            mimeType: "image/jpeg",
            data: imageData,
            boundary: boundary
        )
        body.append("--\(boundary)--\r\n")
        request.httpBody = body

        let (data, response) = try await URLSession.shared.data(for: request)
        let status = (response as? HTTPURLResponse)?.statusCode ?? 0
        let json = try JSONSerialization.jsonObject(with: data) as? [String: Any]

        guard (200..<300).contains(status) else {
            let detail = (json?["detail"] as? String) ?? HTTPURLResponse.localizedString(forStatusCode: status)
            throw GarmentAIGenerationError.server(detail)
        }

        guard let b64 = json?["image_base64"] as? String,
              let resultData = Data(base64Encoded: b64),
              let image = UIImage(data: resultData)
        else {
            throw GarmentAIGenerationError.invalidImageData
        }
        return image
    }

    func generateModel(photo: UIImage, measurements: [GarmentDimension: Double]) async throws -> GeneratedGarmentModelResult {
        guard let imageData = photo.resizedForUpload(maxSide: 1280).jpegData(compressionQuality: 0.82) else {
            throw GarmentAIGenerationError.invalidImageData
        }

        var request = URLRequest(url: baseURL.appendingPathComponent("/api/v1/garment/model"))
        request.httpMethod = "POST"
        request.timeoutInterval = 1800

        let boundary = "Boundary-\(UUID().uuidString)"
        request.setValue("multipart/form-data; boundary=\(boundary)", forHTTPHeaderField: "Content-Type")

        let measurementsPayload = Dictionary(uniqueKeysWithValues: measurements.map { ($0.key.rawValue, $0.value) })
        let measurementsData = try JSONSerialization.data(withJSONObject: measurementsPayload)
        let measurementsText = String(data: measurementsData, encoding: .utf8) ?? "{}"

        var body = Data()
        body.appendMultipartField(name: "measurements", value: measurementsText, boundary: boundary)
        body.appendMultipartFile(
            name: "file",
            filename: "tshirt.jpg",
            mimeType: "image/jpeg",
            data: imageData,
            boundary: boundary
        )
        body.append("--\(boundary)--\r\n")
        request.httpBody = body

        let (data, response) = try await URLSession.shared.data(for: request)
        let status = (response as? HTTPURLResponse)?.statusCode ?? 0

        guard (200..<300).contains(status) else {
            let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
            let detail = (json?["detail"] as? String) ?? HTTPURLResponse.localizedString(forStatusCode: status)
            throw GarmentAIGenerationError.server(detail)
        }

        let responsePayload = try JSONDecoder().decode(GarmentModelResponse.self, from: data)
        if let modelBase64 = responsePayload.modelBase64,
           let modelData = Data(base64Encoded: modelBase64) {
            let ext = responsePayload.fileExt?.isEmpty == false ? responsePayload.fileExt! : "obj"
            let fileName = responsePayload.fileName?.isEmpty == false ? responsePayload.fileName! : "garment.\(ext)"
            let localURL = FileManager.default.temporaryDirectory
                .appendingPathComponent(UUID().uuidString)
                .appendingPathExtension((fileName as NSString).pathExtension.isEmpty ? ext : (fileName as NSString).pathExtension)
            try modelData.write(to: localURL, options: [.atomic])
            return GeneratedGarmentModelResult(
                mesh: responsePayload.mesh,
                localModelURL: localURL,
                provider: responsePayload.provider ?? "",
                jobId: responsePayload.jobId
            )
        }
        if let modelPath = responsePayload.modelUrl,
           let downloadURL = resolvedURL(for: modelPath) {
            let (downloadedFileURL, modelResponse) = try await URLSession.shared.download(from: downloadURL)
            let modelStatus = (modelResponse as? HTTPURLResponse)?.statusCode ?? 0
            guard (200..<300).contains(modelStatus) else {
                throw GarmentAIGenerationError.server("模型文件下载失败：\(modelStatus)")
            }
            let ext = responsePayload.fileExt?.isEmpty == false
                ? responsePayload.fileExt!
                : (downloadURL.pathExtension.isEmpty ? "glb" : downloadURL.pathExtension)
            let localURL = FileManager.default.temporaryDirectory
                .appendingPathComponent(UUID().uuidString)
                .appendingPathExtension(ext)
            if FileManager.default.fileExists(atPath: localURL.path) {
                try FileManager.default.removeItem(at: localURL)
            }
            try FileManager.default.moveItem(at: downloadedFileURL, to: localURL)
            return GeneratedGarmentModelResult(
                mesh: responsePayload.mesh,
                localModelURL: localURL,
                provider: responsePayload.provider ?? "",
                jobId: responsePayload.jobId
            )
        }
        return GeneratedGarmentModelResult(
            mesh: responsePayload.mesh,
            localModelURL: nil,
            provider: responsePayload.provider ?? "",
            jobId: responsePayload.jobId
        )
    }

    private func resolvedURL(for path: String) -> URL? {
        if let absolute = URL(string: path), absolute.scheme != nil {
            return absolute
        }
        guard var components = URLComponents(url: baseURL, resolvingAgainstBaseURL: false) else {
            return nil
        }
        components.path = path.hasPrefix("/") ? path : "/" + path
        components.query = nil
        components.fragment = nil
        return components.url
    }
}

struct GarmentModelResponse: Codable {
    let jobId: String
    let modelUrl: String?
    let mesh: GeneratedGarmentMesh?
    let provider: String?
    let modelBase64: String?
    let fileName: String?
    let fileExt: String?

    enum CodingKeys: String, CodingKey {
        case jobId = "job_id"
        case modelUrl = "model_url"
        case mesh
        case provider
        case modelBase64 = "model_base64"
        case fileName = "file_name"
        case fileExt = "file_ext"
    }
}

struct GeneratedGarmentModelResult {
    let mesh: GeneratedGarmentMesh?
    let localModelURL: URL?
    let provider: String
    let jobId: String
}

struct GeneratedGarmentMesh: Codable {
    let vertices: [[Float]]
    let triangles: [[Int32]]
    let baseColor: [Float]
    let metadata: [String: StringValue]

    enum CodingKeys: String, CodingKey {
        case vertices
        case triangles
        case baseColor = "base_color"
        case metadata
    }
}

enum StringValue: Codable {
    case string(String)
    case int(Int)
    case double(Double)

    init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if let value = try? container.decode(String.self) {
            self = .string(value)
        } else if let value = try? container.decode(Int.self) {
            self = .int(value)
        } else {
            self = .double((try? container.decode(Double.self)) ?? 0)
        }
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        switch self {
        case .string(let value): try container.encode(value)
        case .int(let value): try container.encode(value)
        case .double(let value): try container.encode(value)
        }
    }
}

private extension Data {
    mutating func append(_ string: String) {
        append(Data(string.utf8))
    }

    mutating func appendMultipartField(name: String, value: String, boundary: String) {
        append("--\(boundary)\r\n")
        append("Content-Disposition: form-data; name=\"\(name)\"\r\n\r\n")
        append("\(value)\r\n")
    }

    mutating func appendMultipartFile(name: String, filename: String, mimeType: String, data: Data, boundary: String) {
        append("--\(boundary)\r\n")
        append("Content-Disposition: form-data; name=\"\(name)\"; filename=\"\(filename)\"\r\n")
        append("Content-Type: \(mimeType)\r\n\r\n")
        append(data)
        append("\r\n")
    }
}

private extension UIImage {
    func resizedForUpload(maxSide: CGFloat) -> UIImage {
        let longestSide = max(size.width, size.height)
        guard longestSide > maxSide else { return self }

        let scale = maxSide / longestSide
        let targetSize = CGSize(
            width: max(1, floor(size.width * scale)),
            height: max(1, floor(size.height * scale))
        )
        let format = UIGraphicsImageRendererFormat()
        format.scale = 1
        format.opaque = true
        let renderer = UIGraphicsImageRenderer(size: targetSize, format: format)
        return renderer.image { _ in
            UIColor.white.setFill()
            UIBezierPath(rect: CGRect(origin: .zero, size: targetSize)).fill()
            draw(in: CGRect(origin: .zero, size: targetSize))
        }
    }
}
