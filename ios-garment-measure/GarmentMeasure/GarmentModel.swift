import Foundation
import UIKit

enum GarmentDimension: String, CaseIterable, Identifiable {
    case bodyLength
    case shoulderWidth
    case chestWidth
    case hemWidth
    case leftSleeveLength
    case rightSleeveLength
    case neckWidth

    var id: String { rawValue }

    var title: String {
        switch self {
        case .bodyLength: "衣长"
        case .shoulderWidth: "肩宽"
        case .chestWidth: "胸宽"
        case .hemWidth: "下摆宽"
        case .leftSleeveLength: "左袖长"
        case .rightSleeveLength: "右袖长"
        case .neckWidth: "领宽"
        }
    }

    var fallbackCentimeters: Double {
        switch self {
        case .bodyLength: 68
        case .shoulderWidth: 45
        case .chestWidth: 52
        case .hemWidth: 50
        case .leftSleeveLength: 22
        case .rightSleeveLength: 22
        case .neckWidth: 18
        }
    }
}

struct MeasuredSegment: Identifiable {
    let id = UUID()
    let dimension: GarmentDimension
    let centimeters: Double
    let createdAt = Date()
}

struct GarmentModelArchive: Codable, Identifiable {
    let id: String
    let createdAt: Date
    let jobId: String
    let provider: String
    let modelFileName: String
    let thumbnailFileName: String?
    let measurements: [String: Double]

    var title: String {
        Self.titleFormatter.string(from: createdAt)
    }

    var subtitle: String {
        let body = measurements[GarmentDimension.bodyLength.rawValue] ?? GarmentDimension.bodyLength.fallbackCentimeters
        let chest = measurements[GarmentDimension.chestWidth.rawValue] ?? GarmentDimension.chestWidth.fallbackCentimeters
        return "衣长 \(body.formatted(.number.precision(.fractionLength(1)))) cm / 胸宽 \(chest.formatted(.number.precision(.fractionLength(1)))) cm"
    }

    private static let titleFormatter: DateFormatter = {
        let formatter = DateFormatter()
        formatter.dateFormat = "yyyy-MM-dd HH:mm"
        return formatter
    }()
}

@MainActor
final class GarmentMeasurementStore: ObservableObject {
    @Published var selectedDimension: GarmentDimension = .bodyLength
    @Published private(set) var values: [GarmentDimension: Double] = [:]
    @Published private(set) var history: [MeasuredSegment] = []
    @Published var latestDistanceCentimeters: Double?
    @Published var garmentPhoto: UIImage?
    @Published var garmentTexture: UIImage?
    @Published var generatedPreview: UIImage?
    @Published var generatedMesh: GeneratedGarmentMesh?
    @Published var generatedModelURL: URL?
    @Published private(set) var modelArchives: [GarmentModelArchive] = []
    @Published var activeArchiveID: String?
    @Published var generationStatus: String = ""
    @Published var isGeneratingPreview = false
    @Published private(set) var previewRevision = 0

    init() {
        loadModelArchives()
        if let latest = modelArchives.first {
            selectArchive(latest)
        }
    }

    func setLatestDistance(_ centimeters: Double) {
        latestDistanceCentimeters = centimeters
    }

    func saveLatestDistance() {
        guard let latestDistanceCentimeters else { return }
        values[selectedDimension] = latestDistanceCentimeters
        history.insert(
            MeasuredSegment(dimension: selectedDimension, centimeters: latestDistanceCentimeters),
            at: 0
        )
    }

    func value(for dimension: GarmentDimension) -> Double {
        values[dimension] ?? dimension.fallbackCentimeters
    }

    func reset() {
        values.removeAll()
        history.removeAll()
        latestDistanceCentimeters = nil
    }

    func setGarmentPhoto(_ image: UIImage?) {
        let normalized = image?.resizedForWorkingCopy(maxSide: 1600)
        garmentPhoto = normalized
        garmentTexture = normalized.map { GarmentTextureProcessor.makeTexture(from: $0) }
        generatedPreview = nil
        generatedMesh = nil
        generatedModelURL = nil
        activeArchiveID = nil
        generationStatus = ""
        bumpPreviewRevision()
    }

    func regenerateTexture() {
        guard let garmentPhoto else { return }
        garmentTexture = GarmentTextureProcessor.makeTexture(from: garmentPhoto)
    }

    func measurementsSnapshot() -> [GarmentDimension: Double] {
        Dictionary(uniqueKeysWithValues: GarmentDimension.allCases.map { ($0, value(for: $0)) })
    }

    func generateAIPreview(serverURL: String) async {
        guard let garmentPhoto else {
            generationStatus = GarmentAIGenerationError.missingImage.localizedDescription
            return
        }

        guard let url = URL(string: serverURL.trimmingCharacters(in: .whitespacesAndNewlines)), url.scheme != nil else {
            generationStatus = GarmentAIGenerationError.invalidURL.localizedDescription
            return
        }

        isGeneratingPreview = true
        generationStatus = "正在生成 AI 3D 预览..."
        defer { isGeneratingPreview = false }

        do {
            let image = try await GarmentAIGenerationClient(baseURL: url)
                .generatePreview(photo: garmentPhoto, measurements: measurementsSnapshot())
            generatedPreview = image
            generationStatus = "AI 3D 预览已生成"
        } catch {
            generationStatus = error.localizedDescription
        }
    }

    func generateModel(serverURL: String) async {
        guard let garmentPhoto else {
            generationStatus = GarmentAIGenerationError.missingImage.localizedDescription
            return
        }

        guard let url = URL(string: serverURL.trimmingCharacters(in: .whitespacesAndNewlines)), url.scheme != nil else {
            generationStatus = GarmentAIGenerationError.invalidURL.localizedDescription
            return
        }

        isGeneratingPreview = true
        generationStatus = "正在生成真 3D 模型，需耗时 10 分钟左右..."
        generatedModelURL = nil
        generatedMesh = nil
        generatedPreview = nil
        activeArchiveID = nil
        bumpPreviewRevision()
        defer { isGeneratingPreview = false }

        do {
            let result = try await GarmentAIGenerationClient(baseURL: url)
                .generateModel(photo: garmentPhoto, measurements: measurementsSnapshot()) { [weak self] status in
                    self?.generationStatus = status
                }
            if let localModelURL = result.localModelURL {
                let archive = try persistGeneratedModel(
                    temporaryURL: localModelURL,
                    sourcePhoto: garmentPhoto,
                    measurements: measurementsSnapshot(),
                    jobId: result.jobId,
                    provider: result.provider
                )
                activeArchiveID = archive.id
                loadModelArchives()
                generatedMesh = result.mesh
                generatedModelURL = archiveModelURL(for: archive)
                generationStatus = "Seed3D 模型已生成"
            } else {
                generatedMesh = result.mesh
                generatedModelURL = nil
                activeArchiveID = nil
                generationStatus = "规则 3D 模型已生成"
            }
            generatedPreview = nil
            bumpPreviewRevision()
        } catch {
            generationStatus = error.localizedDescription
            bumpPreviewRevision()
        }
    }

    func selectArchive(_ archive: GarmentModelArchive) {
        let modelURL = archiveModelURL(for: archive)
        guard FileManager.default.fileExists(atPath: modelURL.path) else {
            generationStatus = "模型文件不存在"
            loadModelArchives()
            return
        }
        restoreArchiveInputs(archive)
        generatedModelURL = modelURL
        generatedMesh = nil
        generatedPreview = nil
        activeArchiveID = archive.id
        generationStatus = "已加载历史 3D 模型"
        bumpPreviewRevision()

        if let thumbnailURL = archiveThumbnailURL(for: archive),
           let image = UIImage(contentsOfFile: thumbnailURL.path) {
            garmentPhoto = image
            garmentTexture = GarmentTextureProcessor.makeTexture(from: image)
        }
    }

    private func restoreArchiveInputs(_ archive: GarmentModelArchive) {
        for dimension in GarmentDimension.allCases {
            if let value = archive.measurements[dimension.rawValue] {
                values[dimension] = value
            }
        }
        if let thumbnailURL = archiveThumbnailURL(for: archive),
           let image = UIImage(contentsOfFile: thumbnailURL.path) {
            garmentPhoto = image
            garmentTexture = GarmentTextureProcessor.makeTexture(from: image)
        }
    }

    func thumbnail(for archive: GarmentModelArchive) -> UIImage? {
        guard let url = archiveThumbnailURL(for: archive) else { return nil }
        return UIImage(contentsOfFile: url.path)
    }

    var activeQuickLookURL: URL? {
        guard let activeArchiveID,
              let archive = modelArchives.first(where: { $0.id == activeArchiveID }) else {
            return nil
        }
        return quickLookURL(for: archive)
    }

    func quickLookURL(for archive: GarmentModelArchive) -> URL? {
        let url = archiveModelURL(for: archive)
        guard FileManager.default.fileExists(atPath: url.path) else { return nil }
        return url
    }

    func deleteArchive(_ archive: GarmentModelArchive) {
        let directory = archiveRootURL.appendingPathComponent(archive.id, isDirectory: true)
        try? FileManager.default.removeItem(at: directory)

        if activeArchiveID == archive.id {
            generatedModelURL = nil
            generatedMesh = nil
            generatedPreview = nil
            activeArchiveID = nil
            generationStatus = "历史模型已删除"
            bumpPreviewRevision()
        }

        loadModelArchives()
    }

    private func persistGeneratedModel(
        temporaryURL: URL,
        sourcePhoto: UIImage,
        measurements: [GarmentDimension: Double],
        jobId: String,
        provider: String
    ) throws -> GarmentModelArchive {
        let createdAt = Date()
        let id = archiveIDFormatter.string(from: createdAt)
        let directory = archiveRootURL.appendingPathComponent(id, isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)

        let ext = temporaryURL.pathExtension.isEmpty ? "usdz" : temporaryURL.pathExtension
        let modelFileName = "model.\(ext)"
        let modelURL = directory.appendingPathComponent(modelFileName)
        if FileManager.default.fileExists(atPath: modelURL.path) {
            try FileManager.default.removeItem(at: modelURL)
        }
        try FileManager.default.copyItem(at: temporaryURL, to: modelURL)

        let thumbnailFileName = "thumbnail.jpg"
        let thumbnailURL = directory.appendingPathComponent(thumbnailFileName)
        if let data = sourcePhoto.resizedForArchiveThumbnail(maxSide: 512).jpegData(compressionQuality: 0.72) {
            try data.write(to: thumbnailURL, options: [.atomic])
        }

        let archive = GarmentModelArchive(
            id: id,
            createdAt: createdAt,
            jobId: jobId,
            provider: provider,
            modelFileName: modelFileName,
            thumbnailFileName: FileManager.default.fileExists(atPath: thumbnailURL.path) ? thumbnailFileName : nil,
            measurements: Dictionary(uniqueKeysWithValues: measurements.map { ($0.key.rawValue, $0.value) })
        )
        let metadataURL = directory.appendingPathComponent("metadata.json")
        let metadata = try JSONEncoder.archiveEncoder.encode(archive)
        try metadata.write(to: metadataURL, options: [.atomic])
        return archive
    }

    private func loadModelArchives() {
        let root = archiveRootURL
        guard let directories = try? FileManager.default.contentsOfDirectory(
            at: root,
            includingPropertiesForKeys: [.isDirectoryKey],
            options: [.skipsHiddenFiles]
        ) else {
            modelArchives = []
            return
        }

        modelArchives = directories.compactMap { directory in
            let metadataURL = directory.appendingPathComponent("metadata.json")
            guard let data = try? Data(contentsOf: metadataURL) else { return nil }
            return try? JSONDecoder.archiveDecoder.decode(GarmentModelArchive.self, from: data)
        }
        .filter { archive in
            FileManager.default.fileExists(atPath: archiveModelURL(for: archive).path)
        }
        .sorted { $0.createdAt > $1.createdAt }
    }

    private func archiveModelURL(for archive: GarmentModelArchive) -> URL {
        archiveRootURL
            .appendingPathComponent(archive.id, isDirectory: true)
            .appendingPathComponent(archive.modelFileName)
    }

    private func archiveThumbnailURL(for archive: GarmentModelArchive) -> URL? {
        guard let thumbnailFileName = archive.thumbnailFileName else { return nil }
        return archiveRootURL
            .appendingPathComponent(archive.id, isDirectory: true)
            .appendingPathComponent(thumbnailFileName)
    }

    private var archiveRootURL: URL {
        let root = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("Garment3DHistory", isDirectory: true)
        try? FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        return root
    }

    private func bumpPreviewRevision() {
        previewRevision &+= 1
    }

    private var archiveIDFormatter: DateFormatter {
        let formatter = DateFormatter()
        formatter.dateFormat = "yyyyMMdd-HHmmss"
        return formatter
    }
}

private extension JSONEncoder {
    static var archiveEncoder: JSONEncoder {
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        return encoder
    }
}

private extension JSONDecoder {
    static var archiveDecoder: JSONDecoder {
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        return decoder
    }
}

private extension UIImage {
    func resizedForWorkingCopy(maxSide: CGFloat) -> UIImage {
        resizedToFit(maxSide: maxSide, opaque: false)
    }

    func resizedForArchiveThumbnail(maxSide: CGFloat) -> UIImage {
        resizedToFit(maxSide: maxSide, opaque: true)
    }

    private func resizedToFit(maxSide: CGFloat, opaque: Bool) -> UIImage {
        let longestSide = max(size.width, size.height)
        guard longestSide > maxSide else { return self }

        let scale = maxSide / longestSide
        let targetSize = CGSize(
            width: max(1, floor(size.width * scale)),
            height: max(1, floor(size.height * scale))
        )
        let format = UIGraphicsImageRendererFormat()
        format.scale = 1
        format.opaque = opaque
        let renderer = UIGraphicsImageRenderer(size: targetSize, format: format)
        return renderer.image { _ in
            if opaque {
                UIColor.white.setFill()
                UIBezierPath(rect: CGRect(origin: .zero, size: targetSize)).fill()
            }
            draw(in: CGRect(origin: .zero, size: targetSize))
        }
    }
}
