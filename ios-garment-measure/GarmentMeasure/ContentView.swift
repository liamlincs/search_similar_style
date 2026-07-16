import SwiftUI
import UIKit
import RealityKit
import ARKit
import SceneKit
import _RealityKit_SwiftUI

struct ContentView: View {
    @EnvironmentObject private var garment: GarmentMeasurementStore
    @State private var selectedTab = AppTab.measure

    var body: some View {
        TabView(selection: $selectedTab) {
            MeasureScreen(isActive: selectedTab == .measure)
                .tabItem {
                    Label("测量", systemImage: "camera.viewfinder")
                }
                .tag(AppTab.measure)

            ScanModelScreen()
                .tabItem {
                    Label("扫描", systemImage: "viewfinder")
                }
                .tag(AppTab.scan)

            GarmentPreviewScreen()
                .tabItem {
                    Label("3D", systemImage: "tshirt")
                }
                .tag(AppTab.preview)
        }
    }
}

private enum AppTab {
    case measure
    case scan
    case preview
}

private struct ScanModelScreen: View {
    @EnvironmentObject private var garment: GarmentMeasurementStore
    @State private var showingScanner = false
    @State private var latestModelURL: URL?
    @State private var scanMessage = "按引导绕物体拍一圈，完成后生成 USDZ。"
    @State private var isReconstructing = false

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(spacing: 16) {
                    Image(systemName: "camera.viewfinder")
                        .font(.system(size: 42, weight: .semibold))
                        .foregroundStyle(.blue)
                        .frame(width: 82, height: 82)
                        .background(Color.blue.opacity(0.12), in: RoundedRectangle(cornerRadius: 18))
                        .padding(.top, 8)

                    VStack(spacing: 6) {
                        Text("高质量扫描")
                            .font(.system(size: 30, weight: .bold))

                        Text("Object Capture 拍摄，生成带纹理 USDZ。")
                            .font(.subheadline)
                            .foregroundStyle(.secondary)
                            .multilineTextAlignment(.center)
                    }

                    VStack(alignment: .leading, spacing: 10) {
                        ScanCapabilityRow(icon: "cube.transparent", title: "USDZ", detail: "可在 3D / AR 查看")
                        ScanCapabilityRow(icon: "timer", title: "几分钟", detail: "复杂物体会更久")
                        ScanCapabilityRow(icon: "tshirt", title: "衣服", detail: "建议撑开或套假人")
                    }
                    .padding(14)
                    .background(Color(.secondarySystemGroupedBackground), in: RoundedRectangle(cornerRadius: 16))
                    .padding(.horizontal, 18)

                    statusView

                    Button {
                        showingScanner = true
                    } label: {
                        Label(isReconstructing ? "正在重建模型..." : "开始扫描", systemImage: "camera.viewfinder")
                            .font(.headline)
                            .frame(maxWidth: .infinity)
                            .padding(.vertical, 14)
                    }
                    .buttonStyle(.borderedProminent)
                    .disabled(!ObjectCaptureSession.isSupported || isReconstructing)
                    .padding(.horizontal, 18)

                    if let latestModelURL {
                        ShareLink(item: latestModelURL) {
                            Label("导出 USDZ", systemImage: "square.and.arrow.up")
                                .font(.subheadline.weight(.semibold))
                        }
                    }

                    Spacer(minLength: 112)
                }
                .frame(maxWidth: .infinity)
            }
            .background(Color(.systemGroupedBackground))
            .navigationTitle("扫描")
            .fullScreenCover(isPresented: $showingScanner) {
                ObjectCaptureScanView { result in
                    switch result {
                    case .success(let directory):
                        reconstructObject(from: directory)
                    case .failure(let error):
                        scanMessage = error.localizedDescription
                    }
                    showingScanner = false
                }
            }
        }
    }

    @ViewBuilder
    private var statusView: some View {
        if !ObjectCaptureSession.isSupported {
            Text("当前设备不支持 Object Capture")
                .font(.footnote.weight(.medium))
                .foregroundStyle(.orange)
                .multilineTextAlignment(.center)
                .padding(.horizontal, 18)
        } else {
            HStack(spacing: 8) {
                if isReconstructing {
                    ProgressView()
                        .controlSize(.small)
                }
                Text(scanMessage)
                    .font(.footnote)
                    .foregroundStyle(.secondary)
                    .lineLimit(3)
                    .multilineTextAlignment(.center)
            }
            .padding(.horizontal, 18)
        }
    }

    private func reconstructObject(from imagesDirectory: URL) {
        let imageCount = ObjectCaptureFileStore.captureImageCount(in: imagesDirectory)
        guard imageCount >= ObjectCaptureReconstructor.minimumImageCount else {
            scanMessage = "只拍了 \(imageCount) 张，暂不重建。建议至少 \(ObjectCaptureReconstructor.minimumImageCount) 张，并绕物体拍满一圈。"
            return
        }

        isReconstructing = true
        scanMessage = "已拍 \(imageCount) 张，正在生成高质量 USDZ..."
        Task {
            do {
                let outputURL = try await ObjectCaptureReconstructor.reconstruct(
                    imagesDirectory: imagesDirectory
                ) { message in
                    await MainActor.run {
                        scanMessage = message
                    }
                }
                let archive = try garment.importObjectCaptureModel(
                    temporaryURL: outputURL,
                    imagesDirectory: imagesDirectory
                )
                latestModelURL = garment.quickLookURL(for: archive)
                scanMessage = "Object Capture 模型已生成"
            } catch {
                scanMessage = ObjectCaptureReconstructor.userFacingErrorMessage(for: error)
            }
            isReconstructing = false
        }
    }
}

private struct ScanCapabilityRow: View {
    let icon: String
    let title: String
    let detail: String

    var body: some View {
        HStack(alignment: .top, spacing: 12) {
            Image(systemName: icon)
                .font(.system(size: 18, weight: .semibold))
                .foregroundStyle(.blue)
                .frame(width: 28)

            VStack(alignment: .leading, spacing: 3) {
                Text(title)
                    .font(.subheadline.weight(.semibold))
                Text(detail)
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        }
    }
}

private struct ObjectCaptureScanView: View {
    typealias Completion = @MainActor (Result<URL, Error>) -> Void

    @StateObject private var controller = ObjectCaptureController()
    let completion: Completion

    var body: some View {
        ZStack {
            ObjectCaptureView(session: controller.session) {
                EmptyView()
            }
            .ignoresSafeArea()

            VStack {
                topBar
                Spacer()
                captureGuideOverlay
                Spacer()
                bottomPanel
            }
        }
        .background(Color.black)
        .onAppear {
            controller.startMonitoring { result in
                completion(result)
            }
            controller.start()
        }
    }

    private var topBar: some View {
        HStack {
            Button {
                controller.cancelAndMarkCompleted()
                completion(.failure(ObjectCaptureFlowError.cancelled))
            } label: {
                Image(systemName: "xmark")
                    .font(.system(size: 21, weight: .bold))
                    .frame(width: 52, height: 52)
            }
            .buttonStyle(.plain)
            .foregroundStyle(.black)
            .background(.white.opacity(0.88), in: Circle())

            Spacer()

            Text(controller.counterText)
                .font(.subheadline.weight(.bold))
                .monospacedDigit()
                .foregroundStyle(.white)
                .padding(.horizontal, 14)
                .padding(.vertical, 9)
                .background(.black.opacity(0.48), in: Capsule())
        }
        .padding(.horizontal, 22)
        .padding(.top, 54)
    }

    private var captureGuideOverlay: some View {
        ObjectCaptureOrbitGuide(
            shotCount: controller.shotCount,
            recommendedShotCount: controller.recommendedShotCount,
            completedPass: controller.userCompletedScanPass,
            currentDirection: controller.currentDirection
        )
        .opacity(controller.shouldShowCoverageGuide ? 1 : 0)
    }

    private var bottomPanel: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack(alignment: .firstTextBaseline) {
                Text(controller.title)
                    .font(.headline)
                    .foregroundStyle(.white)

                Spacer()

                Text("步骤 \(controller.captureStep.index)/3")
                    .font(.caption.weight(.bold))
                    .foregroundStyle(.white.opacity(0.76))
                    .padding(.horizontal, 10)
                    .padding(.vertical, 5)
                    .background(.white.opacity(0.14), in: Capsule())
            }

            if controller.shouldShowStepGuide {
                ObjectCaptureStepGuide(step: controller.captureStep)
            } else {
                ObjectCaptureReadinessGuide(
                    title: controller.readinessTitle,
                    message: controller.readinessMessage,
                    feedback: controller.feedbackText
                )
            }

            Text(controller.message)
                .font(.caption)
                .foregroundStyle(.white.opacity(0.68))
                .lineLimit(2)

            if !controller.feedbackText.isEmpty {
                Text(controller.feedbackText)
                    .font(.caption.weight(.medium))
                    .foregroundStyle(.yellow)
            }

            HStack(spacing: 12) {
                Button {
                    controller.performPrimaryAction()
                } label: {
                    Text(controller.primaryActionTitle)
                        .font(.headline)
                        .frame(maxWidth: .infinity)
                        .padding(.vertical, 14)
                }
                .buttonStyle(.borderedProminent)
                .disabled(!controller.canPerformPrimaryAction)

                Button {
                    controller.captureStillImage()
                } label: {
                    Image(systemName: "camera.fill")
                        .font(.system(size: 18, weight: .semibold))
                        .frame(width: 52, height: 52)
                }
                .buttonStyle(.bordered)
                .tint(.white)
                .disabled(!controller.canRequestImageCapture)
            }
        }
        .padding(18)
        .background(.black.opacity(0.62), in: RoundedRectangle(cornerRadius: 22))
        .padding(.horizontal, 18)
        .padding(.bottom, 24)
    }
}

@MainActor
private final class ObjectCaptureController: ObservableObject {
    let session = ObjectCaptureSession()
    @Published var state: ObjectCaptureSession.CaptureState = .initializing
    @Published var feedback: Set<ObjectCaptureSession.Feedback> = []
    @Published var shotCount = 0
    @Published var canRequestImageCapture = false
    @Published var userCompletedScanPass = false
    @Published var cameraTracking: ObjectCaptureSession.Tracking = .notAvailable
    @Published var isPaused = false
    @Published var initializationTimedOut = false

    let recommendedShotCount = 200

    private var imagesDirectory: URL?
    private var didComplete = false
    private var didStartMonitoring = false
    private let directions = ["正面", "右前", "右侧", "右后", "背面", "左后", "左侧", "左前"]

    var shouldShowCoverageGuide: Bool {
        switch state {
        case .capturing, .finishing, .completed:
            true
        default:
            false
        }
    }

    var shouldShowStepGuide: Bool {
        switch state {
        case .capturing, .finishing, .completed:
            true
        default:
            false
        }
    }

    var counterText: String {
        switch state {
        case .capturing, .finishing, .completed:
            "\(shotCount)/\(recommendedShotCount)"
        default:
            shotCount == 0 ? "未开始" : "\(shotCount)/\(recommendedShotCount)"
        }
    }

    var readinessTitle: String {
        if initializationTimedOut {
            return "相机未准备好"
        }

        return switch state {
        case .initializing: "等待相机准备"
        case .ready: "先检测物体"
        case .detecting: "确认取景后开始"
        case .failed: "扫描失败"
        default: "准备扫描"
        }
    }

    var readinessMessage: String {
        if initializationTimedOut {
            return "Object Capture 一直停在初始化。请关掉扫描页重进；如果仍不行，重启 App，并确认没有其它 App 占用相机。"
        }

        if feedbackText.contains("光线") || feedbackText.localizedCaseInsensitiveContains("light") {
            return "当前光线不足，移动不会计数。请补光或换到更亮的位置，等按钮变蓝后再开始。"
        }

        if isPaused {
            return "扫描会话已暂停，请保持 App 在前台并重新进入扫描。"
        }

        if let trackingMessage {
            return trackingMessage
        }

        switch state {
        case .initializing:
            return "现在还没有开始采集，绕拍不会增加张数。请保持手机稳定，等待按钮可点。"
        case .ready:
            return "把物体完整放进取景框，点击开始检测。"
        case .detecting:
            return "确认白色框覆盖物体后，点击开始拍摄。"
        case .failed(let error):
            return error.localizedDescription
        default:
            return "请按屏幕提示继续。"
        }
    }

    var trackingMessage: String? {
        switch cameraTracking {
        case .normal:
            nil
        case .notAvailable:
            "相机追踪暂不可用，请缓慢移动手机并让画面里有清晰纹理。"
        case .limited(let reason):
            switch reason {
            case .initializing:
                "正在初始化相机追踪，请保持手机稳定。"
            case .relocalizing:
                "正在重新定位，请慢慢移动手机对准物体。"
            case .excessiveMotion:
                "移动太快，请放慢速度。"
            case .insufficientFeatures:
                "画面特征太少，请让背景有纹理，避免纯白桌面或纯黑物体占满画面。"
            @unknown default:
                "相机追踪受限，请调整光线和拍摄角度。"
            }
        @unknown default:
            nil
        }
    }

    var currentDirection: String {
        guard shotCount > 0 else { return "正面" }
        let sector = min(directions.count - 1, shotCount / 8)
        return directions[sector]
    }

    var captureStep: ObjectCaptureScanStep {
        if shotCount >= 96 || userCompletedScanPass {
            return ObjectCaptureScanStep(
                index: 3,
                title: "补顶部和细节",
                message: "手机略抬高，补拍领口、边缘和反光少的位置。",
                icon: "arrow.up.forward"
            )
        }
        if shotCount >= 48 {
            return ObjectCaptureScanStep(
                index: 2,
                title: "降低角度再拍一圈",
                message: "向下移动到物体底部高度，慢慢绕拍底边。",
                icon: "arrow.down.forward"
            )
        }
        return ObjectCaptureScanStep(
            index: 1,
            title: "平视绕拍一圈",
            message: "保持物体完整入镜，按环形提示顺时针慢慢移动。",
            icon: "arrow.clockwise"
        )
    }

    var title: String {
        switch state {
        case .initializing: "初始化扫描"
        case .ready: "检测物体"
        case .detecting: "确认取景"
        case .capturing: "正在拍摄"
        case .finishing: "正在整理照片"
        case .completed: "拍摄完成"
        case .failed: "扫描失败"
        @unknown default: "扫描"
        }
    }

    var message: String {
        switch state {
        case .initializing:
            "请保持手机稳定，等待相机准备完成。"
        case .ready:
            "把物体放在光线均匀的位置，先让 App 识别目标。"
        case .detecting:
            "让物体完整进入画面，准备好后开始拍摄。"
        case .capturing:
            "\(captureStep.message) 当前建议：\(currentDirection)。"
        case .finishing:
            "正在保存拍摄数据，请稍等。"
        case .completed:
            "拍摄完成，接下来会生成带纹理 USDZ。"
        case .failed(let error):
            error.localizedDescription
        @unknown default:
            "请按屏幕提示继续扫描。"
        }
    }

    var primaryActionTitle: String {
        switch state {
        case .ready: "开始检测"
        case .detecting: "开始拍摄"
        case .capturing: "完成拍摄"
        case .completed: "生成模型"
        case .failed: "关闭"
        default: "请稍等"
        }
    }

    var canPerformPrimaryAction: Bool {
        switch state {
        case .ready, .detecting, .capturing, .completed, .failed:
            true
        default:
            false
        }
    }

    var feedbackText: String {
        feedback.map(\.localizedText).sorted().joined(separator: " / ")
    }

    func start() {
        guard imagesDirectory == nil else { return }
        do {
            let directory = try ObjectCaptureFileStore.newSessionDirectory()
            imagesDirectory = directory.appendingPathComponent("Images", isDirectory: true)
            let checkpointDirectory = directory.appendingPathComponent("Checkpoints", isDirectory: true)
            try FileManager.default.createDirectory(at: imagesDirectory!, withIntermediateDirectories: true)
            try FileManager.default.createDirectory(at: checkpointDirectory, withIntermediateDirectories: true)

            var configuration = ObjectCaptureSession.Configuration()
            configuration.checkpointDirectory = checkpointDirectory
            configuration.isOverCaptureEnabled = true
            session.start(imagesDirectory: imagesDirectory!, configuration: configuration)

            Task { @MainActor [weak self] in
                try? await Task.sleep(for: .seconds(8))
                guard let self else { return }
                if case .initializing = self.state {
                    self.initializationTimedOut = true
                }
            }
        } catch {
            state = .failed(error)
        }
    }

    func startMonitoring(completion: @escaping ObjectCaptureScanView.Completion) {
        guard !didStartMonitoring else { return }
        didStartMonitoring = true

        Task { @MainActor [weak self] in
            guard let self else { return }
            self.state = self.session.state
            for await state in session.stateUpdates {
                self.state = state
                if state != .initializing {
                    self.initializationTimedOut = false
                }
                if case .completed = state, !self.didComplete {
                    self.didComplete = true
                    completion(.success(self.imagesDirectory ?? ObjectCaptureFileStore.rootDirectory))
                } else if case .failed(let error) = state, !self.didComplete {
                    if let objectCaptureError = error as? ObjectCaptureSession.Error,
                       case .cancelled = objectCaptureError {
                        self.didComplete = true
                        completion(.failure(ObjectCaptureFlowError.cancelled))
                        continue
                    }
                    self.didComplete = true
                    completion(.failure(error))
                }
            }
        }

        Task { @MainActor [weak self] in
            guard let self else { return }
            for await feedback in session.feedbackUpdates {
                self.feedback = feedback
            }
        }

        Task { @MainActor [weak self] in
            guard let self else { return }
            for await count in session.numberOfShotsTakenUpdates {
                self.shotCount = count
            }
        }

        Task { @MainActor [weak self] in
            guard let self else { return }
            for await value in session.canRequestImageCaptureUpdates {
                self.canRequestImageCapture = value
            }
        }

        Task { @MainActor [weak self] in
            guard let self else { return }
            for await value in session.userCompletedScanPassUpdates {
                self.userCompletedScanPass = value
            }
        }

        Task { @MainActor [weak self] in
            guard let self else { return }
            self.cameraTracking = self.session.cameraTracking
            while !Task.isCancelled && !self.didComplete {
                self.cameraTracking = self.session.cameraTracking
                try? await Task.sleep(for: .seconds(1))
            }
        }

        Task { @MainActor [weak self] in
            guard let self else { return }
            self.isPaused = self.session.isPaused
            for await value in session.isPausedUpdates {
                self.isPaused = value
            }
        }
    }

    func performPrimaryAction() {
        switch state {
        case .ready:
            _ = session.startDetecting()
        case .detecting:
            session.startCapturing()
        case .capturing:
            session.finish()
        case .completed:
            break
        case .failed:
            session.cancel()
        default:
            break
        }
    }

    func captureStillImage() {
        guard canRequestImageCapture else { return }
        session.requestImageCapture()
    }

    func cancel() {
        session.cancel()
    }

    func cancelAndMarkCompleted() {
        didComplete = true
        session.cancel()
    }
}

private enum ObjectCaptureFlowError: LocalizedError {
    case cancelled

    var errorDescription: String? {
        switch self {
        case .cancelled: "已取消扫描"
        }
    }
}

private struct ObjectCaptureReadinessGuide: View {
    let title: String
    let message: String
    let feedback: String

    var body: some View {
        HStack(alignment: .top, spacing: 12) {
            Image(systemName: feedback.contains("光线") || feedback.localizedCaseInsensitiveContains("light") ? "lightbulb.max.fill" : "camera.viewfinder")
                .font(.system(size: 22, weight: .bold))
                .foregroundStyle(.white)
                .frame(width: 42, height: 42)
                .background(.white.opacity(0.16), in: Circle())

            VStack(alignment: .leading, spacing: 6) {
                Text(title)
                    .font(.title3.weight(.bold))
                    .foregroundStyle(.white)

                Text(message)
                    .font(.subheadline)
                    .foregroundStyle(.white.opacity(0.78))
                    .lineLimit(3)
            }
        }
        .padding(12)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(.white.opacity(0.09), in: RoundedRectangle(cornerRadius: 16))
    }
}

private struct ObjectCaptureScanStep {
    let index: Int
    let title: String
    let message: String
    let icon: String
}

private struct ObjectCaptureStepGuide: View {
    let step: ObjectCaptureScanStep

    var body: some View {
        HStack(alignment: .top, spacing: 12) {
            Image(systemName: step.icon)
                .font(.system(size: 22, weight: .bold))
                .foregroundStyle(.white)
                .frame(width: 42, height: 42)
                .background(.white.opacity(0.16), in: Circle())

            VStack(alignment: .leading, spacing: 5) {
                HStack(spacing: 7) {
                    ForEach(1...3, id: \.self) { index in
                        Circle()
                            .fill(index <= step.index ? Color.white : Color.white.opacity(0.28))
                            .frame(width: index == step.index ? 10 : 8, height: index == step.index ? 10 : 8)
                    }
                }

                Text(step.title)
                    .font(.title3.weight(.bold))
                    .foregroundStyle(.white)

                Text(step.message)
                    .font(.subheadline)
                    .foregroundStyle(.white.opacity(0.76))
                    .lineLimit(2)
            }
        }
        .padding(12)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(.white.opacity(0.09), in: RoundedRectangle(cornerRadius: 16))
    }
}

private struct ObjectCaptureOrbitGuide: View {
    let shotCount: Int
    let recommendedShotCount: Int
    let completedPass: Bool
    let currentDirection: String

    private let tickCount = 64

    var body: some View {
        VStack(spacing: 10) {
            ZStack {
                ForEach(0..<tickCount, id: \.self) { index in
                    Capsule()
                        .fill(index < coveredTicks ? Color.white : Color.white.opacity(0.23))
                        .frame(width: 4, height: index % 8 == 0 ? 24 : 14)
                        .offset(y: -86)
                        .rotationEffect(.degrees(Double(index) / Double(tickCount) * 360))
                }

                Circle()
                    .fill(.black.opacity(0.22))
                    .frame(width: 128, height: 128)

                VStack(spacing: 4) {
                    Text(completedPass ? "一圈完成" : "绕拍覆盖")
                        .font(.subheadline.weight(.bold))
                        .foregroundStyle(.white)
                    Text(currentDirection)
                        .font(.title2.weight(.heavy))
                        .foregroundStyle(.white)
                    Text("\(shotCount)/\(recommendedShotCount)")
                        .font(.caption.weight(.semibold))
                        .monospacedDigit()
                        .foregroundStyle(.white.opacity(0.76))
                }
            }

            Text(completedPass ? "可继续补顶部/底部，或完成拍摄" : "按圆环空缺方向继续移动")
                .font(.headline.weight(.bold))
                .foregroundStyle(.white)
                .padding(.horizontal, 16)
                .padding(.vertical, 8)
                .background(.black.opacity(0.42), in: Capsule())
        }
        .frame(maxWidth: .infinity)
        .padding(.bottom, 12)
    }

    private var coveredTicks: Int {
        if completedPass { return tickCount }
        let progress = min(1, Double(shotCount) / Double(max(1, recommendedShotCount)))
        return Int(progress * Double(tickCount))
    }
}

private enum ObjectCaptureFileStore {
    static var rootDirectory: URL {
        FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("ObjectCaptureSessions", isDirectory: true)
    }

    static func newSessionDirectory() throws -> URL {
        let formatter = DateFormatter()
        formatter.dateFormat = "yyyyMMdd-HHmmss"
        let directory = rootDirectory.appendingPathComponent(formatter.string(from: Date()), isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        return directory
    }

    static func captureImageCount(in directory: URL) -> Int {
        let extensions = Set(["jpg", "jpeg", "heic", "png"])
        guard let files = try? FileManager.default.contentsOfDirectory(
            at: directory,
            includingPropertiesForKeys: nil
        ) else {
            return 0
        }
        return files.filter { extensions.contains($0.pathExtension.lowercased()) }.count
    }
}

private enum ObjectCaptureReconstructor {
    static let minimumImageCount = 60

    static func reconstruct(imagesDirectory: URL, progress: @escaping @Sendable (String) async -> Void) async throws -> URL {
        let outputURL = imagesDirectory
            .deletingLastPathComponent()
            .appendingPathComponent("model.usdz")
        if FileManager.default.fileExists(atPath: outputURL.path) {
            try FileManager.default.removeItem(at: outputURL)
        }

        var configuration = PhotogrammetrySession.Configuration()
        configuration.sampleOrdering = .sequential
        configuration.featureSensitivity = .high
        configuration.isObjectMaskingEnabled = true
        let checkpointDirectory = imagesDirectory
            .deletingLastPathComponent()
            .appendingPathComponent("ReconstructionCheckpoints", isDirectory: true)
        try FileManager.default.createDirectory(at: checkpointDirectory, withIntermediateDirectories: true)
        configuration.checkpointDirectory = checkpointDirectory

        let session = try PhotogrammetrySession(input: imagesDirectory, configuration: configuration)
        let request = PhotogrammetrySession.Request.modelFile(url: outputURL, detail: .reduced)
        try session.process(requests: [request])

        for try await output in session.outputs {
            switch output {
            case .requestProgress(_, let fraction):
                await progress("正在重建 USDZ：\(Int(fraction * 100))%")
            case .requestProgressInfo(_, let info):
                let stage = info.processingStage?.localizedText ?? "处理中"
                if let remaining = info.estimatedRemainingTime {
                    await progress("\(stage)，预计剩余 \(Int(remaining)) 秒")
                } else {
                    await progress(stage)
                }
            case .requestComplete(_, .modelFile(let url)):
                return url
            case .requestError(_, let error):
                throw error
            case .processingCancelled:
                throw ObjectCaptureFlowError.cancelled
            default:
                break
            }
        }

        guard FileManager.default.fileExists(atPath: outputURL.path) else {
            throw ObjectCaptureReconstructionError.missingOutput
        }
        return outputURL
    }

    static func userFacingErrorMessage(for error: Error) -> String {
        let nsError = error as NSError
        if nsError.domain.contains("PhotogrammetrySession") || nsError.domain.contains("CoreOC") {
            switch nsError.code {
            case 6:
                return "重建失败：照片数量或覆盖角度不足。请至少拍 \(minimumImageCount) 张，平视一圈后再补低角度和顶部细节。"
            default:
                return "重建失败：照片质量不够或覆盖不足（\(nsError.domain) \(nsError.code)）。请补光、放慢移动，并多拍几个角度。"
            }
        }
        return error.localizedDescription
    }
}

private enum ObjectCaptureReconstructionError: LocalizedError {
    case missingOutput

    var errorDescription: String? {
        "Object Capture 已结束，但没有生成 USDZ 文件。"
    }
}

private extension ObjectCaptureSession.Feedback {
    var localizedText: String {
        switch self {
        case .objectTooClose: "离物体太近"
        case .objectTooFar: "离物体太远"
        case .movingTooFast: "移动太快"
        case .environmentLowLight: "光线偏暗"
        case .environmentTooDark: "环境太暗"
        case .outOfFieldOfView: "物体不在画面内"
        case .objectNotFlippable: "不适合翻面扫描"
        case .overCapturing: "重复拍摄过多"
        case .objectNotDetected: "未检测到物体"
        @unknown default: "需要调整拍摄"
        }
    }
}

private extension PhotogrammetrySession.Output.ProcessingStage {
    var localizedText: String {
        switch self {
        case .preProcessing: "预处理照片"
        case .imageAlignment: "对齐照片"
        case .pointCloudGeneration: "生成点云"
        case .meshGeneration: "生成网格"
        case .textureMapping: "贴图映射"
        case .optimization: "优化模型"
        @unknown default: "处理中"
        }
    }
}

private struct MeasureScreen: View {
    @EnvironmentObject private var garment: GarmentMeasurementStore
    let isActive: Bool

    var body: some View {
        ZStack(alignment: .bottom) {
            ARMeasurementView(isActive: isActive) { distance in
                garment.setLatestDistance(distance * 100)
            }
            .ignoresSafeArea()

            VStack(spacing: 12) {
                Picker("尺寸", selection: $garment.selectedDimension) {
                    ForEach(GarmentDimension.allCases) { dimension in
                        Text(dimension.title).tag(dimension)
                    }
                }
                .pickerStyle(.menu)
                .tint(.white)
                .padding(.horizontal, 14)
                .padding(.vertical, 10)
                .background(.black.opacity(0.62), in: RoundedRectangle(cornerRadius: 8))

                HStack(spacing: 12) {
                    VStack(alignment: .leading, spacing: 3) {
                        Text(garment.selectedDimension.title)
                            .font(.caption)
                            .foregroundStyle(.white.opacity(0.75))

                        Text(latestDistanceText)
                            .font(.system(size: 28, weight: .semibold, design: .rounded))
                            .foregroundStyle(.white)
                            .monospacedDigit()
                    }

                    Spacer()

                    Button {
                        garment.saveLatestDistance()
                    } label: {
                        Image(systemName: "checkmark")
                            .font(.system(size: 20, weight: .bold))
                            .frame(width: 48, height: 48)
                    }
                    .buttonStyle(.borderedProminent)
                    .tint(.green)
                    .disabled(garment.latestDistanceCentimeters == nil)

                    Button {
                        garment.reset()
                    } label: {
                        Image(systemName: "arrow.counterclockwise")
                            .font(.system(size: 19, weight: .semibold))
                            .frame(width: 48, height: 48)
                    }
                    .buttonStyle(.bordered)
                    .tint(.white)
                }
                .padding(14)
                .background(.black.opacity(0.68), in: RoundedRectangle(cornerRadius: 8))

                ScrollView(.horizontal, showsIndicators: false) {
                    HStack(spacing: 8) {
                        ForEach(GarmentDimension.allCases) { dimension in
                            DimensionChip(
                                title: dimension.title,
                                value: garment.values[dimension]
                            )
                        }
                    }
                    .padding(.horizontal, 2)
                }
            }
            .padding(.horizontal, 14)
            .padding(.bottom, 18)
        }
    }

    private var latestDistanceText: String {
        guard let value = garment.latestDistanceCentimeters else {
            return "点选两点"
        }
        return "\(value.formatted(.number.precision(.fractionLength(1)))) cm"
    }
}

private struct DimensionChip: View {
    let title: String
    let value: Double?

    var body: some View {
        VStack(spacing: 3) {
            Text(title)
                .font(.caption2)
                .foregroundStyle(.white.opacity(0.7))
            Text(valueText)
                .font(.caption.weight(.semibold))
                .foregroundStyle(.white)
                .monospacedDigit()
        }
        .frame(width: 72, height: 50)
        .background(.black.opacity(0.56), in: RoundedRectangle(cornerRadius: 8))
    }

    private var valueText: String {
        guard let value else { return "--" }
        return value.formatted(.number.precision(.fractionLength(1)))
    }
}

private struct GarmentPreviewScreen: View {
    @EnvironmentObject private var garment: GarmentMeasurementStore
    @AppStorage("garmentAIBaseURL") private var aiBaseURL = "https://api.openfire.cloud"
    @AppStorage("shows3DDimensionGuides") private var showsDimensionGuides = true
    @State private var isPreviewExpanded = false
    @State private var showingImageSource = false
    @State private var showingImagePicker = false
    @State private var showingModelHistory = false
    @State private var detailModelURL: ModelPreviewURL?
    @State private var imageSource: UIImagePickerController.SourceType = .camera
    private let columns = [
        GridItem(.flexible(), spacing: 10),
        GridItem(.flexible(), spacing: 10)
    ]

    var body: some View {
        GeometryReader { proxy in
            VStack(spacing: 0) {
                ZStack(alignment: .bottom) {
                    GarmentPreviewView(
                        store: garment,
                        isExpanded: isPreviewExpanded,
                        showsDimensionGuides: showsDimensionGuides
                    )
                        .ignoresSafeArea(edges: .top)

                    HStack(spacing: 8) {
                        Image(systemName: isPreviewExpanded ? "chevron.up" : "chevron.down")
                            .font(.caption.weight(.bold))
                        Text(isPreviewExpanded ? "上滑收起" : "下拉放大")
                            .font(.caption.weight(.medium))
                    }
                    .foregroundStyle(.black.opacity(0.58))
                    .padding(.horizontal, 12)
                    .padding(.vertical, 7)
                    .background(Color.white.opacity(0.72), in: Capsule())
                    .padding(.bottom, 12)
                    .contentShape(Rectangle())
                    .onTapGesture {
                        togglePreview()
                    }
                    .gesture(
                        DragGesture(minimumDistance: 18)
                            .onEnded { value in
                                withAnimation(.spring(response: 0.35, dampingFraction: 0.82)) {
                                    if value.translation.height > 28 {
                                        isPreviewExpanded = true
                                    } else if value.translation.height < -28 {
                                        isPreviewExpanded = false
                                    }
                                }
                            }
                    )
                }
                .ignoresSafeArea(edges: .top)
                .frame(height: previewHeight(for: proxy.size.height))

                VStack(alignment: .leading, spacing: 12) {
                    compactActionBar
                    if !garment.generationStatus.isEmpty {
                        Text(garment.generationStatus)
                            .font(.caption)
                            .foregroundStyle(.white.opacity(0.58))
                            .lineLimit(2)
                    }

                    LazyVGrid(columns: columns, spacing: 10) {
                        ForEach(GarmentDimension.allCases) { dimension in
                            MeasurementTile(
                                title: dimension.title,
                                value: garment.value(for: dimension),
                                measured: garment.values[dimension] != nil
                            )
                        }
                    }
                }
                .padding(.horizontal, 18)
                .padding(.top, 14)
                .padding(.bottom, 34)
                .background(
                    UnevenRoundedRectangle(topLeadingRadius: 26, topTrailingRadius: 26)
                        .fill(Color(red: 0.085, green: 0.087, blue: 0.098))
                        .shadow(color: .black.opacity(0.22), radius: 16, x: 0, y: -6)
                )
            }
        }
        .background(Color.white)
        .confirmationDialog("T 恤照片", isPresented: $showingImageSource, titleVisibility: .visible) {
            if UIImagePickerController.isSourceTypeAvailable(.camera) {
                Button("拍摄 T 恤") {
                    imageSource = .camera
                    showingImagePicker = true
                }
            }

            Button("从相册选择") {
                imageSource = .photoLibrary
                showingImagePicker = true
            }

            if garment.garmentPhoto != nil {
                Button("清除照片", role: .destructive) {
                    garment.setGarmentPhoto(nil)
                }
            }

            Button("取消", role: .cancel) {}
        }
        .sheet(isPresented: $showingImagePicker) {
            ImagePicker(sourceType: imageSource) { image in
                garment.setGarmentPhoto(image)
            }
            .ignoresSafeArea()
        }
        .sheet(isPresented: $showingModelHistory) {
            ModelHistoryView(
                onPreview: { archive in
                    garment.selectArchive(archive)
                    let url = garment.quickLookURL(for: archive)
                    showingModelHistory = false
                    DispatchQueue.main.asyncAfter(deadline: .now() + 0.25) {
                        detailModelURL = url.map(ModelPreviewURL.init)
                    }
                }
            )
            .environmentObject(garment)
        }
        .fullScreenCover(item: $detailModelURL) { item in
            RealModelDetailView(url: item.url, showsDimensionGuides: $showsDimensionGuides)
                .environmentObject(garment)
        }
        .onAppear {
            if aiBaseURL == "http://192.168.0.106:8001" || aiBaseURL == "http://192.168.0.111:8001" {
                aiBaseURL = "https://api.openfire.cloud"
            }
        }
    }

    private var compactActionBar: some View {
        HStack(spacing: 10) {
            Button {
                showingImageSource = true
            } label: {
                Image(systemName: garment.garmentPhoto == nil ? "camera.fill" : "photo.fill")
                    .font(.system(size: 15, weight: .semibold))
                    .frame(width: 36, height: 36)
            }
            .buttonStyle(.plain)
            .foregroundStyle(.white)
            .background(Color.white.opacity(0.1), in: Circle())

            Button {
                Task {
                    await garment.generateModel(serverURL: aiBaseURL)
                    if let url = garment.activeQuickLookURL {
                        detailModelURL = ModelPreviewURL(url: url)
                    }
                }
            } label: {
                if garment.isGeneratingPreview {
                    ProgressView()
                        .tint(.white)
                        .frame(width: 36, height: 36)
                } else {
                    Image(systemName: "wand.and.stars")
                        .font(.system(size: 15, weight: .bold))
                        .frame(width: 36, height: 36)
                }
            }
            .buttonStyle(.plain)
            .foregroundStyle(.white)
            .background(Color.blue.opacity(0.9), in: Circle())
            .disabled(garment.garmentPhoto == nil || garment.isGeneratingPreview)
            .opacity(garment.garmentPhoto == nil ? 0.45 : 1)

            Button {
                showsDimensionGuides.toggle()
            } label: {
                Image(systemName: showsDimensionGuides ? "ruler.fill" : "ruler")
                    .font(.system(size: 15, weight: .semibold))
                    .frame(width: 36, height: 36)
            }
            .buttonStyle(.plain)
            .foregroundStyle(showsDimensionGuides ? .white : .white.opacity(0.5))
            .background(Color.white.opacity(showsDimensionGuides ? 0.16 : 0.08), in: Circle())

            Button {
                showingModelHistory = true
            } label: {
                Image(systemName: "clock.arrow.circlepath")
                    .font(.system(size: 15, weight: .semibold))
                    .frame(width: 36, height: 36)
            }
            .buttonStyle(.plain)
            .foregroundStyle(.white)
            .background(Color.white.opacity(0.1), in: Circle())

            Button {
                detailModelURL = garment.activeQuickLookURL.map(ModelPreviewURL.init)
            } label: {
                Image(systemName: "cube.transparent")
                    .font(.system(size: 15, weight: .semibold))
                    .frame(width: 36, height: 36)
            }
            .buttonStyle(.plain)
            .foregroundStyle(garment.activeQuickLookURL == nil ? .white.opacity(0.35) : .white)
            .background(Color.white.opacity(garment.activeQuickLookURL == nil ? 0.06 : 0.1), in: Circle())
            .disabled(garment.activeQuickLookURL == nil)

            if garment.garmentPhoto != nil {
                Button {
                    garment.setGarmentPhoto(nil)
                } label: {
                    Image(systemName: "xmark")
                        .font(.system(size: 14, weight: .bold))
                        .frame(width: 36, height: 36)
                }
                .buttonStyle(.plain)
                .foregroundStyle(.white.opacity(0.78))
                .background(Color.white.opacity(0.1), in: Circle())
            }
        }
    }

    private func previewHeight(for totalHeight: CGFloat) -> CGFloat {
        let ratio = isPreviewExpanded ? 0.68 : 0.5
        return min(isPreviewExpanded ? 560 : 430, max(isPreviewExpanded ? 460 : 340, totalHeight * ratio))
    }

    private func togglePreview() {
        withAnimation(.spring(response: 0.35, dampingFraction: 0.82)) {
            isPreviewExpanded.toggle()
        }
    }
}

private struct ModelHistoryView: View {
    @EnvironmentObject private var garment: GarmentMeasurementStore
    @Environment(\.dismiss) private var dismiss
    var onPreview: (GarmentModelArchive) -> Void

    var body: some View {
        NavigationStack {
            Group {
                if garment.modelArchives.isEmpty {
                    VStack(spacing: 12) {
                        Image(systemName: "cube.transparent")
                            .font(.system(size: 44, weight: .semibold))
                            .foregroundStyle(.secondary)
                        Text("还没有历史 3D 模型")
                            .font(.headline)
                        Text("生成成功后会按时间保存在这里。")
                            .font(.subheadline)
                            .foregroundStyle(.secondary)
                    }
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
                    .background(Color(.systemGroupedBackground))
                } else {
                    List {
                        ForEach(garment.modelArchives) { archive in
                            Button {
                                onPreview(archive)
                            } label: {
                                HStack(spacing: 12) {
                                    thumbnail(for: archive)

                                    if archive.id == garment.activeArchiveID {
                                        Text("当前")
                                            .font(.caption2.weight(.bold))
                                            .foregroundStyle(.white)
                                            .padding(.horizontal, 6)
                                            .padding(.vertical, 3)
                                            .background(Color.blue, in: Capsule())
                                    }

                                    VStack(alignment: .leading, spacing: 6) {
                                        Text("T 恤 3D 模型")
                                            .font(.headline)
                                            .foregroundStyle(.primary)
                                        Text(summary(for: archive))
                                            .font(.caption)
                                            .foregroundStyle(.secondary)
                                            .lineLimit(1)
                                    }

                                    Spacer()
                                }
                            }
                            .buttonStyle(.plain)
                            .padding(.vertical, 6)
                            .swipeActions(edge: .trailing, allowsFullSwipe: false) {
                                Button(role: .destructive) {
                                    garment.deleteArchive(archive)
                                } label: {
                                    Label("删除", systemImage: "trash")
                                }
                            }
                            .listRowInsets(EdgeInsets(top: 8, leading: 18, bottom: 8, trailing: 18))
                            .listRowBackground(Color.clear)
                        }
                    }
                    .listStyle(.plain)
                    .scrollContentBackground(.hidden)
                }
            }
            .navigationTitle("历史 3D 模型")
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    Button("关闭") {
                        dismiss()
                    }
                }
            }
        }
    }

    private func summary(for archive: GarmentModelArchive) -> String {
        let body = archive.measurements[GarmentDimension.bodyLength.rawValue]
            ?? GarmentDimension.bodyLength.fallbackCentimeters
        let chest = archive.measurements[GarmentDimension.chestWidth.rawValue]
            ?? GarmentDimension.chestWidth.fallbackCentimeters
        let shoulder = archive.measurements[GarmentDimension.shoulderWidth.rawValue]
            ?? GarmentDimension.shoulderWidth.fallbackCentimeters
        return "衣长 \(body.formatted(.number.precision(.fractionLength(1)))) / 胸宽 \(chest.formatted(.number.precision(.fractionLength(1)))) / 肩宽 \(shoulder.formatted(.number.precision(.fractionLength(1)))) cm"
    }

    @ViewBuilder
    private func thumbnail(for archive: GarmentModelArchive) -> some View {
        if let image = garment.thumbnail(for: archive) {
            Image(uiImage: image)
                .resizable()
                .scaledToFill()
                .frame(width: 96, height: 96)
                .clipShape(RoundedRectangle(cornerRadius: 8))
        } else {
            RoundedRectangle(cornerRadius: 10)
                .fill(Color.secondary.opacity(0.15))
                .frame(width: 96, height: 96)
                .overlay {
                    Image(systemName: "cube")
                        .foregroundStyle(.secondary)
                }
        }
    }
}

private struct ModelPreviewURL: Identifiable {
    let url: URL
    var id: String { url.path }
}

private struct RealModelDetailView: View {
    @EnvironmentObject private var garment: GarmentMeasurementStore
    @Environment(\.dismiss) private var dismiss
    let url: URL
    @Binding var showsDimensionGuides: Bool
    @State private var resetModelToken = 0

    var body: some View {
        ZStack {
            RealityModelPreview(url: url, initialScale: 0.5, resetToken: resetModelToken)
                .ignoresSafeArea()

            VStack {
                HStack(spacing: 12) {
                    Button {
                        dismiss()
                    } label: {
                        Image(systemName: "xmark")
                            .font(.system(size: 21, weight: .bold))
                            .frame(width: 52, height: 52)
                    }
                    .buttonStyle(.plain)
                    .foregroundStyle(.black)
                    .background(.white.opacity(0.86), in: Circle())

                    Spacer()

                    Button {
                        resetModelToken += 1
                    } label: {
                        Image(systemName: "scope")
                            .font(.system(size: 18, weight: .semibold))
                            .frame(width: 52, height: 52)
                    }
                    .buttonStyle(.plain)
                    .foregroundStyle(.black)
                    .background(.white.opacity(0.86), in: Circle())

                    Button {
                        showsDimensionGuides.toggle()
                    } label: {
                        Image(systemName: showsDimensionGuides ? "ruler.fill" : "ruler")
                            .font(.system(size: 18, weight: .semibold))
                            .frame(width: 52, height: 52)
                    }
                    .buttonStyle(.plain)
                    .foregroundStyle(.black)
                    .background(.white.opacity(0.86), in: Circle())
                }
                .padding(.horizontal, 24)
                .padding(.top, 54)

                Spacer()

                if showsDimensionGuides {
                    DimensionSummaryOverlay()
                        .environmentObject(garment)
                        .padding(.horizontal, 18)
                        .padding(.bottom, 28)
                        .transition(.move(edge: .bottom).combined(with: .opacity))
                }
            }
        }
    }
}

private struct RealityModelPreview: UIViewRepresentable {
    let url: URL
    let initialScale: Float
    let resetToken: Int

    func makeCoordinator() -> Coordinator {
        Coordinator()
    }

    func makeUIView(context: Context) -> ARView {
        let view = ARView(frame: .zero)
        view.environment.sceneUnderstanding.options = []
        view.renderOptions.insert(.disableMotionBlur)
        context.coordinator.attachGestures(to: view)
        context.coordinator.load(url: url, scale: initialScale, into: view)
        return view
    }

    func updateUIView(_ uiView: ARView, context: Context) {
        context.coordinator.load(url: url, scale: initialScale, into: uiView)
        context.coordinator.resetIfNeeded(resetToken)
    }

    @MainActor
    static func dismantleUIView(_ uiView: ARView, coordinator: Coordinator) {
        uiView.scene.anchors.removeAll()
        uiView.session.pause()
    }

    final class Coordinator: NSObject, UIGestureRecognizerDelegate {
        private var loadedKey = ""
        private weak var view: ARView?
        private weak var modelEntity: Entity?
        private var lastPinchScale: CGFloat = 1
        private var lastRotation: CGFloat = 0
        private var lastPanTranslation: CGPoint = .zero
        private var initialScale: SIMD3<Float> = .one
        private var initialPosition: SIMD3<Float> = .zero
        private var initialOrientation = simd_quatf(angle: 0, axis: SIMD3<Float>(0, 1, 0))
        private var handledResetToken = 0

        @MainActor
        func attachGestures(to view: ARView) {
            guard self.view !== view else { return }
            self.view = view

            let pinch = UIPinchGestureRecognizer(target: self, action: #selector(handlePinch(_:)))
            let rotation = UIRotationGestureRecognizer(target: self, action: #selector(handleRotation(_:)))
            let rotatePan = UIPanGestureRecognizer(target: self, action: #selector(handlePan(_:)))
            rotatePan.minimumNumberOfTouches = 1
            rotatePan.maximumNumberOfTouches = 1
            let movePan = UIPanGestureRecognizer(target: self, action: #selector(handleMovePan(_:)))
            movePan.minimumNumberOfTouches = 2
            movePan.maximumNumberOfTouches = 2
            let doubleTap = UITapGestureRecognizer(target: self, action: #selector(handleDoubleTap(_:)))
            doubleTap.numberOfTapsRequired = 2

            [pinch, rotation, rotatePan, movePan, doubleTap].forEach {
                $0.delegate = self
                view.addGestureRecognizer($0)
            }
        }

        @MainActor
        func load(url: URL, scale: Float, into view: ARView) {
            let key = "\(url.path)#\(scale)"
            guard loadedKey != key else { return }
            loadedKey = key
            view.scene.anchors.removeAll()

            do {
                let entity = try Entity.load(contentsOf: url)
                entity.scale = SIMD3<Float>(repeating: scale)
                center(entity)
                modelEntity = entity
                captureInitialTransform(entity)

                let anchor = AnchorEntity(world: SIMD3<Float>(0, -0.08, -0.82))
                anchor.addChild(entity)
                view.scene.addAnchor(anchor)
            } catch {
                let anchor = AnchorEntity(world: SIMD3<Float>(0, 0, -0.6))
                let fallback = ModelEntity(
                    mesh: .generateBox(size: 0.18),
                    materials: [SimpleMaterial(color: .systemGreen, roughness: 0.8, isMetallic: false)]
                )
                modelEntity = fallback
                captureInitialTransform(fallback)
                anchor.addChild(fallback)
                view.scene.addAnchor(anchor)
            }
        }

        @MainActor
        func resetIfNeeded(_ token: Int) {
            guard token != handledResetToken else { return }
            handledResetToken = token
            resetModel()
        }

        @MainActor
        private func center(_ entity: Entity) {
            let bounds = entity.visualBounds(relativeTo: nil)
            let center = bounds.center
            entity.position -= SIMD3<Float>(center.x, center.y, center.z)
        }

        @MainActor
        private func captureInitialTransform(_ entity: Entity) {
            initialScale = entity.scale
            initialPosition = entity.position
            initialOrientation = entity.orientation
        }

        @MainActor
        private func resetModel() {
            guard let entity = modelEntity else { return }
            entity.scale = initialScale
            entity.position = initialPosition
            entity.orientation = initialOrientation
        }

        func gestureRecognizer(_ gestureRecognizer: UIGestureRecognizer, shouldRecognizeSimultaneouslyWith otherGestureRecognizer: UIGestureRecognizer) -> Bool {
            true
        }

        @objc private func handlePinch(_ recognizer: UIPinchGestureRecognizer) {
            guard let entity = modelEntity else { return }
            if recognizer.state == .began {
                lastPinchScale = 1
            }
            let delta = Float(recognizer.scale / lastPinchScale)
            lastPinchScale = recognizer.scale
            let next = clamp(entity.scale.x * delta, min: 0.12, max: 2.2)
            entity.scale = SIMD3<Float>(repeating: next)
        }

        @objc private func handleRotation(_ recognizer: UIRotationGestureRecognizer) {
            guard let entity = modelEntity else { return }
            if recognizer.state == .began {
                lastRotation = 0
            }
            let delta = Float(recognizer.rotation - lastRotation)
            lastRotation = recognizer.rotation
            entity.orientation = simd_quatf(angle: -delta, axis: SIMD3<Float>(0, 0, 1)) * entity.orientation
        }

        @objc private func handlePan(_ recognizer: UIPanGestureRecognizer) {
            guard let entity = modelEntity, let view else { return }
            if recognizer.state == .began {
                lastPanTranslation = .zero
            }
            let translation = recognizer.translation(in: view)
            let dx = Float(translation.x - lastPanTranslation.x)
            let dy = Float(translation.y - lastPanTranslation.y)
            lastPanTranslation = translation

            let yaw = simd_quatf(angle: dx * 0.008, axis: SIMD3<Float>(0, 1, 0))
            let pitch = simd_quatf(angle: dy * 0.008, axis: SIMD3<Float>(1, 0, 0))
            entity.orientation = yaw * pitch * entity.orientation
        }

        @objc private func handleMovePan(_ recognizer: UIPanGestureRecognizer) {
            guard let entity = modelEntity, let view else { return }
            if recognizer.state == .began {
                lastPanTranslation = .zero
            }
            let translation = recognizer.translation(in: view)
            let dx = Float(translation.x - lastPanTranslation.x)
            let dy = Float(translation.y - lastPanTranslation.y)
            lastPanTranslation = translation
            entity.position += SIMD3<Float>(dx * 0.0012, -dy * 0.0012, 0)
            entity.position.x = clamp(entity.position.x, min: initialPosition.x - 0.55, max: initialPosition.x + 0.55)
            entity.position.y = clamp(entity.position.y, min: initialPosition.y - 0.42, max: initialPosition.y + 0.42)
        }

        @objc private func handleDoubleTap(_ recognizer: UITapGestureRecognizer) {
            guard recognizer.state == .ended else { return }
            resetModel()
        }

        private func clamp(_ value: Float, min minValue: Float, max maxValue: Float) -> Float {
            Swift.min(Swift.max(value, minValue), maxValue)
        }
    }
}

private struct DimensionSummaryOverlay: View {
    @EnvironmentObject private var garment: GarmentMeasurementStore

    var body: some View {
        LazyVGrid(columns: [GridItem(.adaptive(minimum: 92), spacing: 8)], spacing: 8) {
            ForEach(GarmentDimension.allCases) { dimension in
                VStack(alignment: .leading, spacing: 2) {
                    Text(dimension.title)
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                    Text("\(garment.value(for: dimension).formatted(.number.precision(.fractionLength(1)))) cm")
                        .font(.caption.weight(.semibold))
                        .monospacedDigit()
                }
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(.horizontal, 10)
                .padding(.vertical, 8)
                .background(Color.white.opacity(0.82), in: RoundedRectangle(cornerRadius: 8))
            }
        }
        .padding(10)
        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 12))
    }
}

private struct GarmentPhotoStrip: View {
    let image: UIImage?
    let texture: UIImage?
    let hasGeneratedMesh: Bool
    let generatedPreview: UIImage?
    let status: String
    let isGenerating: Bool
    var onAdd: () -> Void
    var onRegenerate: () -> Void
    var onClear: () -> Void

    var body: some View {
        HStack(spacing: 12) {
            Group {
                if hasGeneratedMesh {
                    Image(systemName: "tshirt.fill")
                        .font(.system(size: 34, weight: .semibold))
                        .foregroundStyle(.blue)
                } else if let generatedPreview {
                    Image(uiImage: generatedPreview)
                        .resizable()
                        .scaledToFill()
                } else if let texture {
                    Image(uiImage: texture)
                        .resizable()
                        .scaledToFit()
                        .padding(6)
                } else if let image {
                    Image(uiImage: image)
                        .resizable()
                        .scaledToFill()
                } else {
                    Image(systemName: "tshirt")
                        .font(.system(size: 28, weight: .semibold))
                        .foregroundStyle(.white.opacity(0.36))
                }
            }
            .frame(width: 68, height: 68)
            .clipShape(RoundedRectangle(cornerRadius: 14))
            .overlay(
                RoundedRectangle(cornerRadius: 14)
                    .stroke(Color.white.opacity(0.1), lineWidth: 1)
            )

            VStack(alignment: .leading, spacing: 4) {
                Text(title)
                    .font(.subheadline.weight(.semibold))
                    .foregroundStyle(.white)
                Text(subtitle)
                    .font(.caption)
                    .foregroundStyle(.white.opacity(0.52))
                    .lineLimit(2)
            }

            Spacer(minLength: 4)

            if image == nil {
                Button(action: onAdd) {
                    Image(systemName: "plus")
                        .font(.system(size: 17, weight: .bold))
                        .frame(width: 38, height: 38)
                }
                .buttonStyle(.plain)
                .foregroundStyle(.white)
                .background(Color.blue, in: Circle())
            } else {
                HStack(spacing: 8) {
                    Button(action: onRegenerate) {
                        if isGenerating {
                            ProgressView()
                                .tint(.white)
                                .frame(width: 34, height: 34)
                        } else {
                            Image(systemName: "wand.and.stars")
                                .font(.system(size: 14, weight: .bold))
                                .frame(width: 34, height: 34)
                        }
                    }
                    .buttonStyle(.plain)
                    .foregroundStyle(.white)
                    .background(Color.blue.opacity(0.9), in: Circle())
                    .disabled(isGenerating)

                    Button(action: onClear) {
                        Image(systemName: "xmark")
                            .font(.system(size: 14, weight: .bold))
                            .frame(width: 34, height: 34)
                    }
                    .buttonStyle(.plain)
                    .foregroundStyle(.white.opacity(0.78))
                    .background(Color.white.opacity(0.1), in: Circle())
                }
            }
        }
        .padding(12)
        .background(
            RoundedRectangle(cornerRadius: 16)
                .fill(Color.white.opacity(0.055))
                .overlay(
                    RoundedRectangle(cornerRadius: 16)
                        .stroke(Color.white.opacity(0.08), lineWidth: 1)
                )
        )
    }

    private var title: String {
        if hasGeneratedMesh {
            return "真 3D 模型已生成"
        }
        if generatedPreview != nil {
            return "效果图已生成"
        }
        return image == nil ? "添加实拍 T 恤" : "点击魔法棒生成真 3D 模型"
    }

    private var subtitle: String {
        if !status.isEmpty {
            return status
        }
        return image == nil ? "平铺拍摄效果最好，尽量让衣服占满画面。" : "会上传照片和尺寸，返回可旋转 mesh 模型。"
    }
}

private struct MeasurementTile: View {
    let title: String
    let value: Double
    let measured: Bool

    var body: some View {
        HStack(alignment: .firstTextBaseline) {
            VStack(alignment: .leading, spacing: 5) {
                Text(title)
                    .font(.subheadline.weight(.medium))
                    .foregroundStyle(.white.opacity(0.74))

                HStack(alignment: .firstTextBaseline, spacing: 4) {
                    Text(value.formatted(.number.precision(.fractionLength(1))))
                        .font(.system(size: 22, weight: .semibold, design: .rounded))
                        .foregroundStyle(.white)
                        .monospacedDigit()
                    Text("cm")
                        .font(.caption.weight(.medium))
                        .foregroundStyle(.white.opacity(0.48))
                }
            }

            Spacer(minLength: 4)

            Circle()
                .fill(measured ? Color.green : Color.white.opacity(0.18))
                .frame(width: 7, height: 7)
        }
        .padding(12)
        .frame(height: 74)
        .background(
            RoundedRectangle(cornerRadius: 12)
                .fill(Color.white.opacity(0.055))
                .overlay(
                    RoundedRectangle(cornerRadius: 12)
                        .stroke(Color.white.opacity(0.08), lineWidth: 1)
                )
        )
    }
}
