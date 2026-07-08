import SwiftUI
import UIKit
import RealityKit

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
    case preview
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
                onSelect: { archive in
                    garment.selectArchive(archive)
                    showingModelHistory = false
                },
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

            TextField("AI 服务地址", text: $aiBaseURL)
                .textInputAutocapitalization(.never)
                .keyboardType(.URL)
                .font(.caption2)
                .foregroundStyle(.white.opacity(0.72))
                .padding(.horizontal, 10)
                .frame(height: 34)
                .background(Color.black.opacity(0.22), in: Capsule())
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
    var onSelect: (GarmentModelArchive) -> Void
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
    @Binding var aiBaseURL: String
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
        .overlay(alignment: .bottomLeading) {
            if image != nil {
                TextField("AI 服务地址", text: $aiBaseURL)
                    .textInputAutocapitalization(.never)
                    .keyboardType(.URL)
                    .font(.caption2)
                    .foregroundStyle(.white.opacity(0.7))
                    .padding(.horizontal, 10)
                    .padding(.vertical, 7)
                    .background(Color.black.opacity(0.22), in: Capsule())
                    .padding(.leading, 92)
                    .padding(.bottom, 8)
                    .frame(maxWidth: 310)
            }
        }
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
