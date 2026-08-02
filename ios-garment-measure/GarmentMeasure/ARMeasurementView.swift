import ARKit
import SceneKit
import SwiftUI
import UIKit
import simd

struct ARMeasurementView: UIViewRepresentable {
    var isActive: Bool
    var photoCaptureRequestID: Int
    var clearMeasurementRequestID: Int
    var retakeRequestID: Int
    var restoredPhoto: UIImage?
    var restoredPhotoRevision: Int
    var restoredMeasurementPlane: SavedMeasurementPlane?
    var restoredProjection: SavedCameraProjection?
    var onPhotoCaptured: (UIImage, SavedMeasurementPlane?, SavedCameraProjection?) -> Void
    var onPhotoStateChanged: (Bool) -> Void
    var onMeasurementCleared: () -> Void
    var onDistanceChanged: (Double) -> Void

    func makeCoordinator() -> Coordinator {
        Coordinator(
            onPhotoCaptured: onPhotoCaptured,
            onPhotoStateChanged: onPhotoStateChanged,
            onMeasurementCleared: onMeasurementCleared,
            onDistanceChanged: onDistanceChanged
        )
    }

    func makeUIView(context: Context) -> ARSCNView {
        let view = ARSCNView(frame: .zero)
        view.autoenablesDefaultLighting = true
        view.automaticallyUpdatesLighting = true
        view.scene = SCNScene()
        view.delegate = context.coordinator
        context.coordinator.sceneView = view

        let tap = UITapGestureRecognizer(target: context.coordinator, action: #selector(Coordinator.handleTap(_:)))
        view.addGestureRecognizer(tap)
        context.coordinator.registerTapGesture(tap)

        if isActive {
            context.coordinator.startSession()
        }

        return view
    }

    func updateUIView(_ uiView: ARSCNView, context: Context) {
        context.coordinator.sceneView = uiView
        context.coordinator.onPhotoCaptured = onPhotoCaptured
        context.coordinator.onPhotoStateChanged = onPhotoStateChanged
        context.coordinator.onMeasurementCleared = onMeasurementCleared
        context.coordinator.onDistanceChanged = onDistanceChanged
        context.coordinator.processPhotoCaptureRequest(photoCaptureRequestID)
        context.coordinator.processClearMeasurementRequest(clearMeasurementRequestID)
        context.coordinator.processRetakeRequest(retakeRequestID)
        context.coordinator.processRestoredPhoto(
            restoredPhoto,
            revision: restoredPhotoRevision,
            measurementPlane: restoredMeasurementPlane,
            projection: restoredProjection
        )

        if isActive, !context.coordinator.hasFrozenPhoto {
            context.coordinator.startSession()
        } else {
            context.coordinator.pauseSession()
        }
    }

    static func dismantleUIView(_ uiView: ARSCNView, coordinator: Coordinator) {
        uiView.session.pause()
    }

    @MainActor
    final class Coordinator: NSObject, ARSCNViewDelegate {
        weak var sceneView: ARSCNView?
        private var points: [SCNVector3] = []
        private var nodes: [SCNNode] = []
        private var screenPoints: [CGPoint] = []
        private var markerViews: [EndpointMarkerView] = []
        private var lockedMeasurementPlane: MeasurementPlane?
        private var activeProjection: CameraProjection?
        private var photoOverlay: UIView?
        private var photoDrawingViews: [UIView] = []
        private weak var tapGesture: UITapGestureRecognizer?
        private var lastPhotoCaptureRequestID = 0
        private var lastClearMeasurementRequestID = 0
        private var lastRetakeRequestID = 0
        private var lastRestoredPhotoRevision = 0
        private(set) var hasFrozenPhoto = false
        private var isRestoredPhoto = false
        private var sessionRunning = false
        var onPhotoCaptured: (UIImage, SavedMeasurementPlane?, SavedCameraProjection?) -> Void
        var onPhotoStateChanged: (Bool) -> Void
        var onMeasurementCleared: () -> Void
        var onDistanceChanged: (Double) -> Void

        init(
            onPhotoCaptured: @escaping (UIImage, SavedMeasurementPlane?, SavedCameraProjection?) -> Void,
            onPhotoStateChanged: @escaping (Bool) -> Void,
            onMeasurementCleared: @escaping () -> Void,
            onDistanceChanged: @escaping (Double) -> Void
        ) {
            self.onPhotoCaptured = onPhotoCaptured
            self.onPhotoStateChanged = onPhotoStateChanged
            self.onMeasurementCleared = onMeasurementCleared
            self.onDistanceChanged = onDistanceChanged
        }

        func startSession() {
            guard !sessionRunning else { return }
            guard let sceneView else { return }
            let configuration = ARWorldTrackingConfiguration()
            configuration.planeDetection = [.horizontal]
            configuration.environmentTexturing = .automatic
            sceneView.session.run(configuration)
            sessionRunning = true
        }

        func registerTapGesture(_ gesture: UITapGestureRecognizer) {
            tapGesture = gesture
        }

        func processPhotoCaptureRequest(_ requestID: Int) {
            guard requestID != lastPhotoCaptureRequestID else { return }
            lastPhotoCaptureRequestID = requestID
            guard requestID > 0 else { return }
            freezeCurrentCameraImage()
        }

        func processClearMeasurementRequest(_ requestID: Int) {
            guard requestID != lastClearMeasurementRequestID else { return }
            lastClearMeasurementRequestID = requestID
            guard requestID > 0 else { return }
            clearPhotoMeasurement()
            notifyMeasurementCleared()
        }

        func processRetakeRequest(_ requestID: Int) {
            guard requestID != lastRetakeRequestID else { return }
            lastRetakeRequestID = requestID
            guard requestID > 0 else { return }
            clearFrozenPhoto()
            startSession()
        }

        func processRestoredPhoto(
            _ image: UIImage?,
            revision: Int,
            measurementPlane: SavedMeasurementPlane?,
            projection: SavedCameraProjection?
        ) {
            guard revision != lastRestoredPhotoRevision else { return }
            lastRestoredPhotoRevision = revision
            guard revision > 0, let image else { return }
            showFrozenPhoto(image)
            lockedMeasurementPlane = measurementPlane.map(MeasurementPlane.init(saved:))
            if measurementPlane != nil {
                activeProjection = projection.flatMap(CameraProjection.init(saved:))
            } else {
                activeProjection = nil
            }
            isRestoredPhoto = true
            pauseSession()
            notifyPhotoStateChanged(true)
        }

        func pauseSession() {
            sceneView?.session.pause()
            sessionRunning = false
        }

        @objc func handleTap(_ gesture: UITapGestureRecognizer) {
            guard let sceneView else { return }
            guard hasFrozenPhoto else { return }
            let location = gesture.location(in: sceneView)

            if points.count == 2 || screenPoints.count == 2 {
                clearPhotoMeasurement()
                notifyMeasurementCleared()
            }

            let position: SCNVector3
            if let lockedMeasurementPlane, let activeProjection {
                guard let intersection = worldPosition(
                    at: location,
                    on: lockedMeasurementPlane,
                    projection: activeProjection,
                    currentSize: sceneView.bounds.size
                ) else { return }
                position = intersection
            } else if let lockedMeasurementPlane {
                guard let intersection = worldPosition(at: location, on: lockedMeasurementPlane, in: sceneView) else { return }
                position = intersection
            } else if isRestoredPhoto {
                return
            } else {
                guard let plane = measurementPlane(at: location, in: sceneView),
                      let intersection = worldPosition(at: location, on: plane, in: sceneView) else {
                    return
                }
                lockedMeasurementPlane = plane
                position = intersection
            }

            points.append(position)
            screenPoints.append(location)
            addPhotoMarker(at: location, index: points.count - 1, in: sceneView)

            if points.count == 2 {
                updatePhotoMeasurementOverlay(in: sceneView)
            }
        }

        private func freezeCurrentCameraImage() {
            guard let sceneView else { return }
            clearFrozenPhoto()
            clearMeasurement()

            let photo = sceneView.snapshot()
            let savedPlane = defaultMeasurementPlane(in: sceneView)?.saved
            let savedProjection = CameraProjection(sceneView: sceneView)?.saved
            showFrozenPhoto(photo)
            lockedMeasurementPlane = savedPlane.map(MeasurementPlane.init(saved:))
            if savedPlane != nil {
                activeProjection = savedProjection.flatMap(CameraProjection.init(saved:))
            } else {
                activeProjection = nil
            }
            isRestoredPhoto = false
            hasFrozenPhoto = true
            pauseSession()
            notifyPhotoCaptured(photo, savedPlane, savedProjection)
            notifyPhotoStateChanged(true)
        }

        private func showFrozenPhoto(_ image: UIImage) {
            guard let sceneView else { return }
            photoOverlay?.removeFromSuperview()
            clearPhotoMeasurement()

            let overlay = UIView(frame: sceneView.bounds)
            overlay.autoresizingMask = [.flexibleWidth, .flexibleHeight]
            overlay.isUserInteractionEnabled = false

            let imageView = UIImageView(frame: overlay.bounds)
            imageView.autoresizingMask = [.flexibleWidth, .flexibleHeight]
            imageView.contentMode = .scaleToFill
            imageView.image = image
            overlay.addSubview(imageView)

            sceneView.addSubview(overlay)
            photoOverlay = overlay
            hasFrozenPhoto = true
        }

        private func clearFrozenPhoto() {
            photoOverlay?.removeFromSuperview()
            photoOverlay = nil
            clearPhotoMeasurement()
            lockedMeasurementPlane = nil
            activeProjection = nil
            isRestoredPhoto = false
            hasFrozenPhoto = false
            notifyPhotoStateChanged(false)
        }

        private func notifyPhotoCaptured(
            _ photo: UIImage,
            _ plane: SavedMeasurementPlane?,
            _ projection: SavedCameraProjection?
        ) {
            DispatchQueue.main.async { [onPhotoCaptured] in
                onPhotoCaptured(photo, plane, projection)
            }
        }

        private func notifyPhotoStateChanged(_ captured: Bool) {
            DispatchQueue.main.async { [onPhotoStateChanged] in
                onPhotoStateChanged(captured)
            }
        }

        private func notifyMeasurementCleared() {
            DispatchQueue.main.async { [onMeasurementCleared] in
                onMeasurementCleared()
            }
        }

        private func measurementPlane(at point: CGPoint, in sceneView: ARSCNView) -> MeasurementPlane? {
            if let result = sceneView.raycastQuery(from: point, allowing: .estimatedPlane, alignment: .horizontal)
                .flatMap({ sceneView.session.raycast($0).first }) {
                let transform = result.worldTransform
                return MeasurementPlane(
                    point: SCNVector3(transform.columns.3.x, transform.columns.3.y, transform.columns.3.z),
                    normal: SCNVector3(0, 1, 0)
                )
            }

            guard let result = sceneView.raycastQuery(from: point, allowing: .estimatedPlane, alignment: .any)
                .flatMap({ sceneView.session.raycast($0).first }) else {
                return nil
            }
            let transform = result.worldTransform
            return MeasurementPlane(
                point: SCNVector3(transform.columns.3.x, transform.columns.3.y, transform.columns.3.z),
                normal: SCNVector3(transform.columns.1.x, transform.columns.1.y, transform.columns.1.z).normalized()
            )
        }

        private func defaultMeasurementPlane(in sceneView: ARSCNView) -> MeasurementPlane? {
            let center = CGPoint(x: sceneView.bounds.midX, y: sceneView.bounds.midY)
            return measurementPlane(at: center, in: sceneView)
        }

        private func worldPosition(at point: CGPoint, on plane: MeasurementPlane, in sceneView: ARSCNView) -> SCNVector3? {
            let near = sceneView.unprojectPoint(SCNVector3(Float(point.x), Float(point.y), 0))
            let far = sceneView.unprojectPoint(SCNVector3(Float(point.x), Float(point.y), 1))
            let direction = (far - near).normalized()
            let denominator = plane.normal.dot(direction)
            guard abs(denominator) > 0.0001 else { return nil }

            let t = (plane.point - near).dot(plane.normal) / denominator
            guard t.isFinite, t >= 0 else { return nil }
            return near + direction * t
        }

        private func worldPosition(
            at point: CGPoint,
            on plane: MeasurementPlane,
            projection: CameraProjection,
            currentSize: CGSize
        ) -> SCNVector3? {
            let normalizedPoint = CGPoint(
                x: point.x / max(currentSize.width, 1),
                y: point.y / max(currentSize.height, 1)
            )
            let savedPoint = CGPoint(
                x: normalizedPoint.x * projection.viewSize.width,
                y: normalizedPoint.y * projection.viewSize.height
            )
            let nearClip = projection.unproject(savedPoint, z: 0)
            let farClip = projection.unproject(savedPoint, z: 1)
            let near = nearClip.dehomogenized
            let far = farClip.dehomogenized
            let direction = (far - near).normalized()
            let denominator = plane.normal.dot(direction)
            guard abs(denominator) > 0.0001 else { return nil }

            let t = (plane.point - near).dot(plane.normal) / denominator
            guard t.isFinite, t >= 0 else { return nil }
            return near + direction * t
        }

        private func addPhotoMarker(at point: CGPoint, index: Int, in sceneView: ARSCNView) {
            let marker = EndpointMarkerView(frame: CGRect(x: point.x - 18, y: point.y - 18, width: 36, height: 36))
            marker.endpointIndex = index
            marker.backgroundColor = .clear
            marker.layer.cornerRadius = 18
            marker.isUserInteractionEnabled = true

            let pan = UIPanGestureRecognizer(target: self, action: #selector(handleMarkerPan(_:)))
            marker.addGestureRecognizer(pan)
            tapGesture?.require(toFail: pan)
            sceneView.addSubview(marker)
            markerViews.append(marker)
            photoDrawingViews.append(marker)
        }

        @objc private func handleMarkerPan(_ gesture: UIPanGestureRecognizer) {
            guard let marker = gesture.view as? EndpointMarkerView,
                  let sceneView,
                  marker.endpointIndex >= 0,
                  marker.endpointIndex < screenPoints.count,
                  marker.endpointIndex < points.count else {
                return
            }

            let location = gesture.location(in: sceneView)
            let clampedLocation = CGPoint(
                x: min(max(location.x, 0), sceneView.bounds.width),
                y: min(max(location.y, 0), sceneView.bounds.height)
            )

            guard let position = measurementWorldPosition(at: clampedLocation, in: sceneView) else { return }
            marker.center = clampedLocation
            screenPoints[marker.endpointIndex] = clampedLocation
            points[marker.endpointIndex] = position

            if points.count == 2 {
                updatePhotoMeasurementOverlay(in: sceneView)
            }
        }

        private func measurementWorldPosition(at location: CGPoint, in sceneView: ARSCNView) -> SCNVector3? {
            if let lockedMeasurementPlane, let activeProjection {
                return worldPosition(
                    at: location,
                    on: lockedMeasurementPlane,
                    projection: activeProjection,
                    currentSize: sceneView.bounds.size
                )
            }

            if let lockedMeasurementPlane {
                return worldPosition(at: location, on: lockedMeasurementPlane, in: sceneView)
            }

            if isRestoredPhoto {
                return nil
            }

            guard let plane = measurementPlane(at: location, in: sceneView),
                  let intersection = worldPosition(at: location, on: plane, in: sceneView) else {
                return nil
            }
            lockedMeasurementPlane = plane
            return intersection
        }

        private func updatePhotoMeasurementOverlay(in sceneView: ARSCNView) {
            photoDrawingViews
                .filter { !($0 is EndpointMarkerView) }
                .forEach { $0.removeFromSuperview() }
            photoDrawingViews.removeAll { !($0 is EndpointMarkerView) }

            guard points.count == 2, screenPoints.count == 2 else { return }
            let distance = points[0].distance(to: points[1])
            addPhotoLine(from: screenPoints[0], to: screenPoints[1], in: sceneView)
            addPhotoDistanceLabel(distance, from: screenPoints[0], to: screenPoints[1], in: sceneView)
            markerViews.forEach { sceneView.bringSubviewToFront($0) }
            onDistanceChanged(Double(distance))
        }

        private func addPhotoLine(from start: CGPoint, to end: CGPoint, in sceneView: ARSCNView) {
            let layer = CAShapeLayer()
            let path = UIBezierPath()
            path.move(to: start)
            path.addLine(to: end)
            layer.path = path.cgPath
            layer.strokeColor = UIColor.systemYellow.cgColor
            layer.lineWidth = 4
            layer.lineCap = .round

            let drawingView = UIView(frame: sceneView.bounds)
            drawingView.autoresizingMask = [.flexibleWidth, .flexibleHeight]
            drawingView.isUserInteractionEnabled = false
            drawingView.layer.addSublayer(layer)
            sceneView.addSubview(drawingView)
            photoDrawingViews.append(drawingView)
        }

        private func addPhotoDistanceLabel(_ distance: Float, from start: CGPoint, to end: CGPoint, in sceneView: ARSCNView) {
            let label = UILabel()
            label.text = "\(Double(distance * 100).formatted(.number.precision(.fractionLength(1)))) cm"
            label.font = .systemFont(ofSize: 17, weight: .semibold)
            label.textColor = .white
            label.backgroundColor = UIColor.black.withAlphaComponent(0.68)
            label.textAlignment = .center
            label.layer.cornerRadius = 8
            label.layer.masksToBounds = true
            label.isUserInteractionEnabled = false
            label.sizeToFit()

            let width = max(label.bounds.width + 18, 82)
            let height: CGFloat = 34
            let midpoint = CGPoint(x: (start.x + end.x) * 0.5, y: (start.y + end.y) * 0.5)
            label.frame = CGRect(
                x: midpoint.x - width * 0.5,
                y: midpoint.y - height - 12,
                width: width,
                height: height
            )
            sceneView.addSubview(label)
            photoDrawingViews.append(label)
        }

        private func addMarker(at position: SCNVector3) {
            let sphere = SCNSphere(radius: 0.007)
            sphere.firstMaterial?.diffuse.contents = UIColor.systemGreen
            sphere.firstMaterial?.emission.contents = UIColor.systemGreen
            let node = SCNNode(geometry: sphere)
            node.position = position
            sceneView?.scene.rootNode.addChildNode(node)
            nodes.append(node)
        }

        private func addLine(from start: SCNVector3, to end: SCNVector3) {
            let source = SCNGeometrySource(vertices: [start, end])
            let element = SCNGeometryElement(indices: [Int32(0), Int32(1)], primitiveType: .line)
            let geometry = SCNGeometry(sources: [source], elements: [element])
            geometry.firstMaterial?.diffuse.contents = UIColor.systemYellow
            geometry.firstMaterial?.emission.contents = UIColor.systemYellow
            let node = SCNNode(geometry: geometry)
            sceneView?.scene.rootNode.addChildNode(node)
            nodes.append(node)
        }

        private func addDistanceLabel(_ distance: Float, from start: SCNVector3, to end: SCNVector3) {
            let text = SCNText(
                string: "\(Double(distance * 100).formatted(.number.precision(.fractionLength(1)))) cm",
                extrusionDepth: 0.001
            )
            text.font = .systemFont(ofSize: 0.05, weight: .semibold)
            text.firstMaterial?.diffuse.contents = UIColor.white
            text.firstMaterial?.emission.contents = UIColor.white

            let node = SCNNode(geometry: text)
            node.scale = SCNVector3(0.18, 0.18, 0.18)
            node.position = SCNVector3(
                (start.x + end.x) * 0.5,
                (start.y + end.y) * 0.5 + 0.025,
                (start.z + end.z) * 0.5
            )

            let constraint = SCNBillboardConstraint()
            constraint.freeAxes = .all
            node.constraints = [constraint]

            sceneView?.scene.rootNode.addChildNode(node)
            nodes.append(node)
        }

        private func clearMeasurement() {
            nodes.forEach { $0.removeFromParentNode() }
            nodes.removeAll()
            points.removeAll()
            screenPoints.removeAll()
            markerViews.removeAll()
        }

        private func clearPhotoMeasurement() {
            photoDrawingViews.forEach { $0.removeFromSuperview() }
            photoDrawingViews.removeAll()
            clearMeasurement()
        }
    }
}

private extension SCNVector3 {
    func distance(to other: SCNVector3) -> Float {
        let dx = x - other.x
        let dy = y - other.y
        let dz = z - other.z
        return sqrt(dx * dx + dy * dy + dz * dz)
    }

    func dot(_ other: SCNVector3) -> Float {
        x * other.x + y * other.y + z * other.z
    }

    func normalized() -> SCNVector3 {
        let length = sqrt(x * x + y * y + z * z)
        guard length > 0 else { return self }
        return SCNVector3(x / length, y / length, z / length)
    }

    static func + (left: SCNVector3, right: SCNVector3) -> SCNVector3 {
        SCNVector3(left.x + right.x, left.y + right.y, left.z + right.z)
    }

    static func - (left: SCNVector3, right: SCNVector3) -> SCNVector3 {
        SCNVector3(left.x - right.x, left.y - right.y, left.z - right.z)
    }

    static func * (vector: SCNVector3, scalar: Float) -> SCNVector3 {
        SCNVector3(vector.x * scalar, vector.y * scalar, vector.z * scalar)
    }
}

private extension SIMD4<Float> {
    var dehomogenized: SCNVector3 {
        guard w != 0 else { return SCNVector3(x, y, z) }
        return SCNVector3(x / w, y / w, z / w)
    }
}

private extension CGPoint {
    func distance(to other: CGPoint) -> Double {
        let dx = x - other.x
        let dy = y - other.y
        return sqrt(dx * dx + dy * dy)
    }
}

private struct MeasurementPlane {
    let point: SCNVector3
    let normal: SCNVector3

    init(point: SCNVector3, normal: SCNVector3) {
        self.point = point
        self.normal = normal.normalized()
    }

    init(saved: SavedMeasurementPlane) {
        point = SCNVector3(saved.pointX, saved.pointY, saved.pointZ)
        normal = SCNVector3(saved.normalX, saved.normalY, saved.normalZ).normalized()
    }

    var saved: SavedMeasurementPlane {
        SavedMeasurementPlane(
            pointX: point.x,
            pointY: point.y,
            pointZ: point.z,
            normalX: normal.x,
            normalY: normal.y,
            normalZ: normal.z
        )
    }
}

private struct CameraProjection {
    let viewSize: CGSize
    let inverseViewProjection: simd_float4x4

    @MainActor
    init?(sceneView: ARSCNView) {
        guard let frame = sceneView.session.currentFrame else { return nil }
        let viewportSize = sceneView.bounds.size
        guard viewportSize.width > 0, viewportSize.height > 0 else { return nil }

        let orientation = UIWindow.interfaceOrientationForCurrentScene ?? .portrait
        let viewMatrix = frame.camera.viewMatrix(for: orientation)
        let projectionMatrix = frame.camera.projectionMatrix(
            for: orientation,
            viewportSize: viewportSize,
            zNear: 0.001,
            zFar: 100
        )
        viewSize = viewportSize
        inverseViewProjection = simd_inverse(projectionMatrix * viewMatrix)
    }

    init?(saved: SavedCameraProjection) {
        guard saved.projectionTransform.count == 16 else { return nil }
        viewSize = CGSize(width: saved.viewWidth, height: saved.viewHeight)
        inverseViewProjection = simd_float4x4(saved.projectionTransform)
    }

    var saved: SavedCameraProjection {
        SavedCameraProjection(
            viewWidth: viewSize.width,
            viewHeight: viewSize.height,
            viewToCameraTransform: [],
            projectionTransform: inverseViewProjection.values
        )
    }

    func unproject(_ point: CGPoint, z: CGFloat) -> SIMD4<Float> {
        let normalizedX = Float((point.x / max(viewSize.width, 1)) * 2 - 1)
        let normalizedY = Float(1 - (point.y / max(viewSize.height, 1)) * 2)
        let clip = SIMD4<Float>(normalizedX, normalizedY, Float(z * 2 - 1), 1)
        return inverseViewProjection * clip
    }
}

private extension simd_float4x4 {
    init(_ values: [Float]) {
        self.init(
            SIMD4<Float>(values[0], values[1], values[2], values[3]),
            SIMD4<Float>(values[4], values[5], values[6], values[7]),
            SIMD4<Float>(values[8], values[9], values[10], values[11]),
            SIMD4<Float>(values[12], values[13], values[14], values[15])
        )
    }

    var values: [Float] {
        [
            columns.0.x, columns.0.y, columns.0.z, columns.0.w,
            columns.1.x, columns.1.y, columns.1.z, columns.1.w,
            columns.2.x, columns.2.y, columns.2.z, columns.2.w,
            columns.3.x, columns.3.y, columns.3.z, columns.3.w
        ]
    }
}

private extension UIWindow {
    static var interfaceOrientationForCurrentScene: UIInterfaceOrientation? {
        UIApplication.shared.connectedScenes
            .compactMap { $0 as? UIWindowScene }
            .first { $0.activationState == .foregroundActive }?
            .interfaceOrientation
    }
}

private final class EndpointMarkerView: UIView {
    var endpointIndex = 0

    override init(frame: CGRect) {
        super.init(frame: frame)
        isOpaque = false
    }

    required init?(coder: NSCoder) {
        super.init(coder: coder)
        isOpaque = false
    }

    override func draw(_ rect: CGRect) {
        guard let context = UIGraphicsGetCurrentContext() else { return }
        let outerRect = CGRect(x: rect.midX - 8, y: rect.midY - 8, width: 16, height: 16)
        context.setShadow(offset: CGSize(width: 0, height: 2), blur: 5, color: UIColor.black.withAlphaComponent(0.24).cgColor)
        UIColor.white.setFill()
        context.fillEllipse(in: outerRect)
        context.setShadow(offset: .zero, blur: 0, color: nil)

        let innerRect = outerRect.insetBy(dx: 3, dy: 3)
        UIColor.systemGreen.setFill()
        context.fillEllipse(in: innerRect)
    }
}
