import ARKit
import SceneKit
import SwiftUI

struct ARMeasurementView: UIViewRepresentable {
    var isActive: Bool
    var onDistanceChanged: (Double) -> Void

    func makeCoordinator() -> Coordinator {
        Coordinator(onDistanceChanged: onDistanceChanged)
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

        if isActive {
            context.coordinator.startSession()
        }

        return view
    }

    func updateUIView(_ uiView: ARSCNView, context: Context) {
        context.coordinator.sceneView = uiView
        if isActive {
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
        private var sessionRunning = false
        private let onDistanceChanged: (Double) -> Void

        init(onDistanceChanged: @escaping (Double) -> Void) {
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

        func pauseSession() {
            sceneView?.session.pause()
            sessionRunning = false
        }

        @objc func handleTap(_ gesture: UITapGestureRecognizer) {
            guard let sceneView else { return }
            let location = gesture.location(in: sceneView)
            guard let position = worldPosition(at: location, in: sceneView) else { return }

            if points.count == 2 {
                clearMeasurement()
            }

            points.append(position)
            addMarker(at: position)

            if points.count == 2 {
                let distance = points[0].distance(to: points[1])
                addLine(from: points[0], to: points[1])
                addDistanceLabel(distance, from: points[0], to: points[1])
                onDistanceChanged(Double(distance))
            }
        }

        private func worldPosition(at point: CGPoint, in sceneView: ARSCNView) -> SCNVector3? {
            if let result = sceneView.raycastQuery(from: point, allowing: .estimatedPlane, alignment: .horizontal)
                .flatMap({ sceneView.session.raycast($0).first }) {
                let transform = result.worldTransform
                return SCNVector3(transform.columns.3.x, transform.columns.3.y, transform.columns.3.z)
            }

            guard let result = sceneView.raycastQuery(from: point, allowing: .estimatedPlane, alignment: .any)
                .flatMap({ sceneView.session.raycast($0).first }) else {
                return nil
            }
            let transform = result.worldTransform
            return SCNVector3(transform.columns.3.x, transform.columns.3.y, transform.columns.3.z)
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
}
