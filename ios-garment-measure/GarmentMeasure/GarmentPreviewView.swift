import SceneKit
import SwiftUI
import UIKit

struct GarmentPreviewView: UIViewRepresentable {
    @ObservedObject var store: GarmentMeasurementStore
    var isExpanded: Bool = false
    var showsDimensionGuides: Bool = true
    var externalModelURL: URL?

    func makeCoordinator() -> Coordinator {
        Coordinator()
    }

    func makeUIView(context: Context) -> SCNView {
        let view = SCNView(frame: .zero)
        view.backgroundColor = previewBackgroundColor
        view.allowsCameraControl = true
        view.autoenablesDefaultLighting = false
        view.antialiasingMode = .multisampling4X
        view.preferredFramesPerSecond = 60
        context.coordinator.renderKey = renderKey
        view.scene = makeScene()
        return view
    }

    func updateUIView(_ uiView: SCNView, context: Context) {
        uiView.backgroundColor = previewBackgroundColor
        guard context.coordinator.renderKey != renderKey else { return }
        context.coordinator.renderKey = renderKey
        uiView.scene = nil
        uiView.scene = makeScene()
    }

    static func dismantleUIView(_ uiView: SCNView, coordinator: Coordinator) {
        uiView.scene = nil
        coordinator.renderKey = ""
    }

    final class Coordinator {
        var renderKey = ""
    }

    private var renderKey: String {
        let modelPath = externalModelURL?.path ?? store.generatedModelURL?.path ?? "none"
        let measurementKey = GarmentDimension.allCases
            .map { "\($0.rawValue):\(store.value(for: $0).rounded(toPlaces: 1))" }
            .joined(separator: "|")
        return [
            "rev:\(store.previewRevision)",
            modelPath,
            isExpanded ? "expanded" : "normal",
            showsDimensionGuides ? "guides-on" : "guides-off",
            measurementKey,
            store.generatedPreview == nil ? "preview-none" : "preview-image",
            store.generatedMesh == nil ? "mesh-none" : "mesh-present"
        ].joined(separator: "#")
    }

    private func makeScene() -> SCNScene {
        let scene = SCNScene()
        let hasExternalModel = externalModelURL != nil
        scene.background.contents = previewBackgroundColor
        scene.lightingEnvironment.contents = UIColor.white
        scene.lightingEnvironment.intensity = 0.82

        let camera = SCNCamera()
        camera.usesOrthographicProjection = true
        camera.orthographicScale = hasExternalModel
            ? (isExpanded ? 0.52 : 0.66)
            : (isExpanded ? 0.58 : 0.88)
        let cameraNode = SCNNode()
        cameraNode.camera = camera
        cameraNode.position = SCNVector3(0, hasExternalModel ? 0.0 : (isExpanded ? -0.08 : -0.02), 1.45)
        scene.rootNode.addChildNode(cameraNode)

        let keyLight = SCNLight()
        keyLight.type = .area
        keyLight.intensity = 1450
        keyLight.areaType = .rectangle
        keyLight.areaExtents = simd_float3(1.0, 0.85, 0.1)
        let keyNode = SCNNode()
        keyNode.light = keyLight
        keyNode.position = SCNVector3(-0.28, 0.48, 0.72)
        scene.rootNode.addChildNode(keyNode)

        let fillLight = SCNLight()
        fillLight.type = .ambient
        fillLight.intensity = 720
        fillLight.color = UIColor.white
        let fillNode = SCNNode()
        fillNode.light = fillLight
        scene.rootNode.addChildNode(fillNode)

        let frontLight = SCNLight()
        frontLight.type = .omni
        frontLight.intensity = 360
        frontLight.color = UIColor.white
        let frontNode = SCNNode()
        frontNode.light = frontLight
        frontNode.position = SCNVector3(0.12, 0.06, 1.0)
        scene.rootNode.addChildNode(frontNode)

        if store.generatedMesh == nil && !hasExternalModel {
            // Keep the preview white even before an external model is generated.
        }

        let root = SCNNode()
        root.position = SCNVector3(0, hasExternalModel ? 0.0 : (isExpanded ? -0.11 : -0.045), 0)
        root.eulerAngles.x = -.pi / 140
        scene.rootNode.addChildNode(root)

        if let generatedModelURL = externalModelURL, addExternalModel(generatedModelURL, to: root) {
            // Loaded external model.
        } else if store.generatedModelURL != nil {
            addTShirt(to: root)
        } else if let generatedMesh = store.generatedMesh {
            addGeneratedMesh(generatedMesh, to: root)
        } else if let generatedPreview = store.generatedPreview {
            addGeneratedPreview(generatedPreview, to: root)
        } else {
            addTShirt(to: root)
        }
        return scene
    }

    private var previewBackgroundColor: UIColor {
        .white
    }

    @discardableResult
    private func addExternalModel(_ url: URL, to root: SCNNode) -> Bool {
        do {
            let modelScene = try SCNScene(url: url, options: nil)
            let wrapper = SCNNode()
            for child in modelScene.rootNode.childNodes {
                wrapper.addChildNode(child.clone())
            }
            removeGeneratedBackdropPlanes(from: wrapper)
            guard containsRenderableGeometry(wrapper) else { return false }
            brightenMaterials(in: wrapper)
            normalizeModel(wrapper)
            applyMeasuredProportions(to: wrapper)
            wrapper.eulerAngles = SCNVector3(-Float.pi / 28, Float.pi / 10, 0)
            let extraScale = Float(isExpanded ? 1.24 : 1.08)
            wrapper.scale = SCNVector3(wrapper.scale.x * extraScale, wrapper.scale.y * extraScale, wrapper.scale.z * extraScale)
            root.addChildNode(wrapper)
            if showsDimensionGuides {
                let guideRoot = SCNNode()
                guideRoot.eulerAngles = wrapper.eulerAngles
                root.addChildNode(guideRoot)
                addDimensionGuides(to: guideRoot, dimensions: measuredDimensions(), modelScale: Double(extraScale))
            }
            return true
        } catch {
            return false
        }
    }

    private func brightenMaterials(in node: SCNNode) {
        if let geometry = node.geometry {
            for material in geometry.materials {
                material.lightingModel = .physicallyBased
                material.diffuse.intensity = 1.18
                material.ambient.contents = UIColor.white
                material.emission.contents = UIColor(white: 1, alpha: 0.035)
                material.isDoubleSided = true
            }
        }
        for child in node.childNodes {
            brightenMaterials(in: child)
        }
    }

    private func removeGeneratedBackdropPlanes(from node: SCNNode) {
        for child in node.childNodes {
            removeGeneratedBackdropPlanes(from: child)
            if isGeneratedBackdropPlane(child) {
                child.removeFromParentNode()
            }
        }
    }

    private func isGeneratedBackdropPlane(_ node: SCNNode) -> Bool {
        guard node.geometry != nil else { return false }
        let bounds = node.boundingBox
        let size = SCNVector3(
            abs(bounds.max.x - bounds.min.x),
            abs(bounds.max.y - bounds.min.y),
            abs(bounds.max.z - bounds.min.z)
        )
        let sorted = [size.x, size.y, size.z].sorted()
        let maxSide = max(sorted[2], 0.0001)
        let isVeryFlat = sorted[0] / maxSide < 0.025
        guard isVeryFlat else { return false }

        let materialLooksWhite = node.geometry?.materials.contains(where: isMostlyWhiteMaterial) == true
        return materialLooksWhite
    }

    private func containsRenderableGeometry(_ node: SCNNode) -> Bool {
        if node.geometry != nil {
            return true
        }
        return node.childNodes.contains(where: containsRenderableGeometry)
    }

    private func isMostlyWhiteMaterial(_ material: SCNMaterial) -> Bool {
        if let color = material.diffuse.contents as? UIColor {
            return color.isMostlyWhite
        }
        if let color = material.emission.contents as? UIColor {
            return color.isMostlyWhite
        }
        let name = material.name?.lowercased() ?? ""
        return name.contains("white") || name.contains("background") || name.contains("backdrop")
    }

    private func normalizeModel(_ node: SCNNode) {
        let bounds = node.boundingBox
        let minV = bounds.min
        let maxV = bounds.max
        let size = SCNVector3(maxV.x - minV.x, maxV.y - minV.y, maxV.z - minV.z)
        let maxSide = max(size.x, max(size.y, size.z))
        guard maxSide > 0 else { return }
        let scale = 0.9 / maxSide
        node.scale = SCNVector3(scale, scale, scale)
        let center = SCNVector3((minV.x + maxV.x) / 2, (minV.y + maxV.y) / 2, (minV.z + maxV.z) / 2)
        node.position = SCNVector3(-center.x * scale, -center.y * scale, -center.z * scale)
    }

    private func applyMeasuredProportions(to node: SCNNode) {
        let measured = measuredDimensions()
        let fallback = fallbackDimensions()
        let measuredWidth = max(measured.shoulderWidth, measured.chestWidth, measured.hemWidth)
        let fallbackWidth = max(fallback.shoulderWidth, fallback.chestWidth, fallback.hemWidth)
        let widthScale = measuredWidth / max(0.01, fallbackWidth)
        let lengthScale = measured.bodyLength / max(0.01, fallback.bodyLength)
        let depthScale = measured.chestWidth / max(0.01, fallback.chestWidth)

        node.scale = SCNVector3(
            node.scale.x * Float(widthScale.clamped(to: 0.72...1.35)),
            node.scale.y * Float(lengthScale.clamped(to: 0.72...1.35)),
            node.scale.z * Float(depthScale.clamped(to: 0.78...1.28))
        )
    }

    private func addGeneratedMesh(_ mesh: GeneratedGarmentMesh, to root: SCNNode) {
        let vectors = mesh.vertices.compactMap { item -> SCNVector3? in
            guard item.count >= 3 else { return nil }
            return SCNVector3(item[0], item[1], item[2])
        }
        guard !vectors.isEmpty else {
            addTShirt(to: root)
            return
        }

        let indices = mesh.triangles.flatMap { $0 }
        let source = SCNGeometrySource(vertices: vectors)
        let element = SCNGeometryElement(indices: indices, primitiveType: .triangles)
        let geometry = SCNGeometry(sources: [source], elements: [element])

        let material = SCNMaterial()
        let color = mesh.baseColor.count >= 3
            ? UIColor(red: CGFloat(mesh.baseColor[0]), green: CGFloat(mesh.baseColor[1]), blue: CGFloat(mesh.baseColor[2]), alpha: 1)
            : UIColor(red: 0.12, green: 0.38, blue: 0.28, alpha: 1)
        material.diffuse.contents = color
        material.roughness.contents = 0.88
        material.metalness.contents = 0
        material.lightingModel = .physicallyBased
        geometry.firstMaterial = material

        let node = SCNNode(geometry: geometry)
        let scale = Float(isExpanded ? 1.18 : 1.0)
        node.scale = SCNVector3(scale, scale, scale)
        node.eulerAngles = SCNVector3(-Float.pi / 18, Float.pi / 7, 0)
        root.addChildNode(node)

        let seamColor = UIColor.white.withAlphaComponent(0.28)
        addLine(to: root, from: SCNVector3(-0.24, -0.31, 0.08), to: SCNVector3(0.24, -0.31, 0.08), color: seamColor, radius: 0.0009)
        addLine(to: root, from: SCNVector3(-0.18, 0.2, 0.08), to: SCNVector3(0.18, 0.2, 0.08), color: seamColor, radius: 0.0009)
        addLine(to: root, from: SCNVector3(-0.18, 0.2, 0.08), to: SCNVector3(-0.25, 0.05, 0.08), color: seamColor, radius: 0.0008)
        addLine(to: root, from: SCNVector3(0.18, 0.2, 0.08), to: SCNVector3(0.25, 0.05, 0.08), color: seamColor, radius: 0.0008)
    }

    private func addGeneratedPreview(_ image: UIImage, to root: SCNNode) {
        let plane = SCNPlane(width: isExpanded ? 0.92 : 0.78, height: isExpanded ? 0.92 : 0.78)
        let material = SCNMaterial()
        material.diffuse.contents = image
        material.lightingModel = .constant
        plane.firstMaterial = material

        let node = SCNNode(geometry: plane)
        node.position = SCNVector3(0, -0.01, 0.02)
        root.addChildNode(node)

        let shadow = SCNPlane(width: isExpanded ? 0.96 : 0.82, height: isExpanded ? 0.96 : 0.82)
        let shadowMaterial = SCNMaterial()
        shadowMaterial.diffuse.contents = UIColor.black.withAlphaComponent(0.28)
        shadowMaterial.lightingModel = .constant
        shadow.firstMaterial = shadowMaterial
        let shadowNode = SCNNode(geometry: shadow)
        shadowNode.position = SCNVector3(0.025, -0.035, -0.02)
        root.addChildNode(shadowNode)
    }

    private func addTShirt(to root: SCNNode) {
        let dimensions = measuredDimensions()

        let cloth = SCNMaterial()
        cloth.diffuse.contents = fabricTexture(source: store.garmentTexture)
        cloth.diffuse.wrapS = .clamp
        cloth.diffuse.wrapT = .clamp
        cloth.roughness.contents = 0.95
        cloth.metalness.contents = 0
        cloth.lightingModel = .physicallyBased

        let shape = SCNShape(path: shirtPath(dimensions), extrusionDepth: 0.014)
        shape.chamferRadius = 0.003
        shape.chamferMode = .both
        shape.firstMaterial = cloth
        let body = SCNNode(geometry: shape)
        body.position.z = -0.007
        root.addChildNode(body)

        addSoftShadow(to: root, dimensions: dimensions)
        addOutline(to: root, dimensions: dimensions)
        addSeams(to: root, dimensions: dimensions)
        addCollar(to: root, dimensions: dimensions)
        if showsDimensionGuides {
            addDimensionGuides(to: root, dimensions: dimensions, modelScale: 1)
        }
    }

    private func shirtPath(_ d: ShirtDimensions) -> UIBezierPath {
        let topY = d.bodyLength / 2
        let bottomY = -d.bodyLength / 2
        let shoulderY = topY - max(0.045, d.bodyLength * 0.08)
        let underarmY = topY - max(0.16, d.bodyLength * 0.29)
        let shoulderHalf = d.shoulderWidth / 2
        let chestHalf = d.chestWidth / 2
        let hemHalf = d.hemWidth / 2
        let sleeveReach = max(0.12, d.sleeveLength * 1.06)
        let sleeveDrop = max(0.09, d.sleeveLength * 0.68)
        let cuffHeight = max(0.055, d.sleeveLength * 0.36)

        let path = UIBezierPath()
        path.move(to: CGPoint(x: -hemHalf, y: bottomY))
        path.addCurve(
            to: CGPoint(x: -chestHalf, y: underarmY),
            controlPoint1: CGPoint(x: -hemHalf * 0.98, y: bottomY + d.bodyLength * 0.30),
            controlPoint2: CGPoint(x: -chestHalf * 0.98, y: underarmY - d.bodyLength * 0.12)
        )
        path.addCurve(
            to: CGPoint(x: -shoulderHalf - sleeveReach * 0.68, y: shoulderY - sleeveDrop - cuffHeight),
            controlPoint1: CGPoint(x: -chestHalf - sleeveReach * 0.10, y: underarmY - 0.005),
            controlPoint2: CGPoint(x: -shoulderHalf - sleeveReach * 0.46, y: shoulderY - sleeveDrop - cuffHeight * 0.9)
        )
        path.addQuadCurve(
            to: CGPoint(x: -shoulderHalf - sleeveReach, y: shoulderY - sleeveDrop),
            controlPoint: CGPoint(x: -shoulderHalf - sleeveReach * 0.94, y: shoulderY - sleeveDrop - cuffHeight * 0.92)
        )
        path.addCurve(
            to: CGPoint(x: -shoulderHalf, y: shoulderY),
            controlPoint1: CGPoint(x: -shoulderHalf - sleeveReach * 0.64, y: shoulderY - sleeveDrop * 0.42),
            controlPoint2: CGPoint(x: -shoulderHalf - sleeveReach * 0.18, y: shoulderY - 0.018)
        )
        path.addCurve(
            to: CGPoint(x: -d.neckWidth * 0.48, y: topY - 0.005),
            controlPoint1: CGPoint(x: -shoulderHalf * 0.68, y: topY),
            controlPoint2: CGPoint(x: -d.neckWidth * 0.75, y: topY + 0.005)
        )
        path.addQuadCurve(
            to: CGPoint(x: d.neckWidth * 0.48, y: topY - 0.005),
            controlPoint: CGPoint(x: 0, y: topY - max(0.035, d.neckWidth * 0.26))
        )
        path.addCurve(
            to: CGPoint(x: shoulderHalf, y: shoulderY),
            controlPoint1: CGPoint(x: d.neckWidth * 0.75, y: topY + 0.005),
            controlPoint2: CGPoint(x: shoulderHalf * 0.68, y: topY)
        )
        path.addCurve(
            to: CGPoint(x: shoulderHalf + sleeveReach, y: shoulderY - sleeveDrop),
            controlPoint1: CGPoint(x: shoulderHalf + sleeveReach * 0.18, y: shoulderY - 0.018),
            controlPoint2: CGPoint(x: shoulderHalf + sleeveReach * 0.64, y: shoulderY - sleeveDrop * 0.42)
        )
        path.addQuadCurve(
            to: CGPoint(x: shoulderHalf + sleeveReach * 0.72, y: shoulderY - sleeveDrop - cuffHeight),
            controlPoint: CGPoint(x: shoulderHalf + sleeveReach * 0.94, y: shoulderY - sleeveDrop - cuffHeight * 0.92)
        )
        path.addCurve(
            to: CGPoint(x: chestHalf, y: underarmY),
            controlPoint1: CGPoint(x: shoulderHalf + sleeveReach * 0.46, y: shoulderY - sleeveDrop - cuffHeight * 0.9),
            controlPoint2: CGPoint(x: chestHalf + sleeveReach * 0.10, y: underarmY - 0.005)
        )
        path.addCurve(
            to: CGPoint(x: hemHalf, y: bottomY),
            controlPoint1: CGPoint(x: chestHalf * 0.98, y: underarmY - d.bodyLength * 0.12),
            controlPoint2: CGPoint(x: hemHalf * 0.98, y: bottomY + d.bodyLength * 0.30)
        )
        path.addQuadCurve(
            to: CGPoint(x: -hemHalf, y: bottomY),
            controlPoint: CGPoint(x: 0, y: bottomY - 0.016)
        )
        path.close()
        return path
    }

    private func addBackdrop(to root: SCNNode) {
        let plane = SCNPlane(width: 1.8, height: 1.5)
        let material = SCNMaterial()
        material.diffuse.contents = UIColor(red: 0.048, green: 0.055, blue: 0.066, alpha: 1)
        material.lightingModel = .constant
        plane.firstMaterial = material
        let node = SCNNode(geometry: plane)
        node.position = SCNVector3(0, -0.02, -0.12)
        root.addChildNode(node)
    }

    private func addSoftShadow(to root: SCNNode, dimensions d: ShirtDimensions) {
        let shadow = SCNPlane(width: CGFloat(max(d.shoulderWidth + d.sleeveLength * 2.1, d.hemWidth) * 1.18), height: CGFloat(d.bodyLength * 1.08))
        let material = SCNMaterial()
        material.diffuse.contents = UIColor.black.withAlphaComponent(0.24)
        material.lightingModel = .constant
        shadow.firstMaterial = material
        let node = SCNNode(geometry: shadow)
        node.position = SCNVector3(0.025, -0.035, -0.022)
        root.addChildNode(node)
    }

    private func addOutline(to root: SCNNode, dimensions d: ShirtDimensions) {
        let outlinePath = shirtPath(d).cgPath.copy(strokingWithWidth: 0.006, lineCap: .round, lineJoin: .round, miterLimit: 2)
        let shape = SCNShape(path: UIBezierPath(cgPath: outlinePath), extrusionDepth: 0.001)
        let material = SCNMaterial()
        material.diffuse.contents = UIColor(red: 0.045, green: 0.17, blue: 0.28, alpha: 0.9)
        material.lightingModel = .constant
        shape.firstMaterial = material
        let node = SCNNode(geometry: shape)
        node.position.z = 0.016
        root.addChildNode(node)
    }

    private func addCollar(to root: SCNNode, dimensions d: ShirtDimensions) {
        let topY = d.bodyLength / 2
        let collar = SCNTorus(ringRadius: CGFloat(max(0.045, d.neckWidth * 0.42)), pipeRadius: 0.0038)
        collar.firstMaterial?.diffuse.contents = UIColor(red: 0.87, green: 0.93, blue: 0.98, alpha: 1)
        collar.firstMaterial?.roughness.contents = 0.9

        let node = SCNNode(geometry: collar)
        node.position = SCNVector3(0, Float(topY - 0.024), 0.013)
        node.scale = SCNVector3(1.16, 0.42, 0.08)
        root.addChildNode(node)
    }

    private func addSeams(to root: SCNNode, dimensions d: ShirtDimensions) {
        let topY = d.bodyLength / 2
        let bottomY = -d.bodyLength / 2
        let shoulderY = topY - 0.035
        let underarmY = topY - max(0.15, d.bodyLength * 0.24)
        let shoulderHalf = d.shoulderWidth / 2
        let chestHalf = d.chestWidth / 2
        let sleeveReach = max(0.1, d.sleeveLength * 0.92)
        let sleeveDrop = max(0.07, d.sleeveLength * 0.52)

        let seamColor = UIColor(red: 0.78, green: 0.9, blue: 0.98, alpha: 0.5)
        addLine(to: root, from: SCNVector3(-Float(shoulderHalf * 0.82), Float(shoulderY - 0.012), 0.018), to: SCNVector3(-Float(chestHalf * 0.76), Float(underarmY + 0.018), 0.018), color: seamColor, radius: 0.0012)
        addLine(to: root, from: SCNVector3(Float(shoulderHalf * 0.82), Float(shoulderY - 0.012), 0.018), to: SCNVector3(Float(chestHalf * 0.76), Float(underarmY + 0.018), 0.018), color: seamColor, radius: 0.0012)
        addLine(to: root, from: SCNVector3(-Float(shoulderHalf + sleeveReach * 0.74), Float(shoulderY - sleeveDrop - 0.017), 0.018), to: SCNVector3(-Float(shoulderHalf + sleeveReach * 0.28), Float(shoulderY - sleeveDrop * 0.44), 0.018), color: seamColor, radius: 0.0012)
        addLine(to: root, from: SCNVector3(Float(shoulderHalf + sleeveReach * 0.74), Float(shoulderY - sleeveDrop - 0.017), 0.018), to: SCNVector3(Float(shoulderHalf + sleeveReach * 0.28), Float(shoulderY - sleeveDrop * 0.44), 0.018), color: seamColor, radius: 0.0012)
        addLine(to: root, from: SCNVector3(-Float(d.hemWidth / 2 * 0.84), Float(bottomY + 0.028), 0.018), to: SCNVector3(Float(d.hemWidth / 2 * 0.84), Float(bottomY + 0.028), 0.018), color: seamColor.withAlphaComponent(0.42), radius: 0.001)
    }

    private func addDimensionGuides(to root: SCNNode, dimensions d: ShirtDimensions, modelScale: Double) {
        let guideColor = UIColor(red: 0.04, green: 0.34, blue: 0.76, alpha: 0.92)
        let labelColor = UIColor(red: 0.02, green: 0.13, blue: 0.26, alpha: 0.98)
        let scale = max(0.85, min(1.28, modelScale))
        let bodyLength = d.bodyLength * scale
        let shoulderWidth = d.shoulderWidth * scale
        let chestWidth = d.chestWidth * scale
        let hemWidth = d.hemWidth * scale
        let leftSleeveLength = d.leftSleeveLength * scale
        let rightSleeveLength = d.rightSleeveLength * scale
        let neckWidth = d.neckWidth * scale
        let topY = bodyLength / 2
        let bottomY = -bodyLength / 2
        let shoulderY = topY - max(0.05, bodyLength * 0.08)
        let chestY = topY - max(0.15, bodyLength * 0.24)
        let hemY = bottomY + max(0.035, bodyLength * 0.055)
        let sleeveY = shoulderY - max(0.08, bodyLength * 0.16)
        let z = Float(isExpanded ? 0.08 : 0.07)

        addGuide(
            to: root,
            title: "肩宽 \(centimeters(d.shoulderWidth)) cm",
            from: SCNVector3(-Float(shoulderWidth / 2), Float(shoulderY), z),
            to: SCNVector3(Float(shoulderWidth / 2), Float(shoulderY), z),
            labelOffset: SCNVector3(0, 0.035, 0),
            color: guideColor,
            labelColor: labelColor
        )
        addGuide(
            to: root,
            title: "胸宽 \(centimeters(d.chestWidth)) cm",
            from: SCNVector3(-Float(chestWidth / 2), Float(chestY), z),
            to: SCNVector3(Float(chestWidth / 2), Float(chestY), z),
            labelOffset: SCNVector3(0, -0.035, 0),
            color: guideColor,
            labelColor: labelColor
        )
        addGuide(
            to: root,
            title: "下摆宽 \(centimeters(d.hemWidth)) cm",
            from: SCNVector3(-Float(hemWidth / 2), Float(hemY), z),
            to: SCNVector3(Float(hemWidth / 2), Float(hemY), z),
            labelOffset: SCNVector3(0, -0.04, 0),
            color: guideColor,
            labelColor: labelColor
        )
        addGuide(
            to: root,
            title: "衣长 \(centimeters(d.bodyLength)) cm",
            from: SCNVector3(Float(max(hemWidth, shoulderWidth) / 2 + 0.075), Float(bottomY), z),
            to: SCNVector3(Float(max(hemWidth, shoulderWidth) / 2 + 0.075), Float(topY), z),
            labelOffset: SCNVector3(0.07, 0, 0),
            color: guideColor,
            labelColor: labelColor
        )
        addGuide(
            to: root,
            title: "左袖 \(centimeters(d.leftSleeveLength)) cm",
            from: SCNVector3(-Float(shoulderWidth / 2), Float(shoulderY), z),
            to: SCNVector3(-Float(shoulderWidth / 2 + leftSleeveLength * 0.92), Float(sleeveY), z),
            labelOffset: SCNVector3(-0.04, -0.02, 0),
            color: guideColor,
            labelColor: labelColor
        )
        addGuide(
            to: root,
            title: "右袖 \(centimeters(d.rightSleeveLength)) cm",
            from: SCNVector3(Float(shoulderWidth / 2), Float(shoulderY), z),
            to: SCNVector3(Float(shoulderWidth / 2 + rightSleeveLength * 0.92), Float(sleeveY), z),
            labelOffset: SCNVector3(0.04, -0.02, 0),
            color: guideColor,
            labelColor: labelColor
        )
        addGuide(
            to: root,
            title: "领宽 \(centimeters(d.neckWidth)) cm",
            from: SCNVector3(-Float(neckWidth / 2), Float(topY - 0.035), z + 0.004),
            to: SCNVector3(Float(neckWidth / 2), Float(topY - 0.035), z + 0.004),
            labelOffset: SCNVector3(0, -0.032, 0),
            color: guideColor,
            labelColor: labelColor
        )
    }

    private func addGuide(
        to root: SCNNode,
        title: String,
        from start: SCNVector3,
        to end: SCNVector3,
        labelOffset: SCNVector3,
        color: UIColor,
        labelColor: UIColor
    ) {
        addLine(to: root, from: start, to: end, color: color, radius: 0.0012)
        addEndpoint(to: root, at: start, color: color)
        addEndpoint(to: root, at: end, color: color)

        let middle = SCNVector3(
            (start.x + end.x) / 2 + labelOffset.x,
            (start.y + end.y) / 2 + labelOffset.y,
            (start.z + end.z) / 2 + labelOffset.z
        )
        addText(title, to: root, at: middle, color: labelColor, size: 0.022)
    }

    private func addEndpoint(to root: SCNNode, at position: SCNVector3, color: UIColor) {
        let sphere = SCNSphere(radius: 0.004)
        sphere.firstMaterial?.diffuse.contents = color
        let node = SCNNode(geometry: sphere)
        node.position = position
        root.addChildNode(node)
    }

    private func addLine(to root: SCNNode, from start: SCNVector3, to end: SCNVector3, color: UIColor, radius: CGFloat) {
        let vector = SCNVector3(end.x - start.x, end.y - start.y, end.z - start.z)
        let length = CGFloat(sqrt(vector.x * vector.x + vector.y * vector.y + vector.z * vector.z))
        let cylinder = SCNCylinder(radius: radius, height: length)
        cylinder.firstMaterial?.diffuse.contents = color
        cylinder.firstMaterial?.emission.contents = color.withAlphaComponent(0.18)

        let node = SCNNode(geometry: cylinder)
        node.position = SCNVector3((start.x + end.x) / 2, (start.y + end.y) / 2, (start.z + end.z) / 2)
        node.eulerAngles = cylinderRotation(from: start, to: end)
        root.addChildNode(node)
    }

    private func addText(_ text: String, to root: SCNNode, at position: SCNVector3, color: UIColor, size: CGFloat) {
        let image = labelImage(text: text, textColor: color)
        let height = CGFloat(isExpanded ? 0.048 : 0.055)
        let width = height * image.size.width / max(1, image.size.height)
        let geometry = SCNPlane(width: width, height: height)
        let material = SCNMaterial()
        material.diffuse.contents = image
        material.lightingModel = .constant
        material.isDoubleSided = true
        material.readsFromDepthBuffer = false
        material.writesToDepthBuffer = false
        geometry.firstMaterial = material

        let node = SCNNode(geometry: geometry)
        node.position = SCNVector3(position.x, position.y, position.z + 0.045)
        node.constraints = [SCNBillboardConstraint()]
        node.renderingOrder = 1000
        root.addChildNode(node)
    }

    private func labelImage(text: String, textColor: UIColor) -> UIImage {
        let font = UIFont.systemFont(ofSize: 32, weight: .bold)
        let horizontalPadding: CGFloat = 18
        let verticalPadding: CGFloat = 10
        let textSize = (text as NSString).size(withAttributes: [.font: font])
        let size = CGSize(
            width: ceil(textSize.width + horizontalPadding * 2),
            height: ceil(textSize.height + verticalPadding * 2)
        )
        let renderer = UIGraphicsImageRenderer(size: size)
        return renderer.image { context in
            let rect = CGRect(origin: .zero, size: size)
            let path = UIBezierPath(roundedRect: rect, cornerRadius: 16)
            UIColor.white.withAlphaComponent(0.92).setFill()
            path.fill()
            UIColor(red: 0.04, green: 0.24, blue: 0.58, alpha: 0.18).setStroke()
            path.lineWidth = 2
            path.stroke()

            (text as NSString).draw(
                in: rect.insetBy(dx: horizontalPadding, dy: verticalPadding),
                withAttributes: [
                    .font: font,
                    .foregroundColor: textColor
                ]
            )
            context.cgContext.setBlendMode(.normal)
        }
    }

    private func cylinderRotation(from start: SCNVector3, to end: SCNVector3) -> SCNVector3 {
        let dx = end.x - start.x
        let dy = end.y - start.y
        let dz = end.z - start.z
        let yaw = atan2(dx, dy)
        let pitch = atan2(dz, sqrt(dx * dx + dy * dy))
        return SCNVector3(Float(pitch), 0, Float(-yaw))
    }

    private func meters(_ dimension: GarmentDimension) -> Double {
        store.value(for: dimension) / 100
    }

    private func measuredDimensions() -> ShirtDimensions {
        ShirtDimensions(
            bodyLength: meters(.bodyLength),
            shoulderWidth: meters(.shoulderWidth),
            chestWidth: meters(.chestWidth),
            hemWidth: meters(.hemWidth),
            leftSleeveLength: meters(.leftSleeveLength),
            rightSleeveLength: meters(.rightSleeveLength),
            neckWidth: meters(.neckWidth)
        )
    }

    private func fallbackDimensions() -> ShirtDimensions {
        ShirtDimensions(
            bodyLength: GarmentDimension.bodyLength.fallbackCentimeters / 100,
            shoulderWidth: GarmentDimension.shoulderWidth.fallbackCentimeters / 100,
            chestWidth: GarmentDimension.chestWidth.fallbackCentimeters / 100,
            hemWidth: GarmentDimension.hemWidth.fallbackCentimeters / 100,
            leftSleeveLength: GarmentDimension.leftSleeveLength.fallbackCentimeters / 100,
            rightSleeveLength: GarmentDimension.rightSleeveLength.fallbackCentimeters / 100,
            neckWidth: GarmentDimension.neckWidth.fallbackCentimeters / 100
        )
    }

    private func centimeters(_ meters: Double) -> String {
        (meters * 100).formatted(.number.precision(.fractionLength(1)))
    }

    private func fabricTexture(source: UIImage?) -> UIImage {
        let size = CGSize(width: source == nil ? 96 : 768, height: source == nil ? 96 : 768)
        let renderer = UIGraphicsImageRenderer(size: size)
        return renderer.image { context in
            let rect = CGRect(origin: .zero, size: size)

            if let source {
                UIColor.clear.setFill()
                context.fill(rect)
                source.draw(in: rect)

                UIColor.black.withAlphaComponent(0.04).setFill()
                context.fill(rect)
                UIColor.white.withAlphaComponent(0.05).setStroke()
                for x in stride(from: -size.width, through: size.width * 2, by: 18) {
                    let path = UIBezierPath()
                    path.move(to: CGPoint(x: x, y: 0))
                    path.addLine(to: CGPoint(x: x + size.width, y: size.height))
                    path.lineWidth = 1
                    path.stroke()
                }
            } else {
                UIColor(red: 0.12, green: 0.39, blue: 0.64, alpha: 1).setFill()
                context.fill(rect)

                UIColor(red: 0.16, green: 0.47, blue: 0.72, alpha: 0.32).setStroke()
                for x in stride(from: -96, through: 192, by: 8) {
                    let path = UIBezierPath()
                    path.move(to: CGPoint(x: x, y: 0))
                    path.addLine(to: CGPoint(x: x + 96, y: 96))
                    path.lineWidth = 1
                    path.stroke()
                }

                UIColor(red: 0.05, green: 0.18, blue: 0.31, alpha: 0.18).setStroke()
                for y in stride(from: 0, through: 96, by: 12) {
                    let path = UIBezierPath()
                    path.move(to: CGPoint(x: 0, y: y))
                    path.addLine(to: CGPoint(x: 96, y: y + 4))
                    path.lineWidth = 0.8
                    path.stroke()
                }
            }
        }
    }
}

private extension UIImage {
    func drawAspectFill(in rect: CGRect) {
        let imageAspect = size.width / max(1, size.height)
        let rectAspect = rect.width / max(1, rect.height)
        let drawSize: CGSize

        if imageAspect > rectAspect {
            drawSize = CGSize(width: rect.height * imageAspect, height: rect.height)
        } else {
            drawSize = CGSize(width: rect.width, height: rect.width / imageAspect)
        }

        let drawRect = CGRect(
            x: rect.midX - drawSize.width / 2,
            y: rect.midY - drawSize.height / 2,
            width: drawSize.width,
            height: drawSize.height
        )
        draw(in: drawRect)
    }
}

private extension UIColor {
    var isMostlyWhite: Bool {
        var red: CGFloat = 0
        var green: CGFloat = 0
        var blue: CGFloat = 0
        var alpha: CGFloat = 0
        guard getRed(&red, green: &green, blue: &blue, alpha: &alpha) else {
            return false
        }
        return alpha > 0.35 && red > 0.82 && green > 0.82 && blue > 0.82
    }
}

private struct ShirtDimensions {
    let bodyLength: Double
    let shoulderWidth: Double
    let chestWidth: Double
    let hemWidth: Double
    let leftSleeveLength: Double
    let rightSleeveLength: Double
    let neckWidth: Double

    var sleeveLength: Double {
        (leftSleeveLength + rightSleeveLength) / 2
    }
}

private extension Comparable {
    func clamped(to range: ClosedRange<Self>) -> Self {
        min(max(self, range.lowerBound), range.upperBound)
    }
}

private extension Double {
    func rounded(toPlaces places: Int) -> Double {
        let divisor = pow(10.0, Double(places))
        return (self * divisor).rounded() / divisor
    }
}
