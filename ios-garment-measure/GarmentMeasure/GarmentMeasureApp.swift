import SwiftUI

@main
struct GarmentMeasureApp: App {
    @StateObject private var garment = GarmentMeasurementStore()

    var body: some Scene {
        WindowGroup {
            ContentView()
                .environmentObject(garment)
        }
    }
}
