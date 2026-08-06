import SwiftUI
import Foundation
import WebKit

struct BusinessHomeScreen: View {
    @Binding var showsMainTabBar: Bool
    @State private var path: [LibraryDestination] = []

    private let baseItems: [HomeModule] = [
        .init(title: "账户管理", icon: "person.fill", color: .businessBlue),
        .init(title: "客户管理", icon: "person.crop.circle.fill.badge.checkmark", color: .businessBlue),
        .init(title: "供货商管理", icon: "person.2.fill", color: .businessBlue),
        .init(title: "收款账户", icon: "creditcard.fill", color: .businessBlue),
        .init(title: "仓库管理", icon: "building.columns.fill", color: .businessBlue)
    ]

    private let libraryItems: [HomeModule] = [
        .init(title: "产品库", icon: "shippingbox.fill", color: .coralRed, destination: .productLibrary),
        .init(title: "色卡库", icon: "swatchpalette.fill", color: .coralRed, destination: .colorCardLibrary)
    ]

    private let salesItems: [HomeModule] = [
        .init(title: "订单列表", icon: "doc.text.fill", color: .businessGreen),
        .init(title: "销售开单", icon: "square.and.pencil", color: .businessGreen),
        .init(title: "销售退货", icon: "doc.badge.arrow.up.fill", color: .businessLightGreen),
        .init(title: "订单跟踪", icon: "chart.bar.xaxis", color: .businessGreen)
    ]

    private let purchaseItems: [HomeModule] = [
        .init(title: "采购订单", icon: "basket.fill", color: .businessOrange),
        .init(title: "采购开单", icon: "square.and.pencil", color: .businessOrange),
        .init(title: "采购退货", icon: "arrow.uturn.backward.square.fill", color: .businessOrange),
        .init(title: "采购跟踪", icon: "chart.bar.xaxis", color: .businessOrange)
    ]

    private let warehouseItems: [HomeModule] = [
        .init(title: "入仓单", icon: "basket.fill", color: .businessCyan),
        .init(title: "退供货商", icon: "doc.text.fill", color: .businessCyan)
    ]

    private let financeItems: [HomeModule] = [
        .init(title: "应收款", icon: "yensign.square.fill", color: .businessBlue),
        .init(title: "应付款", icon: "creditcard.fill", color: .businessBlue),
        .init(title: "收支明细", icon: "list.bullet.rectangle.fill", color: .businessBlue),
        .init(title: "利润报表", icon: "chart.line.uptrend.xyaxis", color: .businessBlue)
    ]

    var body: some View {
        NavigationStack(path: $path) {
            ZStack {
                Color(red: 0.95, green: 0.97, blue: 0.97).ignoresSafeArea()

                VStack(spacing: 0) {
                    ScrollView(showsIndicators: false) {
                        VStack(spacing: 10) {
                            HomeSection(title: "基础资料", modules: baseItems)
                            HomeSection(title: "资料库", modules: libraryItems)
                            HomeSection(title: "销售", modules: salesItems)
                            HomeSection(title: "采购", modules: purchaseItems)
                            HomeSection(title: "仓库", modules: warehouseItems)
                            HomeSection(title: "财务", modules: financeItems)
                        }
                        .padding(.horizontal, 10)
                        .padding(.top, 9)
                        .padding(.bottom, 122)
                    }
                }
            }
            .navigationBarHidden(true)
            .navigationDestination(for: LibraryDestination.self) { destination in
                switch destination {
                case .library:
                    LibraryHubScreen()
                case .productLibrary:
                    ProductLibraryScreen()
                case .colorCardLibrary:
                    ColorCardLibraryScreen()
                }
            }
        }
        .onAppear {
            showsMainTabBar = path.isEmpty
        }
        .onChange(of: path) { _, newPath in
            showsMainTabBar = newPath.isEmpty
        }
    }
}

private struct HomeSection: View {
    let title: String
    let modules: [HomeModule]

    var body: some View {
        VStack(alignment: .leading, spacing: 19) {
            Text(title)
                .font(.system(size: 19, weight: .bold))
                .foregroundStyle(Color(red: 0.53, green: 0.58, blue: 0.61))
                .padding(.top, 12)
                .padding(.leading, 14)

            VStack(spacing: 25) {
                ForEach(moduleRows.indices, id: \.self) { rowIndex in
                    HStack(spacing: 0) {
                        ForEach(moduleRows[rowIndex]) { module in
                            HomeModuleCell(module: module)
                                .frame(maxWidth: .infinity)
                        }

                        ForEach(0..<emptySlots(in: moduleRows[rowIndex]), id: \.self) { _ in
                            Spacer()
                                .frame(maxWidth: .infinity)
                                .frame(height: 79)
                        }
                    }
                }
            }
            .padding(.horizontal, 20)
            .padding(.bottom, 23)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Color.white, in: RoundedRectangle(cornerRadius: 7))
        .shadow(color: .black.opacity(0.035), radius: 8, y: 2)
    }

    private var moduleRows: [[HomeModule]] {
        stride(from: 0, to: modules.count, by: 4).map { start in
            Array(modules[start..<min(start + 4, modules.count)])
        }
    }

    private func emptySlots(in row: [HomeModule]) -> Int {
        max(0, 4 - row.count)
    }
}

private struct HomeModuleCell: View {
    let module: HomeModule

    var body: some View {
        if let destination = module.destination {
            NavigationLink(value: destination) {
                HomeModuleButton(module: module)
            }
            .buttonStyle(.plain)
        } else {
            HomeModuleButton(module: module)
        }
    }
}

private struct HomeModuleButton: View {
    let module: HomeModule

    var body: some View {
        VStack(spacing: 8) {
            RoundedRectangle(cornerRadius: 8)
                .fill(module.color)
                .frame(width: 50, height: 50)
                .overlay(
                    Image(systemName: module.icon)
                        .font(.system(size: 25, weight: .semibold))
                        .symbolRenderingMode(.hierarchical)
                        .foregroundStyle(.white)
                )

            Text(module.title)
                .font(.system(size: 15, weight: .semibold))
                .foregroundStyle(Color(red: 0.16, green: 0.17, blue: 0.18))
                .lineLimit(1)
                .minimumScaleFactor(0.72)
                .frame(width: 72, height: 20)
        }
        .frame(width: 72, height: 79)
    }
}

private struct LibraryHubScreen: View {
    var body: some View {
        List {
            NavigationLink(value: LibraryDestination.productLibrary) {
                LibraryRow(icon: "shippingbox.fill", color: .coralRed, title: "产品库", subtitle: "对应 H5 款库检索与管理款库")
            }
            NavigationLink(value: LibraryDestination.colorCardLibrary) {
                LibraryRow(icon: "swatchpalette.fill", color: .coralRed, title: "色卡库", subtitle: "对应 H5 管理色卡与相似色号")
            }
        }
        .navigationTitle("资料库")
    }
}

private struct LibraryRow: View {
    let icon: String
    let color: Color
    let title: String
    let subtitle: String

    var body: some View {
        HStack(spacing: 12) {
            RoundedRectangle(cornerRadius: 8)
                .fill(color)
                .frame(width: 46, height: 46)
                .overlay(
                    Image(systemName: icon)
                        .foregroundStyle(.white)
                        .font(.system(size: 23, weight: .semibold))
                )

            VStack(alignment: .leading, spacing: 4) {
                Text(title)
                    .font(.system(size: 17, weight: .bold))
                Text(subtitle)
                    .font(.system(size: 13))
                    .foregroundStyle(.secondary)
            }
        }
        .padding(.vertical, 5)
    }
}

private struct ProductLibraryScreen: View {
    var body: some View {
        CatalogH5Screen(type: "product", title: "产品库")
    }
}

private struct ProductTile: View {
    let product: CatalogProduct

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            AsyncImage(url: product.coverImageURL) { phase in
                switch phase {
                case .success(let image):
                    image
                        .resizable()
                        .scaledToFill()
                case .failure:
                    Color(red: 0.91, green: 0.94, blue: 0.97)
                        .overlay(
                            Image(systemName: "photo")
                                .font(.system(size: 28, weight: .semibold))
                                .foregroundStyle(.secondary)
                        )
                case .empty:
                    Color(red: 0.94, green: 0.96, blue: 0.98)
                        .overlay(ProgressView())
                @unknown default:
                    Color(red: 0.94, green: 0.96, blue: 0.98)
                }
            }
            .frame(height: 100)
            .clipped()

            VStack(alignment: .leading, spacing: 5) {
                Text(product.styleCode)
                    .font(.system(size: 12, weight: .bold))
                    .lineLimit(1)
                Text("\(product.imageCount) 张")
                    .font(.system(size: 11))
                    .foregroundStyle(.secondary)
                Text(product.tags.joined(separator: " "))
                    .font(.system(size: 10, weight: .semibold))
                    .foregroundStyle(Color(red: 0.22, green: 0.19, blue: 0.64))
                    .lineLimit(2)
            }
            .padding(7)
        }
        .background(Color.white)
        .clipShape(RoundedRectangle(cornerRadius: 7))
        .shadow(color: .black.opacity(0.06), radius: 7, y: 3)
    }
}

private struct ColorCardLibraryScreen: View {
    var body: some View {
        CatalogH5Screen(type: "color", title: "色卡库")
    }
}

private struct CatalogH5Screen: View {
    @Environment(\.dismiss) private var dismiss
    @State private var canGoBack = false
    @State private var backRequestID = 0

    let type: String
    let title: String

    var body: some View {
        CatalogWebView(url: catalogURL, canGoBack: $canGoBack, backRequestID: backRequestID)
            .ignoresSafeArea(edges: .bottom)
            .navigationTitle(title)
            .navigationBarTitleDisplayMode(.inline)
            .navigationBarBackButtonHidden(true)
            .toolbar {
                ToolbarItem(placement: .topBarLeading) {
                    Button {
                        if canGoBack {
                            backRequestID += 1
                        } else {
                            dismiss()
                        }
                    } label: {
                        Image(systemName: "chevron.left")
                            .font(.system(size: 17, weight: .semibold))
                    }
                    .accessibilityLabel("返回")
                }
            }
    }

    private var catalogURL: URL {
        var components = URLComponents(string: "https://api.openfire.cloud/catalog")!
        var queryItems = [URLQueryItem(name: "type", value: type)]
        let token = UserDefaults.standard.string(forKey: "CatalogToken") ?? ""
        if !token.isEmpty {
            queryItems.append(URLQueryItem(name: "token", value: token))
        }
        components.queryItems = queryItems
        return components.url!
    }
}

private struct CatalogWebView: UIViewRepresentable {
    let url: URL
    @Binding var canGoBack: Bool
    let backRequestID: Int

    func makeCoordinator() -> Coordinator {
        Coordinator(canGoBack: $canGoBack)
    }

    func makeUIView(context: Context) -> WKWebView {
        let configuration = WKWebViewConfiguration()
        configuration.allowsInlineMediaPlayback = true
        let webView = WKWebView(frame: .zero, configuration: configuration)
        webView.uiDelegate = context.coordinator
        webView.navigationDelegate = context.coordinator
        webView.scrollView.contentInsetAdjustmentBehavior = .automatic
        webView.load(URLRequest(url: url))
        context.coordinator.loadedInitialURL = true
        context.coordinator.lastBackRequestID = backRequestID
        return webView
    }

    func updateUIView(_ webView: WKWebView, context: Context) {
        context.coordinator.canGoBack = $canGoBack
        if context.coordinator.lastBackRequestID != backRequestID {
            context.coordinator.lastBackRequestID = backRequestID
            if webView.canGoBack {
                webView.goBack()
            }
            return
        }

        if !context.coordinator.loadedInitialURL {
            context.coordinator.loadedInitialURL = true
            webView.load(URLRequest(url: url))
        }
    }

    final class Coordinator: NSObject, WKUIDelegate, WKNavigationDelegate {
        var canGoBack: Binding<Bool>
        var loadedInitialURL = false
        var lastBackRequestID = 0

        init(canGoBack: Binding<Bool>) {
            self.canGoBack = canGoBack
        }

        func webView(
            _ webView: WKWebView,
            createWebViewWith configuration: WKWebViewConfiguration,
            for navigationAction: WKNavigationAction,
            windowFeatures: WKWindowFeatures
        ) -> WKWebView? {
            guard navigationAction.targetFrame == nil, let url = navigationAction.request.url else {
                return nil
            }
            webView.load(URLRequest(url: url))
            canGoBack.wrappedValue = true
            return nil
        }

        func webView(
            _ webView: WKWebView,
            decidePolicyFor navigationAction: WKNavigationAction,
            decisionHandler: @escaping (WKNavigationActionPolicy) -> Void
        ) {
            if navigationAction.targetFrame == nil, let url = navigationAction.request.url {
                webView.load(URLRequest(url: url))
                decisionHandler(.cancel)
                return
            }
            decisionHandler(.allow)
        }

        func webView(_ webView: WKWebView, didFinish navigation: WKNavigation!) {
            canGoBack.wrappedValue = webView.canGoBack
        }

        func webView(_ webView: WKWebView, didCommit navigation: WKNavigation!) {
            canGoBack.wrappedValue = webView.canGoBack
        }
    }
}

private struct LabBox: View {
    let label: String
    let value: String

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(label)
                .font(.system(size: 12))
                .foregroundStyle(.secondary)
            Text(value)
                .font(.system(size: 20, weight: .heavy))
                .lineLimit(1)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(10)
        .background(Color(red: 0.97, green: 0.98, blue: 0.99), in: RoundedRectangle(cornerRadius: 8))
        .overlay(RoundedRectangle(cornerRadius: 8).stroke(Color(red: 0.84, green: 0.87, blue: 0.91)))
    }
}

private struct ColorMatchRow: View {
    let match: ColorCardMatch

    var body: some View {
        HStack {
            VStack(alignment: .leading, spacing: 4) {
                Text("名称：\(match.name)")
                    .font(.system(size: 14, weight: .bold))
                    .lineLimit(1)
                Text("色彩库：\(match.libraryName)")
                    .font(.system(size: 12))
                    .lineLimit(1)
                Text("dE*00：\(match.deltaText) · \(match.labText)")
                    .font(.system(size: 12))
                    .lineLimit(1)
            }
            Spacer()
            Text(match.hexDisplay)
                .font(.system(size: 12, weight: .bold))
        }
        .padding(11)
        .background(match.color, in: RoundedRectangle(cornerRadius: 8))
        .foregroundStyle(match.isDark ? .white : .black)
    }
}

private struct FlowLayout<Content: View>: View {
    let spacing: CGFloat
    @ViewBuilder let content: Content

    init(spacing: CGFloat, @ViewBuilder content: () -> Content) {
        self.spacing = spacing
        self.content = content()
    }

    var body: some View {
        LazyVGrid(columns: Array(repeating: GridItem(.flexible(), spacing: spacing), count: 4), alignment: .leading, spacing: spacing) {
            content
        }
    }
}

@MainActor
private final class ProductLibraryViewModel: ObservableObject {
    @Published var tags: [String] = []
    @Published var selectedTags: Set<String> = []
    @Published var products: [CatalogProduct] = []
    @Published var isLoadingTags = false
    @Published var isLoadingProducts = false
    @Published var isLoadingMore = false
    @Published var hasMore = false
    @Published var errorMessage: String?

    private var offset = 0
    private let limit = 21

    func loadInitial(query: String) async {
        guard tags.isEmpty && products.isEmpty else { return }
        await loadTags()
        await loadProducts(query: query, reset: true)
    }

    func loadTags() async {
        isLoadingTags = true
        defer { isLoadingTags = false }
        do {
            let response = try await CatalogAPIClient.shared.fetchCatalogTags()
            tags = response.displayTags
        } catch {
            errorMessage = CatalogAPIClient.userFacingMessage(error)
        }
    }

    func loadProducts(query: String, reset: Bool) async {
        if reset {
            offset = 0
            products.removeAll()
        }
        isLoadingProducts = true
        errorMessage = nil
        defer { isLoadingProducts = false }
        do {
            let response = try await CatalogAPIClient.shared.fetchCatalogProducts(
                query: query,
                tags: Array(selectedTags),
                limit: limit,
                offset: offset
            )
            products = response.products
            offset = products.count
            hasMore = response.hasMore(productsLoaded: products.count, limit: limit)
        } catch {
            errorMessage = CatalogAPIClient.userFacingMessage(error)
        }
    }

    func loadMore(query: String) async {
        guard !isLoadingMore else { return }
        isLoadingMore = true
        defer { isLoadingMore = false }
        do {
            let response = try await CatalogAPIClient.shared.fetchCatalogProducts(
                query: query,
                tags: Array(selectedTags),
                limit: limit,
                offset: offset
            )
            products.append(contentsOf: response.products)
            offset = products.count
            hasMore = response.hasMore(productsLoaded: products.count, limit: limit)
        } catch {
            errorMessage = CatalogAPIClient.userFacingMessage(error)
        }
    }

    func toggleTag(_ tag: String) {
        if selectedTags.contains(tag) {
            selectedTags.remove(tag)
        } else {
            selectedTags.insert(tag)
        }
    }

    func clearFilters() {
        selectedTags.removeAll()
    }
}

@MainActor
private final class ColorCardLibraryViewModel: ObservableObject {
    @Published var libraries: [ColorCardLibrary] = []
    @Published var selectedLibraryID = ""
    @Published var matches: [ColorCardMatch] = []
    @Published var isMatching = false
    @Published var errorMessage: String?
    @Published var previewHex = "#B7A06A"
    @Published var previewColor = Color(red: 0.72, green: 0.63, blue: 0.42)

    func loadLibraries() async {
        do {
            libraries = try await CatalogAPIClient.shared.fetchColorCardLibraries().libraries
        } catch {
            errorMessage = CatalogAPIClient.userFacingMessage(error)
        }
    }

    func match(l: String, a: String, b: String) async {
        guard
            let lValue = Double(l.trimmingCharacters(in: .whitespacesAndNewlines)),
            let aValue = Double(a.trimmingCharacters(in: .whitespacesAndNewlines)),
            let bValue = Double(b.trimmingCharacters(in: .whitespacesAndNewlines))
        else {
            errorMessage = "请输入合法的 Lab 数值"
            return
        }

        isMatching = true
        errorMessage = nil
        defer { isMatching = false }
        do {
            let response = try await CatalogAPIClient.shared.matchColorCards(
                l: lValue,
                a: aValue,
                b: bValue,
                libraryID: selectedLibraryID
            )
            matches = response.matches
            if let first = response.matches.first {
                previewHex = first.hexDisplay
                previewColor = first.color
            }
        } catch {
            errorMessage = CatalogAPIClient.userFacingMessage(error)
        }
    }
}

@MainActor
private final class CatalogAPIClient {
    static let shared = CatalogAPIClient()

    private let baseURL = URL(string: "https://api.openfire.cloud")!
    private let session: URLSession

    private init(session: URLSession = .shared) {
        self.session = session
    }

    func fetchCatalogTags() async throws -> CatalogTagsResponse {
        try await get("/api/v1/catalog/tags", queryItems: [])
    }

    func fetchCatalogProducts(query: String, tags: [String], limit: Int, offset: Int) async throws -> CatalogProductsResponse {
        var queryItems = [
            URLQueryItem(name: "limit", value: String(limit)),
            URLQueryItem(name: "offset", value: String(offset))
        ]
        let trimmedQuery = query.trimmingCharacters(in: .whitespacesAndNewlines)
        if !trimmedQuery.isEmpty {
            queryItems.append(URLQueryItem(name: "style_code", value: trimmedQuery))
        }
        if !tags.isEmpty {
            queryItems.append(URLQueryItem(name: "tags", value: tags.sorted().joined(separator: ",")))
        }
        return try await get("/api/v1/catalog/products", queryItems: queryItems)
    }

    func fetchColorCardLibraries() async throws -> ColorCardLibrariesResponse {
        try await get("/api/v1/color-card/libraries", queryItems: [])
    }

    func matchColorCards(l: Double, a: Double, b: Double, libraryID: String) async throws -> ColorCardMatchResponse {
        var payload: [String: Any] = [
            "L": l,
            "a": a,
            "b": b,
            "limit": 10
        ]
        if !libraryID.isEmpty {
            payload["library_id"] = libraryID
        }
        return try await post("/api/v1/color-card/match", payload: payload)
    }

    private func get<T: Decodable>(_ path: String, queryItems: [URLQueryItem]) async throws -> T {
        var components = URLComponents(url: baseURL.appendingPathComponent(path), resolvingAgainstBaseURL: false)!
        components.queryItems = queryItems.isEmpty ? nil : queryItems
        guard let url = components.url else { throw CatalogAPIError.invalidURL }
        var request = URLRequest(url: url)
        request.httpMethod = "GET"
        applyHeaders(to: &request)
        return try await decode(request)
    }

    private func post<T: Decodable>(_ path: String, payload: [String: Any]) async throws -> T {
        let url = baseURL.appendingPathComponent(path)
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.httpBody = try JSONSerialization.data(withJSONObject: payload)
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        applyHeaders(to: &request)
        return try await decode(request)
    }

    private func applyHeaders(to request: inout URLRequest) {
        request.setValue("replace-with-real-key-a", forHTTPHeaderField: "X-API-Key")
        let token = UserDefaults.standard.string(forKey: "CatalogToken") ?? ""
        if !token.isEmpty {
            request.setValue(token, forHTTPHeaderField: "X-Catalog-Token")
            request.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
        }
    }

    private func decode<T: Decodable>(_ request: URLRequest) async throws -> T {
        let (data, response) = try await session.data(for: request)
        guard let httpResponse = response as? HTTPURLResponse else {
            throw CatalogAPIError.invalidResponse
        }
        guard (200..<300).contains(httpResponse.statusCode) else {
            let detail = (try? JSONDecoder().decode(APIErrorResponse.self, from: data).detail) ?? "HTTP \(httpResponse.statusCode)"
            throw CatalogAPIError.server(detail)
        }
        return try JSONDecoder().decode(T.self, from: data)
    }

    static func userFacingMessage(_ error: Error) -> String {
        if let apiError = error as? CatalogAPIError {
            switch apiError {
            case .invalidURL:
                return "接口地址无效"
            case .invalidResponse:
                return "服务返回异常"
            case .server(let detail):
                return detail
            }
        }
        return error.localizedDescription
    }
}

private enum CatalogAPIError: Error {
    case invalidURL
    case invalidResponse
    case server(String)
}

private struct APIErrorResponse: Decodable {
    let detail: String?
}

private struct CatalogTagsResponse: Decodable {
    let tags: [String]?
    let tagGroups: TagGroups?

    var displayTags: [String] {
        let grouped = (tagGroups?.year ?? []) + (tagGroups?.category ?? []) + (tagGroups?.subcategory ?? [])
        let source = grouped.isEmpty ? (tags ?? []) : grouped
        return Array(NSOrderedSet(array: source.map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }.filter { !$0.isEmpty })) as? [String] ?? []
    }

    private enum CodingKeys: String, CodingKey {
        case tags
        case tagGroups = "tag_groups"
    }
}

private struct TagGroups: Decodable {
    let year: [String]?
    let category: [String]?
    let subcategory: [String]?
}

private struct CatalogProductsResponse: Decodable {
    let products: [CatalogProduct]
    let total: Int?
    let hasMore: Bool?

    private enum CodingKeys: String, CodingKey {
        case products
        case total
        case hasMore = "has_more"
    }

    func hasMore(productsLoaded: Int, limit: Int) -> Bool {
        if let hasMore { return hasMore }
        if let total { return productsLoaded < total }
        return products.count >= limit
    }
}

private struct CatalogProduct: Decodable, Identifiable {
    var id: String { styleCode }
    let styleCode: String
    let imageCount: Int
    let coverImageUrl: String?
    let tags: [String]

    var coverImageURL: URL? {
        guard let coverImageUrl, !coverImageUrl.isEmpty else { return nil }
        return URL(string: coverImageUrl)
    }

    private enum CodingKeys: String, CodingKey {
        case styleCode = "style_code"
        case styleCodeCamel = "styleCode"
        case imageCount = "image_count"
        case imageCountCamel = "imageCount"
        case coverImageUrl = "cover_image_url"
        case coverImageUrlCamel = "coverImageUrl"
        case tags
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        styleCode = (try? container.decode(String.self, forKey: .styleCode)) ?? (try? container.decode(String.self, forKey: .styleCodeCamel)) ?? ""
        imageCount = (try? container.decode(Int.self, forKey: .imageCount)) ?? (try? container.decode(Int.self, forKey: .imageCountCamel)) ?? 0
        coverImageUrl = (try? container.decode(String.self, forKey: .coverImageUrl)) ?? (try? container.decode(String.self, forKey: .coverImageUrlCamel))
        tags = (try? container.decode([String].self, forKey: .tags)) ?? []
    }
}

private struct ColorCardLibrariesResponse: Decodable {
    let libraries: [ColorCardLibrary]
}

private struct ColorCardLibrary: Decodable, Identifiable {
    let id: String
    let name: String
    let colorCount: Int

    private enum CodingKeys: String, CodingKey {
        case id
        case name
        case colorCount = "color_count"
        case colorCountCamel = "colorCount"
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        id = (try? container.decode(String.self, forKey: .id)) ?? ""
        name = (try? container.decode(String.self, forKey: .name)) ?? id
        colorCount = (try? container.decode(Int.self, forKey: .colorCount)) ?? (try? container.decode(Int.self, forKey: .colorCountCamel)) ?? 0
    }
}

private struct ColorCardMatchResponse: Decodable {
    let matches: [ColorCardMatch]
}

private struct ColorCardMatch: Decodable, Identifiable {
    let id: Int
    let name: String
    let libraryName: String
    let delta: Double?
    let l: Double?
    let a: Double?
    let b: Double?
    let hex: String?

    var deltaText: String {
        guard let delta else { return "-" }
        return String(format: "%.2f", delta)
    }

    var labText: String {
        let parts = [
            l.map { "L \(String(format: "%.1f", $0))" },
            a.map { "a \(String(format: "%.1f", $0))" },
            b.map { "b \(String(format: "%.1f", $0))" }
        ].compactMap { $0 }
        return parts.joined(separator: " ")
    }

    var hexDisplay: String {
        let cleaned = (hex ?? "").trimmingCharacters(in: CharacterSet(charactersIn: "#"))
        return cleaned.isEmpty ? "#B7A06A" : "#\(cleaned.uppercased())"
    }

    var color: Color {
        Color(hex: hexDisplay) ?? Color(red: 0.72, green: 0.63, blue: 0.42)
    }

    var isDark: Bool {
        guard let rgb = RGB(hex: hexDisplay) else { return true }
        return (0.299 * Double(rgb.r) + 0.587 * Double(rgb.g) + 0.114 * Double(rgb.b)) < 150
    }

    private enum CodingKeys: String, CodingKey {
        case id
        case name
        case libraryName = "library_name"
        case libraryNameCamel = "libraryName"
        case delta
        case deltaE = "delta_e"
        case deltaE00 = "delta_e00"
        case l = "L"
        case lowerL = "l"
        case a
        case b
        case hex
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        id = (try? container.decode(Int.self, forKey: .id)) ?? Int.random(in: 1...Int.max)
        name = (try? container.decode(String.self, forKey: .name)) ?? ""
        libraryName = (try? container.decode(String.self, forKey: .libraryName)) ?? (try? container.decode(String.self, forKey: .libraryNameCamel)) ?? ""
        delta = (try? container.decode(Double.self, forKey: .delta)) ?? (try? container.decode(Double.self, forKey: .deltaE)) ?? (try? container.decode(Double.self, forKey: .deltaE00))
        l = (try? container.decode(Double.self, forKey: .l)) ?? (try? container.decode(Double.self, forKey: .lowerL))
        a = try? container.decode(Double.self, forKey: .a)
        b = try? container.decode(Double.self, forKey: .b)
        hex = try? container.decode(String.self, forKey: .hex)
    }
}

private struct RGB {
    let r: Int
    let g: Int
    let b: Int

    init?(hex: String) {
        let cleaned = hex.trimmingCharacters(in: CharacterSet(charactersIn: "#"))
        guard cleaned.count == 6, let value = Int(cleaned, radix: 16) else { return nil }
        r = (value >> 16) & 0xff
        g = (value >> 8) & 0xff
        b = value & 0xff
    }
}

private extension Color {
    init?(hex: String) {
        guard let rgb = RGB(hex: hex) else { return nil }
        self.init(
            red: Double(rgb.r) / 255.0,
            green: Double(rgb.g) / 255.0,
            blue: Double(rgb.b) / 255.0
        )
    }
}

private struct HomeModule: Identifiable {
    var id: String { title }
    let title: String
    let icon: String
    let color: Color
    var destination: LibraryDestination?
}

private enum LibraryDestination: Hashable {
    case library
    case productLibrary
    case colorCardLibrary
}

private extension Color {
    static let businessBlue = Color(red: 0.02, green: 0.52, blue: 0.96)
    static let businessGreen = Color(red: 0.18, green: 0.78, blue: 0.34)
    static let businessLightGreen = Color(red: 0.25, green: 0.84, blue: 0.36)
    static let businessOrange = Color(red: 1.00, green: 0.60, blue: 0.03)
    static let businessCyan = Color(red: 0.33, green: 0.77, blue: 0.93)
    static let coralRed = Color(red: 0.96, green: 0.31, blue: 0.35)
}
