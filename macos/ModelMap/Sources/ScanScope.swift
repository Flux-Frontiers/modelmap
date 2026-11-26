import Foundation

/// What the server scans: the usual places, or folders the user chose.
///
/// It maps onto `modelmap serve` the way the command line does. With no
/// folders, the known stores are read and the home folder is swept. With
/// folders, each is scanned and the home sweep is off; `includeStores`
/// keeps the known stores as well, and without it only the folders are
/// scanned (`--no-defaults`).
struct ScanScope: Equatable, Sendable {
    var folders: [String] = []
    var includeStores = true

    static let usual = ScanScope()

    /// The `modelmap serve` arguments after the subcommand.
    var arguments: [String] {
        if folders.isEmpty { return [] }
        return (includeStores ? [] : ["--no-defaults"]) + folders
    }

    /// A short description for the window's subtitle.
    var summary: String {
        if folders.isEmpty { return "Usual locations" }
        let names = folders.map { ($0 as NSString).abbreviatingWithTildeInPath }.joined(separator: ", ")
        return includeStores ? "Model stores and \(names)" : names
    }

    // MARK: Remembered between launches

    private static let foldersKey = "scanFolders"
    private static let storesKey = "scanIncludeStores"

    static func load(from defaults: UserDefaults = .standard) -> ScanScope {
        let folders = (defaults.stringArray(forKey: foldersKey) ?? [])
            .filter { FileManager.default.fileExists(atPath: $0) }
        let stores = defaults.object(forKey: storesKey) as? Bool ?? true
        return ScanScope(folders: folders, includeStores: folders.isEmpty || stores)
    }

    func save(to defaults: UserDefaults = .standard) {
        defaults.set(folders, forKey: Self.foldersKey)
        defaults.set(includeStores, forKey: Self.storesKey)
    }
}
