import Foundation

/// What the server scans: the usual places, or folders the user chose.
///
/// It maps onto `modelmap serve` the way the command line does. With no
/// folders, the known stores are read and the home folder is swept. With
/// folders, each is scanned; `includeStores` keeps the known stores as
/// well, and without it only the folders are scanned (`--no-defaults`).
/// The home sweep is off once folders are named, unless `sweepHome` keeps
/// it (`--sweep`), which needs the stores.
struct ScanScope: Equatable, Sendable {
    var folders: [String] = []
    var includeStores = true
    var sweepHome = false

    static let usual = ScanScope()

    /// The `modelmap serve` arguments after the subcommand.
    var arguments: [String] {
        if folders.isEmpty { return [] }
        if !includeStores { return ["--no-defaults"] + folders }
        return (sweepHome ? ["--sweep"] : []) + folders
    }

    /// A short description for the window's subtitle.
    var summary: String {
        if folders.isEmpty { return "Usual locations" }
        let names = folders.map { ($0 as NSString).abbreviatingWithTildeInPath }.joined(separator: ", ")
        if !includeStores { return names }
        return sweepHome ? "Model stores, home folder and \(names)" : "Model stores and \(names)"
    }

    // MARK: Remembered between launches

    private static let foldersKey = "scanFolders"
    private static let storesKey = "scanIncludeStores"
    private static let sweepKey = "scanSweepHome"

    static func load(from defaults: UserDefaults = .standard) -> ScanScope {
        let folders = (defaults.stringArray(forKey: foldersKey) ?? [])
            .filter { FileManager.default.fileExists(atPath: $0) }
        if folders.isEmpty { return .usual }
        let stores = defaults.object(forKey: storesKey) as? Bool ?? true
        return ScanScope(folders: folders, includeStores: stores, sweepHome: stores && defaults.bool(forKey: sweepKey))
    }

    func save(to defaults: UserDefaults = .standard) {
        defaults.set(folders, forKey: Self.foldersKey)
        defaults.set(includeStores, forKey: Self.storesKey)
        defaults.set(sweepHome, forKey: Self.sweepKey)
    }
}
