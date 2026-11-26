import AppKit

/// The Open panel that chooses folders to scan.
@MainActor
enum FolderPicker {
    /// Asks for one or more folders; nil when cancelled.
    ///
    /// - Parameter current: The scope now in use, for the checkbox's state.
    static func choose(current: ScanScope) -> ScanScope? {
        let panel = NSOpenPanel()
        panel.title = "Choose Folders to Scan"
        panel.message = "modelmap scans these folders, and any drive among them, for model files."
        panel.prompt = "Scan"
        panel.canChooseDirectories = true
        panel.canChooseFiles = false
        panel.allowsMultipleSelection = true
        panel.canCreateDirectories = false

        let stores = NSButton(
            checkboxWithTitle: "Also read the usual model stores (Hugging Face, Ollama, oMLX ...)",
            target: nil,
            action: nil
        )
        stores.state = current.includeStores ? .on : .off
        let accessory = NSView(frame: NSRect(x: 0, y: 0, width: 460, height: 36))
        stores.frame = NSRect(x: 16, y: 8, width: 440, height: 20)
        accessory.addSubview(stores)
        panel.accessoryView = accessory
        panel.isAccessoryViewDisclosed = true

        guard panel.runModal() == .OK, !panel.urls.isEmpty else { return nil }
        return ScanScope(folders: panel.urls.map(\.path), includeStores: stores.state == .on)
    }
}
