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

        // What to scan with the chosen folders, from least to most.
        let extent = NSPopUpButton(frame: .zero, pullsDown: false)
        extent.addItems(withTitles: [
            "Only these folders",
            "These folders and the model stores (Hugging Face, Ollama, oMLX ...)",
            "These folders, the model stores and the home folder",
        ])
        // Choosing folders means scanning those folders: start on the
        // narrowest choice, or on the one made last time.
        extent.selectItem(at: current.folders.isEmpty || !current.includeStores ? 0 : current.sweepHome ? 2 : 1)
        let label = NSTextField(labelWithString: "Scan:")
        let row = NSStackView(views: [label, extent])
        row.orientation = .horizontal
        row.spacing = 8
        row.edgeInsets = NSEdgeInsets(top: 8, left: 16, bottom: 8, right: 16)
        row.frame = NSRect(x: 0, y: 0, width: 560, height: 40)
        panel.accessoryView = row
        panel.isAccessoryViewDisclosed = true

        guard panel.runModal() == .OK, !panel.urls.isEmpty else { return nil }
        let choice = extent.indexOfSelectedItem
        return ScanScope(folders: panel.urls.map(\.path), includeStores: choice >= 1, sweepHome: choice == 2)
    }
}
