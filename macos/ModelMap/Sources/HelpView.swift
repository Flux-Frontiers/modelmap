import SwiftUI

/// Help inside the app. The full page, with building the app, is
/// docs/mac.md on the documentation site; keep the two in step.
struct HelpView: View {
    static let onlineHelp = URL(string: "https://flux-frontiers.github.io/modelmap/mac/")!
    static let guide = URL(string: "https://flux-frontiers.github.io/modelmap/guide/")!
    static let issues = URL(string: "https://github.com/Flux-Frontiers/modelmap/issues")!

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 22) {
                header
                HelpSection(title: "The map") {
                    Text("Each block is a store, such as Hugging Face, Ollama, oMLX or loose files, and each tile a model, sized by the space it takes.")
                    Bullet("Click a tile to open the model. The square is redrawn as its files, and the panel lists each file with its full path.")
                    Bullet("Press Escape, or use the trail above the map, to go back up.")
                    Bullet("Group by store, format, family or kind, and filter by name, with the controls above the map.")
                    Bullet("The panel lists every entry with a problem, such as an incomplete download, even when its tile is too small to see.")
                }
                HelpSection(title: "Choosing what to scan") {
                    Text("By default the app reads every model store it knows and sweeps your home folder. To scan an external drive or another folder, choose **Map > Choose Folders to Scan** (Cmd-O).")
                    Text("The **Scan** menu at the bottom of the panel says what to scan with them:")
                    Bullet("**These folders, the model stores and the home folder**: the default scan plus your folders.")
                    Bullet("**These folders and the model stores**: without the sweep of your home folder.")
                    Bullet("**Only these folders**: nothing else.")
                    Text("The window's title shows what is scanned, and the choice is remembered. **Map > Scan Usual Locations** goes back to the default.")
                        .foregroundStyle(.secondary)
                }
                HelpSection(title: "Removing a model") {
                    Text("Open the model and press **Remove** in its panel. The dialog lists exactly what will happen:")
                    Bullet("An Ollama model: the `ollama rm` command, with every tag that shares its weights.")
                    Bullet("A Hugging Face cache repository: the `hf cache rm` command.")
                    Bullet("Anything else: every file or folder that will move to the Trash.")
                    Text("Files moved to the Trash can be put back from there. What `ollama rm` and `hf cache rm` remove cannot. Nothing happens until you press **Remove** in the dialog; Return cancels.")
                    Text("Before acting, the app scans again. If anything changed since the dialog opened, it removes nothing and shows the new list.")
                    Text("A model has no Remove button when removing it would do harm: a cache its tool rebuilds, files nothing refers to, or files another model also uses. The panel says why.")
                        .foregroundStyle(.secondary)
                }
                HelpSection(title: "Menus") {
                    Shortcut("Choose Folders to Scan", "Cmd-O", "Scan folders or drives you pick.")
                    Shortcut("Scan Usual Locations", "", "Back to the stores and the home folder.")
                    Shortcut("Rescan", "Cmd-R", "Scan again and redraw the map.")
                    Shortcut("Reload Page", "Shift-Cmd-R", "Redraw the page without scanning.")
                    Shortcut("Open in Browser", "Cmd-B", "Open the same page in your web browser.")
                    Shortcut("Restart Server", "", "Stop and start the server, then scan.")
                    Shortcut("Server Log", "Cmd-L", "Every request and every removal, with the command it ran.")
                }
                HelpSection(title: "Folder access") {
                    Text("The first scan sweeps your home folder, and macOS asks before an app reads Desktop, Documents or Downloads. Choose **Allow** for modelmap to find model files there; **Don't Allow** skips that folder.")
                    Text("Change the answer in System Settings > Privacy & Security > Files and Folders.")
                        .foregroundStyle(.secondary)
                }
                HelpSection(title: "Privacy") {
                    Text("Nothing leaves your Mac. The server listens only on 127.0.0.1, needs a token made for each run, and answers only its own page. The Server Log hides the token.")
                }
                HelpSection(title: "If something is wrong") {
                    Problem("A store is missing",
                            "If a setting such as HF_HOME or OLLAMA_MODELS moves the store, set it in your shell profile, then choose Map > Restart Server. Run `modelmap providers` in Terminal to see where each store is looked for.")
                    Problem("Remove says a command is not found",
                            "Ollama models need `ollama`, and Hugging Face cache repositories need `hf`. Install the one named, then choose Map > Restart Server.")
                    Problem("Remove says Ollama is not running",
                            "`ollama rm` acts on the running Ollama. Start Ollama, then try again.")
                    Problem("modelmap could not start",
                            "Press Try Again, and check Window > Server Log for the cause.")
                }
                HStack(spacing: 18) {
                    Link("Mac app documentation", destination: Self.onlineHelp)
                    Link("Guide", destination: Self.guide)
                    Link("Report an issue", destination: Self.issues)
                }
            }
            .padding(28)
            .frame(maxWidth: 680, alignment: .leading)
            .textSelection(.enabled)
        }
    }

    private var header: some View {
        HStack(spacing: 14) {
            Image(nsImage: NSApp.applicationIconImage)
                .resizable()
                .frame(width: 56, height: 56)
            VStack(alignment: .leading, spacing: 3) {
                Text("modelmap Help").font(.title2.weight(.semibold))
                Text("Find, map and remove the language models on this Mac.")
                    .foregroundStyle(.secondary)
            }
        }
    }
}

private struct HelpSection<Content: View>: View {
    let title: String
    @ViewBuilder let content: Content

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(title).font(.headline)
            content
        }
        .fixedSize(horizontal: false, vertical: true)
    }
}

private struct Bullet: View {
    let text: LocalizedStringKey

    init(_ text: LocalizedStringKey) {
        self.text = text
    }

    var body: some View {
        HStack(alignment: .firstTextBaseline, spacing: 8) {
            Text("-")
            Text(text)
        }
    }
}

private struct Shortcut: View {
    let item: String
    let keys: String
    let does: String

    init(_ item: String, _ keys: String, _ does: String) {
        self.item = item
        self.keys = keys
        self.does = does
    }

    var body: some View {
        HStack(alignment: .firstTextBaseline, spacing: 10) {
            Text(item).fontWeight(.medium).frame(width: 170, alignment: .leading)
            Text(keys).font(.system(.callout, design: .monospaced)).frame(width: 96, alignment: .leading)
            Text(does).foregroundStyle(.secondary)
        }
    }
}

private struct Problem: View {
    let title: String
    let text: LocalizedStringKey

    init(_ title: String, _ text: LocalizedStringKey) {
        self.title = title
        self.text = text
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(title).fontWeight(.medium)
            Text(text).foregroundStyle(.secondary)
        }
    }
}
