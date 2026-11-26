import AppKit
import SwiftUI

@main
struct ModelMapApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) private var delegate
    @State private var page = PageStore()

    var body: some Scene {
        Window("modelmap", id: "map") {
            ContentView(server: delegate.server, page: page)
                .frame(minWidth: 900, minHeight: 640)
        }
        .defaultSize(width: 1280, height: 860)
        .commands {
            CommandGroup(replacing: .newItem) {}
            CommandMenu("Map") {
                Button("Rescan") { page.rescan() }
                    .keyboardShortcut("r")
                    .disabled(!delegate.server.isRunning)
                Button("Reload Page") { page.reload() }
                    .keyboardShortcut("r", modifiers: [.command, .shift])
                    .disabled(!delegate.server.isRunning)
                Divider()
                Button("Open in Browser") {
                    if case let .running(url) = delegate.server.state { NSWorkspace.shared.open(url) }
                }
                .keyboardShortcut("b")
                .disabled(!delegate.server.isRunning)
                Button("Restart Server") { delegate.server.restart() }
            }
            CommandGroup(after: .windowArrangement) {
                OpenWindowButton(title: "Server Log", id: "log", key: "l")
            }
            CommandGroup(replacing: .help) {
                OpenWindowButton(title: "modelmap Help", id: "help", key: "?")
                Divider()
                Link("Mac App Documentation", destination: HelpView.onlineHelp)
                Link("Guide", destination: HelpView.guide)
                Link("Report an Issue", destination: HelpView.issues)
            }
        }

        Window("Server Log", id: "log") {
            LogView(server: delegate.server)
                .frame(minWidth: 560, minHeight: 320)
        }
        .defaultSize(width: 760, height: 420)

        Window("modelmap Help", id: "help") {
            HelpView()
                .frame(minWidth: 560, minHeight: 480)
        }
        .defaultSize(width: 720, height: 760)
    }
}

private struct OpenWindowButton: View {
    let title: String
    let id: String
    let key: KeyEquivalent
    @Environment(\.openWindow) private var openWindow

    var body: some View {
        Button(title) { openWindow(id: id) }
            .keyboardShortcut(key)
    }
}

@MainActor
final class AppDelegate: NSObject, NSApplicationDelegate {
    let server = ServerController()

    func applicationDidFinishLaunching(_ notification: Notification) {
        server.start()
    }

    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool {
        true
    }

    func applicationWillTerminate(_ notification: Notification) {
        server.stop()
    }
}

extension ServerController {
    var isRunning: Bool {
        if case .running = state { return true }
        return false
    }
}
