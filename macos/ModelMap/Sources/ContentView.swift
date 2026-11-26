import SwiftUI

struct ContentView: View {
    let server: ServerController
    let page: PageStore

    var body: some View {
        content.navigationSubtitle(server.scope.summary)
    }

    @ViewBuilder private var content: some View {
        switch server.state {
        case .stopped, .starting:
            VStack(spacing: 14) {
                ProgressView()
                Text("Scanning for models...")
                    .font(.headline)
                Text(server.scope.folders.isEmpty
                     ? "modelmap looks in every store it knows and sweeps your home folder."
                     : "Scanning: \(server.scope.summary)")
                    .foregroundStyle(.secondary)
                    .multilineTextAlignment(.center)
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
        case let .running(url):
            MapView(url: url, store: page)
        case let .failed(message):
            VStack(alignment: .leading, spacing: 12) {
                Label("modelmap could not start", systemImage: "exclamationmark.triangle")
                    .font(.title3.weight(.semibold))
                ScrollView {
                    Text(message)
                        .font(.system(.body, design: .monospaced))
                        .textSelection(.enabled)
                        .frame(maxWidth: .infinity, alignment: .leading)
                }
                .frame(maxHeight: 280)
                Button("Try Again") { server.restart() }
                    .keyboardShortcut(.defaultAction)
            }
            .padding(32)
            .frame(maxWidth: 680, maxHeight: .infinity)
        }
    }
}

/// What the server printed: each request, and each removal with what it ran.
struct LogView: View {
    let server: ServerController

    var body: some View {
        ScrollView {
            Text(server.log.isEmpty ? "Nothing yet." : Self.withoutToken(server.log))
                .font(.system(.callout, design: .monospaced))
                .textSelection(.enabled)
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(12)
        }
    }

    /// The first line holds the run's token; the log shows the address without it.
    nonisolated static func withoutToken(_ log: String) -> String {
        log.replacingOccurrences(of: #"\?t=[A-Za-z0-9_\-]+"#, with: "", options: .regularExpression)
    }
}
