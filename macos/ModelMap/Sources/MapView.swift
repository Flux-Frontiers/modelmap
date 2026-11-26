import AppKit
import SwiftUI
import WebKit

/// Holds the page's web view so menu commands can reach it.
@MainActor @Observable
final class PageStore {
    weak var webView: WKWebView?

    /// Presses the page's own Rescan button, so the page shows its progress.
    func rescan() {
        webView?.evaluateJavaScript("document.getElementById('rescan').click()")
    }

    func reload() {
        webView?.reload()
    }
}

/// The map page, served by `modelmap serve` on 127.0.0.1.
struct MapView: NSViewRepresentable {
    let url: URL
    let store: PageStore

    func makeCoordinator() -> Coordinator {
        Coordinator(server: url)
    }

    func makeNSView(context: Context) -> WKWebView {
        let configuration = WKWebViewConfiguration()
        // Nothing the page does needs storage that outlives the run.
        configuration.websiteDataStore = .nonPersistent()
        let view = WKWebView(frame: .zero, configuration: configuration)
        view.navigationDelegate = context.coordinator
        view.allowsBackForwardNavigationGestures = true
        view.load(URLRequest(url: url))
        store.webView = view
        return view
    }

    func updateNSView(_ view: WKWebView, context: Context) {
        store.webView = view
        if context.coordinator.server != url {
            // A restarted server has a new port and token.
            context.coordinator.server = url
            view.load(URLRequest(url: url))
        }
    }

    final class Coordinator: NSObject, WKNavigationDelegate {
        var server: URL

        init(server: URL) {
            self.server = server
        }

        /// The page stays on the server. Any other link, such as the banner's
        /// link to Flux-Frontiers, opens in the default browser.
        func webView(
            _ webView: WKWebView,
            decidePolicyFor action: WKNavigationAction,
            decisionHandler: @escaping @MainActor (WKNavigationActionPolicy) -> Void
        ) {
            guard let target = action.request.url else { return decisionHandler(.cancel) }
            if Self.isServer(target, server) {
                decisionHandler(.allow)
            } else {
                if target.scheme == "https" || target.scheme == "http" {
                    NSWorkspace.shared.open(target)
                }
                decisionHandler(.cancel)
            }
        }

        nonisolated static func isServer(_ target: URL, _ server: URL) -> Bool {
            target.scheme == server.scheme && target.host == server.host && target.port == server.port
        }
    }
}
