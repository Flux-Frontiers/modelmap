import Foundation
import Observation

/// Runs `modelmap serve` from the Python inside the app and follows it.
///
/// The server scans, prints the page's address with its per-run token, and
/// serves until it is stopped. Scanning, the map and every removal happen in
/// that process, through the same code as the command line; the app only
/// starts it, shows its page and stops it.
@MainActor @Observable
final class ServerController {
    enum State: Equatable {
        case stopped
        case starting
        case running(URL)
        case failed(String)
    }

    private(set) var state: State = .stopped
    /// Everything the server printed: its address, each request, each removal.
    private(set) var log = ""
    /// What the server scans; changing it restarts the server.
    private(set) var scope = ScanScope.load()

    private var process: Process?
    private var pending = ""

    /// The interpreter staged into the app by `make python`.
    nonisolated static var python: URL? {
        Bundle.main.resourceURL?.appendingPathComponent("Python/bin/python3")
    }

    func start() {
        guard process == nil else { return }
        state = .starting
        log = ""
        pending = ""
        Task.detached(priority: .userInitiated) {
            let env = LoginEnvironment.load()
            await MainActor.run { self.launch(env) }
        }
    }

    func stop() {
        guard let process else { return }
        self.process = nil
        // modelmap serve stops cleanly on Ctrl-C.
        if process.isRunning { process.interrupt() }
        state = .stopped
    }

    func restart() {
        stop()
        start()
    }

    /// Scan something else: remembered for the next launch, and the server
    /// restarts to scan it.
    func scan(_ scope: ScanScope) {
        self.scope = scope
        scope.save()
        restart()
    }

    private func launch(_ base: [String: String]) {
        guard let python = Self.python, FileManager.default.isExecutableFile(atPath: python.path) else {
            state = .failed("The Python inside the app is missing. Build it with `make python` in macos/.")
            return
        }
        var env = base
        env["PYTHONUNBUFFERED"] = "1"
        let process = Process()
        process.executableURL = python
        // -I: ignore PYTHONPATH and the user's site-packages; only the
        // modelmap inside the app is imported.
        process.arguments = ["-I", "-m", "modelmap", "serve", "--no-open"] + scope.arguments
        process.environment = env
        process.currentDirectoryURL = FileManager.default.homeDirectoryForCurrentUser
        let pipe = Pipe()
        process.standardOutput = pipe
        process.standardError = pipe
        process.standardInput = FileHandle.nullDevice
        pipe.fileHandleForReading.readabilityHandler = { [weak self] handle in
            let data = handle.availableData
            if data.isEmpty {
                handle.readabilityHandler = nil
                return
            }
            let text = String(decoding: data, as: UTF8.self)
            Task { @MainActor in self?.receive(text) }
        }
        process.terminationHandler = { [weak self] finished in
            let status = finished.terminationStatus
            Task { @MainActor in self?.ended(finished, status: status) }
        }
        do {
            try process.run()
            self.process = process
        } catch {
            state = .failed("Could not start the server: \(error.localizedDescription)")
        }
    }

    private func receive(_ text: String) {
        log += text
        pending += text
        while let newline = pending.firstIndex(of: "\n") {
            let line = String(pending[..<newline])
            pending.removeSubrange(...newline)
            if state == .starting, let url = Self.address(in: line) {
                state = .running(url)
            }
        }
    }

    private func ended(_ finished: Process, status: Int32) {
        // A stop or restart already moved on from this process.
        guard finished === process else { return }
        process = nil
        let tail = log.split(separator: "\n").suffix(12).joined(separator: "\n")
        state = .failed("The server stopped (exit status \(status)).\n\n\(tail)")
    }

    /// The page's address from the line `modelmap serve` prints when it is ready.
    nonisolated static func address(in line: String) -> URL? {
        let prefix = "Serving the map at "
        guard line.hasPrefix(prefix) else { return nil }
        let text = line.dropFirst(prefix.count).trimmingCharacters(in: .whitespaces)
        guard let url = URL(string: text), url.scheme == "http", url.host == "127.0.0.1" else {
            return nil
        }
        return url
    }
}
