import Darwin
import Foundation

/// The environment of the user's login shell.
///
/// An app opened from the Finder inherits launchd's environment, not the
/// shell's: its PATH lacks Homebrew and `~/.local/bin`, so `ollama` and `hf`
/// are not found, and settings such as `HF_HOME` or `OLLAMA_MODELS` that move
/// a store are missing, so the scan would look in the wrong place. The server
/// is given the login shell's environment instead, as a terminal would.
enum LoginEnvironment {
    /// The login shell's variables over this process's own.
    ///
    /// - Parameter timeout: Seconds to wait for the shell before giving up
    ///   and using this process's environment alone.
    static func load(timeout: TimeInterval = 5) -> [String: String] {
        var env = ProcessInfo.processInfo.environment
        if let shell = loginShell(), let output = run(shell, ["-l", "-c", "env -0"], timeout: timeout) {
            env.merge(parse(output)) { _, login in login }
        }
        return env
    }

    /// The user's shell from the account database; `SHELL` is not always set
    /// for an app.
    static func loginShell() -> String? {
        guard let entry = getpwuid(getuid()), let shell = entry.pointee.pw_shell else { return nil }
        let path = String(cString: shell)
        return path.isEmpty ? nil : path
    }

    /// `KEY=VALUE` records separated by NUL bytes, as `env -0` prints them.
    static func parse(_ data: Data) -> [String: String] {
        var env: [String: String] = [:]
        for record in data.split(separator: 0) {
            guard let text = String(data: Data(record), encoding: .utf8),
                  let equals = text.firstIndex(of: "=")
            else { continue }
            let key = String(text[..<equals])
            if !key.isEmpty { env[key] = String(text[text.index(after: equals)...]) }
        }
        return env
    }

    private static func run(_ path: String, _ arguments: [String], timeout: TimeInterval) -> Data? {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: path)
        process.arguments = arguments
        let out = Pipe()
        process.standardOutput = out
        process.standardError = FileHandle.nullDevice
        process.standardInput = FileHandle.nullDevice
        do { try process.run() } catch { return nil }
        // Read before waiting, so a full pipe cannot stall the shell.
        let reader = DispatchQueue.global(qos: .userInitiated)
        let box = DataBox()
        let done = DispatchSemaphore(value: 0)
        reader.async {
            box.data = out.fileHandleForReading.readDataToEndOfFile()
            done.signal()
        }
        if done.wait(timeout: .now() + timeout) == .timedOut {
            process.terminate()
            return nil
        }
        process.waitUntilExit()
        return process.terminationStatus == 0 ? box.data : nil
    }
}

private final class DataBox: @unchecked Sendable {
    var data = Data()
}
