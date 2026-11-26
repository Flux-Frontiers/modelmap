import Foundation
import Testing
@testable import modelmap

struct AddressTests {
    @Test func readsTheLineServePrints() throws {
        let url = try #require(ServerController.address(in: "Serving the map at http://127.0.0.1:51234/?t=abc_DEF-123"))
        #expect(url.port == 51234)
        #expect(url.query == "t=abc_DEF-123")
    }

    @Test func ignoresEveryOtherLine() {
        #expect(ServerController.address(in: "Press Ctrl-C to stop.") == nil)
        #expect(ServerController.address(in: "GET / 200") == nil)
    }

    @Test func acceptsOnlyThisMachine() {
        #expect(ServerController.address(in: "Serving the map at http://example.com:80/?t=x") == nil)
        #expect(ServerController.address(in: "Serving the map at https://127.0.0.1:80/?t=x") == nil)
    }
}

struct EnvironmentTests {
    @Test func parsesNulSeparatedRecords() {
        let data = Data("PATH=/opt/homebrew/bin:/usr/bin\0HF_HOME=/Volumes/Models/hf\0EMPTY=\0EQ=a=b\0".utf8)
        let env = LoginEnvironment.parse(data)
        #expect(env["PATH"] == "/opt/homebrew/bin:/usr/bin")
        #expect(env["HF_HOME"] == "/Volumes/Models/hf")
        #expect(env["EMPTY"] == "")
        #expect(env["EQ"] == "a=b")
    }

    @Test func aMultilineValueStaysOneValue() {
        let env = LoginEnvironment.parse(Data("A=line one\nline two\0B=2\0".utf8))
        #expect(env["A"] == "line one\nline two")
        #expect(env["B"] == "2")
    }

    @Test func theLoginShellIsKnown() {
        #expect(LoginEnvironment.loginShell() != nil)
    }
}

struct LogTests {
    @Test func theTokenIsNotShown() {
        let log = "Serving the map at http://127.0.0.1:5000/?t=SECRET_token-1\nGET / 200\n"
        let shown = LogView.withoutToken(log)
        #expect(!shown.contains("SECRET"))
        #expect(shown.contains("http://127.0.0.1:5000/\n"))
    }
}

struct NavigationTests {
    @Test func onlyTheServerStaysInTheWindow() throws {
        let server = try #require(URL(string: "http://127.0.0.1:5000/?t=x"))
        let page = try #require(URL(string: "http://127.0.0.1:5000/#rec-1"))
        let other = try #require(URL(string: "http://127.0.0.1:6000/"))
        let home = try #require(URL(string: "https://github.com/Flux-Frontiers/modelmap"))
        #expect(MapView.Coordinator.isServer(page, server))
        #expect(!MapView.Coordinator.isServer(other, server))
        #expect(!MapView.Coordinator.isServer(home, server))
    }
}

struct ScanScopeTests {
    @Test func theUsualScopeAddsNoArguments() {
        #expect(ScanScope.usual.arguments.isEmpty)
        #expect(ScanScope.usual.summary == "Usual locations")
    }

    @Test func foldersAreAddedToTheStores() {
        let scope = ScanScope(folders: ["/Volumes/Models", "/tmp/x"], includeStores: true)
        #expect(scope.arguments == ["/Volumes/Models", "/tmp/x"])
        #expect(scope.summary.hasPrefix("Model stores and /Volumes/Models"))
    }

    @Test func foldersAloneSkipTheStores() {
        let scope = ScanScope(folders: ["/Volumes/Models"], includeStores: false)
        #expect(scope.arguments == ["--no-defaults", "/Volumes/Models"])
        #expect(scope.summary == "/Volumes/Models")
    }

    @Test func aRememberedScopeSurvivesARelaunch() throws {
        let defaults = try #require(UserDefaults(suiteName: "modelmap-tests-\(UUID().uuidString)"))
        let folder = FileManager.default.temporaryDirectory.path
        ScanScope(folders: [folder], includeStores: false).save(to: defaults)
        #expect(ScanScope.load(from: defaults) == ScanScope(folders: [folder], includeStores: false))
    }

    @Test func aFolderThatIsGoneIsDropped() throws {
        let defaults = try #require(UserDefaults(suiteName: "modelmap-tests-\(UUID().uuidString)"))
        ScanScope(folders: ["/no/such/folder"], includeStores: false).save(to: defaults)
        // Nothing left to scan, so the usual places again, never --no-defaults alone.
        #expect(ScanScope.load(from: defaults) == .usual)
    }
}
