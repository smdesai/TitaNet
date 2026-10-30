// ModelStore — where each runtime's model lives. Nothing is bundled: on first use a runtime's files come down from
// Hugging Face (ModelDownloader) into Application Support/Models/<runtime>/, and a Core ML variant's .mlpackage files
// are then compiled for this device once (MLModel.compileModel) and kept as .mlmodelc; the packages are deleted after.
//
//   coreai/        titanet.aimodel, export.json                         smdesai/TitaNet-Large-CoreAI
//   coreml-fp32/   TitaNetScan_fp32.mlmodelc, TitaNetHead_fp32.mlmodelc,   smdesai/TitaNet-Large-CoreML
//                  export.json (from export_fp32.json)
//   coreml-fp16/   the same for the fp16c variant                       smdesai/TitaNet-Large-CoreML
//
// A runtime is ready when every item is present, a package counting as present once its compiled model is: the
// downloader places items atomically, so presence means complete.

import CoreML
import Foundation

@MainActor
enum ModelStore {
    static let coreAIRepo = "smdesai/TitaNet-Large-CoreAI"
    static let coreMLRepo = "smdesai/TitaNet-Large-CoreML"

    static var root: URL {
        FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0].appendingPathComponent("Models")
    }

    static func folder(_ r: Runtime) -> URL { root.appendingPathComponent(r.rawValue) }

    static func repo(_ r: Runtime) -> String { r == .coreAI ? coreAIRepo : coreMLRepo }

    /// What a runtime downloads, as the Hub repo listed it (Core ML: both models of the variant plus its export json).
    static func size(_ r: Runtime) -> String {
        switch r {
        case .coreAI: return "47 MB"
        case .coreMLFP32: return "89 MB"
        case .coreMLFP16: return "47 MB"
        }
    }

    /// `remote` paths in the runtime's repo; `local` paths under `root`.
    static func items(_ r: Runtime) -> [ModelDownloader.Item] {
        let dir = r.rawValue
        switch r {
        case .coreAI:
            return [.init(remote: "titanet.aimodel", local: "\(dir)/titanet.aimodel"),
                    .init(remote: "export.json", local: "\(dir)/export.json")]
        case .coreMLFP32, .coreMLFP16:
            let v = r == .coreMLFP32 ? "fp32" : "fp16c"
            return [.init(remote: "TitaNetScan_\(v).mlpackage", local: "\(dir)/TitaNetScan_\(v).mlpackage"),
                    .init(remote: "TitaNetHead_\(v).mlpackage", local: "\(dir)/TitaNetHead_\(v).mlpackage"),
                    .init(remote: "export_\(v).json", local: "\(dir)/export.json")]
        }
    }

    static func isReady(_ r: Runtime) -> Bool { items(r).allSatisfy(present) }

    /// Downloads what is missing, compiles any Core ML package not yet compiled, and returns the runtime's folder.
    /// `compiling` is called before the first compile.
    static func ensure(_ r: Runtime, with downloader: ModelDownloader, compiling: () -> Void) async throws -> URL {
        let fm = FileManager.default
        let missing = items(r).filter { !present($0) }
        if !missing.isEmpty {
            await downloader.fetch(repo: "https://huggingface.co/\(repo(r))", items: missing, into: root)
            if case .failed(let message) = downloader.phase {
                throw StudioError.contract("Downloading \(r.label) failed: \(message)")
            }
        }
        var announced = false
        for item in items(r) where item.local.hasSuffix(".mlpackage") {
            let out = compiled(item.local)
            guard !fm.fileExists(atPath: out.path) else { continue }
            if !announced { compiling(); announced = true }
            let package = root.appendingPathComponent(item.local)
            let temporary = try await MLModel.compileModel(at: package)
            try? fm.removeItem(at: out)
            try fm.moveItem(at: temporary, to: out)
            var values = URLResourceValues()
            values.isExcludedFromBackup = true
            var excluded = out
            try? excluded.setResourceValues(values)
            try? fm.removeItem(at: package)     // the compiled model is what loads; the package was only its source
        }
        guard isReady(r) else { throw StudioError.contract("\(r.label): the download is incomplete") }
        return folder(r)
    }

    private static func present(_ item: ModelDownloader.Item) -> Bool {
        let path = item.local.hasSuffix(".mlpackage") ? compiled(item.local) : root.appendingPathComponent(item.local)
        return FileManager.default.fileExists(atPath: path.path)
    }

    /// TitaNetScan_fp32.mlpackage -> TitaNetScan_fp32.mlmodelc, beside it.
    private static func compiled(_ local: String) -> URL {
        root.appendingPathComponent((local as NSString).deletingPathExtension + ".mlmodelc")
    }
}
