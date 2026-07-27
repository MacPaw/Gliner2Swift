// Copyright 2026 MacPaw Way Ltd.
//
//    Licensed under the Apache License, Version 2.0 (the "License");
//    you may not use this file except in compliance with the License.
//    You may obtain a copy of the License at
//
//        http://www.apache.org/licenses/LICENSE-2.0
//
//    Unless required by applicable law or agreed to in writing, software
//    distributed under the License is distributed on an "AS IS" BASIS,
//    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//    See the License for the specific language governing permissions and
//    limitations under the License.
//
// DeviceInfo.swift
// Device identity and process-level memory footprint. Cross-platform (iOS + macOS) so
// BenchmarkEngine stays testable on the Mac.

import Foundation

#if canImport(UIKit)
import UIKit
#endif

enum DeviceInfo {

    /// Hardware identifier, e.g. "iPhone14,3" on iOS, host name on macOS.
    static var identifier: String {
        #if canImport(UIKit)
        // On a real device `uname` gives the model id; the Simulator reports the Mac's
        // arch, so read the simulated model from the environment there.
        if let simulated = ProcessInfo.processInfo.environment["SIMULATOR_MODEL_IDENTIFIER"] {
            return simulated
        }
        var systemInfo = utsname()
        uname(&systemInfo)
        let identifier = withUnsafeBytes(of: &systemInfo.machine) { raw -> String in
            let bytes = raw.prefix { $0 != 0 }
            return String(decoding: bytes, as: UTF8.self)
        }
        return identifier.isEmpty ? UIDevice.current.model : identifier
        #else
        return ProcessInfo.processInfo.hostName
        #endif
    }

    /// Friendly device name, e.g. "iPhone 13 Pro Max", falling back to the raw identifier.
    static var model: String {
        marketingNames[identifier] ?? identifier
    }

    /// "iPhone 13 Pro Max (iPhone14,3)" — friendly name plus the raw id for the record.
    static var modelLabel: String {
        let id = identifier
        guard let name = marketingNames[id] else { return id }
        return "\(name) (\(id))"
    }

    private static let marketingNames: [String: String] = [
        "iPhone12,1": "iPhone 11", "iPhone12,3": "iPhone 11 Pro", "iPhone12,5": "iPhone 11 Pro Max",
        "iPhone12,8": "iPhone SE (2nd gen)",
        "iPhone13,1": "iPhone 12 mini", "iPhone13,2": "iPhone 12",
        "iPhone13,3": "iPhone 12 Pro", "iPhone13,4": "iPhone 12 Pro Max",
        "iPhone14,4": "iPhone 13 mini", "iPhone14,5": "iPhone 13",
        "iPhone14,2": "iPhone 13 Pro", "iPhone14,3": "iPhone 13 Pro Max",
        "iPhone14,6": "iPhone SE (3rd gen)",
        "iPhone14,7": "iPhone 14", "iPhone14,8": "iPhone 14 Plus",
        "iPhone15,2": "iPhone 14 Pro", "iPhone15,3": "iPhone 14 Pro Max",
        "iPhone15,4": "iPhone 15", "iPhone15,5": "iPhone 15 Plus",
        "iPhone16,1": "iPhone 15 Pro", "iPhone16,2": "iPhone 15 Pro Max",
        "iPhone17,3": "iPhone 16", "iPhone17,4": "iPhone 16 Plus",
        "iPhone17,1": "iPhone 16 Pro", "iPhone17,2": "iPhone 16 Pro Max",
        "iPhone17,5": "iPhone 16e",
    ]

    /// Chip family, inferred from the identifier where known (a nice extra line).
    static var chip: String? {
        switch identifier {
        case "iPhone12,1", "iPhone12,3", "iPhone12,5", "iPhone12,8": return "A13 Bionic"
        case "iPhone13,1", "iPhone13,2", "iPhone13,3", "iPhone13,4": return "A14 Bionic"
        case "iPhone14,4", "iPhone14,5", "iPhone14,2", "iPhone14,3", "iPhone14,6": return "A15 Bionic"
        case "iPhone14,7", "iPhone14,8": return "A15 Bionic"
        case "iPhone15,2", "iPhone15,3": return "A16 Bionic"
        case "iPhone15,4", "iPhone15,5": return "A16 Bionic"
        case "iPhone16,1", "iPhone16,2": return "A17 Pro"
        case "iPhone17,3", "iPhone17,4": return "A18"
        case "iPhone17,1", "iPhone17,2": return "A18 Pro"
        default: return nil
        }
    }

    static var osVersion: String {
        let v = ProcessInfo.processInfo.operatingSystemVersion
        return "\(v.majorVersion).\(v.minorVersion).\(v.patchVersion)"
    }

    /// Resident physical footprint of this process, MB. Uses the "phys_footprint"
    /// field that Instruments and the jetsam limit both track on iOS.
    static func residentMemoryMB() -> Double {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size) / 4
        let result = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
            }
        }
        guard result == KERN_SUCCESS else { return 0 }
        return Double(info.phys_footprint) / 1_048_576
    }
}
