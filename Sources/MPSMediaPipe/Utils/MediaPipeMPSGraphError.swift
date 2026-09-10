// MediaPipeMPSGraphError.swift
//
// This package's own error type -- deliberately simpler than Fabric's own
// FabricError (no severity/category taxonomy, which is Fabric-app-specific
// error-reporting infrastructure out of scope for a standalone library). A
// consuming app that wants richer classification can inspect `message` or
// wrap this error itself.

import Foundation

public struct MediaPipeMPSGraphError: Error, CustomStringConvertible
{
    public let message: String

    public init(_ message: String)
    {
        self.message = message
    }

    public var description: String { self.message }
}
