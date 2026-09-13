// MediaPipeMPSGraphError.swift
//
// A deliberately simple error type -- no severity or category taxonomy. A
// caller wanting richer classification can inspect `message` or wrap this
// error itself.

import Foundation

public struct MediaPipeMPSGraphError: Error, CustomStringConvertible, LocalizedError
{
    public let message: String

    public init(_ message: String)
    {
        self.message = message
    }

    public var description: String { self.message }
    public var errorDescription: String? { self.message }
}
