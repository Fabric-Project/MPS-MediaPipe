import MetalPerformanceShadersGraph

/// The precision a MediaPipe graph computes in. Inputs and outputs are
/// float32 in every mode.
public enum MediaPipePrecision: Sendable, Equatable
{
    /// Float32 throughout.
    case float32
    /// Convolutions and fully connected layers in float16 (weights and
    /// arithmetic), everything else float32.
    case mixedFloat16
    /// Float16 throughout, with one cast after the input and one before each
    /// output.
    case float16

    /// The type activations flow through between layers.
    var activationDataType: MPSDataType { self == .float16 ? .float16 : .float32 }

    /// The type convolutions and fully connected layers run in.
    var layerDataType: MPSDataType { self == .float32 ? .float32 : .float16 }
}
