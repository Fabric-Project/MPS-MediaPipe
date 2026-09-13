//
//  Jacobi3x3.swift
//  MPSMediaPipe
//

import Foundation
import simd

/// Classic cyclic Jacobi eigenvalue algorithm, specialized to 3x3
/// symmetric matrices -- simd/Accelerate don't expose a small dense
/// SVD/eigendecomposition directly. Standard textbook algorithm (Golub &
/// Van Loan, "Matrix Computations", §8.5).
enum Jacobi3x3
{
    /// Flat, stack-allocated 3x3 scalar storage -- avoids a heap-allocated
    /// `[[Float]]` for the working matrices below, which this algorithm
    /// mutates up to `maxIterations * 3` times per call. `subscript(row:col:)`
    /// gives the same `a[row, col]` access pattern a nested array would,
    /// just without the heap/ARC/bounds-checked-array overhead.
    private struct Matrix3
    {
        var m00: Float, m01: Float, m02: Float
        var m10: Float, m11: Float, m12: Float
        var m20: Float, m21: Float, m22: Float

        subscript(row: Int, col: Int) -> Float
        {
            get
            {
                switch (row, col)
                {
                case (0, 0): return m00
                case (0, 1): return m01
                case (0, 2): return m02
                case (1, 0): return m10
                case (1, 1): return m11
                case (1, 2): return m12
                case (2, 0): return m20
                case (2, 1): return m21
                case (2, 2): return m22
                default: fatalError("Matrix3 subscript (\(row), \(col)) out of bounds")
                }
            }
            set
            {
                switch (row, col)
                {
                case (0, 0): m00 = newValue
                case (0, 1): m01 = newValue
                case (0, 2): m02 = newValue
                case (1, 0): m10 = newValue
                case (1, 1): m11 = newValue
                case (1, 2): m12 = newValue
                case (2, 0): m20 = newValue
                case (2, 1): m21 = newValue
                case (2, 2): m22 = newValue
                default: fatalError("Matrix3 subscript (\(row), \(col)) out of bounds")
                }
            }
        }
    }

    /// Returns eigenvalues sorted descending and their corresponding
    /// orthonormal eigenvectors as the columns of a rotation matrix, or nil
    /// if `matrix` isn't (numerically) symmetric. `matrix` need not be
    /// exactly symmetric bit-for-bit -- only the lower triangle is read.
    static func symmetricEigendecomposition(_ matrix: simd_float3x3, maxIterations: Int = 64, tolerance: Float = 1e-12) -> (eigenvalues: [Float], eigenvectors: simd_float3x3)?
    {
        var a = Matrix3(
            m00: matrix.columns.0.x, m01: matrix.columns.1.x, m02: matrix.columns.2.x,
            m10: matrix.columns.0.y, m11: matrix.columns.1.y, m12: matrix.columns.2.y,
            m20: matrix.columns.0.z, m21: matrix.columns.1.z, m22: matrix.columns.2.z
        )
        var v = Matrix3(m00: 1, m01: 0, m02: 0, m10: 0, m11: 1, m12: 0, m20: 0, m21: 0, m22: 1)

        func offDiagonalSumOfSquares() -> Float
        {
            a[0, 1] * a[0, 1] + a[0, 2] * a[0, 2] + a[1, 2] * a[1, 2]
        }

        for _ in 0..<maxIterations
        {
            guard offDiagonalSumOfSquares() > tolerance else { break }

            // Sweep the three off-diagonal pairs each iteration (classic
            // cyclic Jacobi, simpler and just as robust as always picking
            // the single largest entry for a fixed 3x3).
            for (p, q) in [(0, 1), (0, 2), (1, 2)]
            {
                let apq = a[p, q]
                guard abs(apq) > 1e-20 else { continue }

                let theta = (a[q, q] - a[p, p]) / (2 * apq)
                let t = (theta >= 0 ? Float(1) : Float(-1)) / (abs(theta) + sqrt(theta * theta + 1))
                let c = 1 / sqrt(t * t + 1)
                let s = t * c

                for k in 0..<3
                {
                    let akp = a[k, p]
                    let akq = a[k, q]
                    a[k, p] = c * akp - s * akq
                    a[k, q] = s * akp + c * akq
                }
                for k in 0..<3
                {
                    let apk = a[p, k]
                    let aqk = a[q, k]
                    a[p, k] = c * apk - s * aqk
                    a[q, k] = s * apk + c * aqk
                }
                for k in 0..<3
                {
                    let vkp = v[k, p]
                    let vkq = v[k, q]
                    v[k, p] = c * vkp - s * vkq
                    v[k, q] = s * vkp + c * vkq
                }
            }
        }

        guard offDiagonalSumOfSquares().isFinite else { return nil }

        var eigenpairs = (0..<3).map { index in (value: a[index, index], vector: simd_float3(v[0, index], v[1, index], v[2, index])) }
        eigenpairs.sort { $0.value > $1.value }

        let eigenvalues = eigenpairs.map(\.value)
        let eigenvectors = simd_float3x3(columns: (eigenpairs[0].vector, eigenpairs[1].vector, eigenpairs[2].vector))
        return (eigenvalues, eigenvectors)
    }
}
