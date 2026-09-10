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
    /// Returns eigenvalues sorted descending and their corresponding
    /// orthonormal eigenvectors as the columns of a rotation matrix, or nil
    /// if `matrix` isn't (numerically) symmetric. `matrix` need not be
    /// exactly symmetric bit-for-bit -- only the lower triangle is read.
    static func symmetricEigendecomposition(_ matrix: simd_float3x3, maxIterations: Int = 64, tolerance: Float = 1e-12) -> (eigenvalues: [Float], eigenvectors: simd_float3x3)?
    {
        var a: [[Float]] = [
            [matrix.columns.0.x, matrix.columns.1.x, matrix.columns.2.x],
            [matrix.columns.0.y, matrix.columns.1.y, matrix.columns.2.y],
            [matrix.columns.0.z, matrix.columns.1.z, matrix.columns.2.z],
        ]
        var v: [[Float]] = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

        func offDiagonalSumOfSquares() -> Float
        {
            a[0][1] * a[0][1] + a[0][2] * a[0][2] + a[1][2] * a[1][2]
        }

        for _ in 0..<maxIterations
        {
            guard offDiagonalSumOfSquares() > tolerance else { break }

            // Sweep the three off-diagonal pairs each iteration (classic
            // cyclic Jacobi, simpler and just as robust as always picking
            // the single largest entry for a fixed 3x3).
            for (p, q) in [(0, 1), (0, 2), (1, 2)]
            {
                let apq = a[p][q]
                guard abs(apq) > 1e-20 else { continue }

                let theta = (a[q][q] - a[p][p]) / (2 * apq)
                let t = (theta >= 0 ? Float(1) : Float(-1)) / (abs(theta) + sqrt(theta * theta + 1))
                let c = 1 / sqrt(t * t + 1)
                let s = t * c

                for k in 0..<3
                {
                    let akp = a[k][p]
                    let akq = a[k][q]
                    a[k][p] = c * akp - s * akq
                    a[k][q] = s * akp + c * akq
                }
                for k in 0..<3
                {
                    let apk = a[p][k]
                    let aqk = a[q][k]
                    a[p][k] = c * apk - s * aqk
                    a[q][k] = s * apk + c * aqk
                }
                for k in 0..<3
                {
                    let vkp = v[k][p]
                    let vkq = v[k][q]
                    v[k][p] = c * vkp - s * vkq
                    v[k][q] = s * vkp + c * vkq
                }
            }
        }

        guard offDiagonalSumOfSquares().isFinite else { return nil }

        var eigenpairs = (0..<3).map { index in (value: a[index][index], vector: simd_float3(v[0][index], v[1][index], v[2][index])) }
        eigenpairs.sort { $0.value > $1.value }

        let eigenvalues = eigenpairs.map(\.value)
        let eigenvectors = simd_float3x3(columns: (eigenpairs[0].vector, eigenpairs[1].vector, eigenpairs[2].vector))
        return (eigenvalues, eigenvectors)
    }
}
