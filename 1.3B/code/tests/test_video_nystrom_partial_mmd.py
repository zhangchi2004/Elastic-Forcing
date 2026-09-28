import unittest

import torch

from losses.video_nystrom_mmd import (
    VideoNystromMMD,
    build_video_nystrom_reference,
    rbf_frobenius_kernel,
)


class TestVideoNystromPartialMMD(unittest.TestCase):
    def _pack(self):
        reference = torch.randn(
            12,
            1,
            2,
            generator=torch.Generator().manual_seed(9100),
        )
        return build_video_nystrom_reference(
            reference,
            num_landmarks=5,
            metadata={
                "feature": {
                    "name": "tiny-partial-video",
                    "matrix_shape": [1, 2],
                },
                "reference": {"dataset": "partial-unit-test"},
            },
            bandwidth=1.6,
            kmeans_iterations=4,
            ridge=1.0e-5,
            chunk_size=3,
        )

    def _matched_objective(self, *, chunk_size: int = 37):
        return VideoNystromMMD(
            self._pack(),
            kernel_chunk_size=chunk_size,
            attraction_weight=2.0,
            current_teacher_attraction_enabled=True,
            current_teacher_attraction_weight=1.0,
        )

    def test_128_by_256_formula_denominators_and_suffix_only_gradient(self):
        objective = self._matched_objective()
        generator = torch.Generator().manual_seed(9101)
        all_students = torch.randn(
            256, 1, 2, generator=generator, requires_grad=True)
        gradient_students = all_students[-128:]
        teacher = torch.randn(
            128, 1, 2, generator=generator, requires_grad=True)

        loss, metadata = objective(
            gradient_students=gradient_students,
            student_reference=all_students,
            current_teacher=teacher,
            stage=False,
        )

        kernel_gs = rbf_frobenius_kernel(
            gradient_students,
            all_students.detach(),
            objective.sigma,
        )
        fixed_attraction = (
            rbf_frobenius_kernel(
                gradient_students,
                objective.landmarks,
                objective.sigma,
            )
            @ objective.alpha
        ).mean()
        kernel_gt = rbf_frobenius_kernel(
            gradient_students,
            teacher.detach(),
            objective.sigma,
        )
        kernel_tt = rbf_frobenius_kernel(
            teacher.detach(),
            teacher.detach(),
            objective.sigma,
        )
        expected = (
            kernel_gs.sum() / float(128 * 256)
            - 2.0
            * (
                (2.0 / 3.0) * fixed_attraction
                + (1.0 / 3.0) * kernel_gt.sum() / float(128 * 128)
            )
            + (2.0 / 3.0) * objective.k_rr
            + (1.0 / 3.0) * kernel_tt.sum() / float(128 * 128)
        )

        torch.testing.assert_close(
            loss, expected, rtol=2.0e-5, atol=2.0e-6)
        torch.testing.assert_close(
            metadata["repulsion"],
            (kernel_gs.sum() / float(128 * 256)).detach(),
            rtol=2.0e-5,
            atol=2.0e-6,
        )
        self.assertEqual(int(metadata["global_samples"]), 256)
        self.assertEqual(int(metadata["gradient_student_samples"]), 128)
        self.assertEqual(int(metadata["student_reference_samples"]), 256)
        self.assertEqual(int(metadata["current_teacher_samples"]), 128)
        self.assertEqual(int(metadata["repulsion_denominator"]), 32768)
        self.assertEqual(
            int(metadata["student_repulsion_denominator"]), 32768)
        self.assertEqual(
            int(metadata["current_teacher_attraction_denominator"]), 16384)
        self.assertEqual(
            int(metadata["current_teacher_self_kernel_denominator"]), 16384)
        self.assertEqual(
            int(metadata["fixed_teacher_attraction_denominator"]), 128)
        self.assertEqual(
            int(metadata["partial_student_reference_enabled"]), 1)
        self.assertEqual(int(metadata["student_reference_detached"]), 1)
        self.assertNotIn("full_student_partial_gradient", metadata)
        self.assertAlmostEqual(
            float(metadata[
                "fixed_teacher_attraction_normalized_weight"]),
            2.0 / 3.0,
            places=7,
        )
        self.assertAlmostEqual(
            float(metadata[
                "current_teacher_attraction_normalized_weight"]),
            1.0 / 3.0,
            places=7,
        )

        actual_student_gradient, actual_teacher_gradient = (
            torch.autograd.grad(
                loss,
                (all_students, teacher),
                allow_unused=True,
                retain_graph=True,
            )
        )
        expected_student_gradient, = torch.autograd.grad(
            expected, all_students)
        torch.testing.assert_close(
            actual_student_gradient,
            expected_student_gradient,
            rtol=3.0e-5,
            atol=3.0e-6,
        )
        self.assertIsNone(actual_teacher_gradient)
        torch.testing.assert_close(
            actual_student_gradient[:128],
            torch.zeros_like(actual_student_gradient[:128]),
            rtol=0.0,
            atol=0.0,
        )
        self.assertGreater(
            float(actual_student_gradient[-128:].norm()), 0.0)
        self.assertTrue(torch.isfinite(actual_student_gradient).all())

    def test_full_student_partial_gradient_formula_and_denominators(self):
        objective = self._matched_objective()
        generator = torch.Generator().manual_seed(9105)
        student_reference = torch.randn(
            256, 1, 2, generator=generator, requires_grad=True)
        gradient_student_indices = torch.cat([
            torch.arange(rank * 64 + 32, rank * 64 + 64)
            for rank in range(4)
        ])
        gradient_students = (
            student_reference.detach()
            .index_select(0, gradient_student_indices)
            .clone()
            .requires_grad_(True)
        )
        teacher = torch.randn(
            128, 1, 2, generator=generator, requires_grad=True)

        loss, metadata = objective(
            gradient_students=gradient_students,
            student_reference=student_reference,
            current_teacher=teacher,
            gradient_student_indices=gradient_student_indices,
            stage=False,
        )

        full_students = student_reference.detach().index_copy(
            0, gradient_student_indices, gradient_students)
        kernel_ss = rbf_frobenius_kernel(
            full_students, full_students, objective.sigma)
        fixed_attraction = (
            rbf_frobenius_kernel(
                full_students, objective.landmarks, objective.sigma)
            @ objective.alpha
        ).mean()
        kernel_st = rbf_frobenius_kernel(
            full_students, teacher.detach(), objective.sigma)
        kernel_tt = rbf_frobenius_kernel(
            teacher.detach(), teacher.detach(), objective.sigma)
        expected = (
            kernel_ss.sum() / float(256 * 256)
            - 2.0
            * (
                (2.0 / 3.0) * fixed_attraction
                + (1.0 / 3.0) * kernel_st.sum() / float(256 * 128)
            )
            + (2.0 / 3.0) * objective.k_rr
            + (1.0 / 3.0) * kernel_tt.sum() / float(128 * 128)
        )

        torch.testing.assert_close(
            loss, expected, rtol=2.0e-5, atol=2.0e-6)
        self.assertEqual(int(metadata["global_samples"]), 256)
        self.assertEqual(int(metadata["gradient_student_samples"]), 128)
        self.assertEqual(int(metadata["student_reference_samples"]), 256)
        self.assertEqual(int(metadata["current_teacher_samples"]), 128)
        self.assertEqual(int(metadata["repulsion_denominator"]), 65536)
        self.assertEqual(
            int(metadata["student_repulsion_denominator"]), 65536)
        self.assertEqual(
            int(metadata["fixed_teacher_attraction_denominator"]), 256)
        self.assertEqual(
            int(metadata["current_teacher_attraction_denominator"]),
            32768,
        )
        self.assertEqual(
            int(metadata["current_teacher_self_kernel_denominator"]),
            16384,
        )
        self.assertEqual(
            int(metadata["full_student_partial_gradient"]), 1)
        self.assertEqual(
            int(metadata["student_repulsion_gradient_sides"]), 2)

        actual_gradient, reference_gradient, teacher_gradient = (
            torch.autograd.grad(
                loss,
                (gradient_students, student_reference, teacher),
                allow_unused=True,
                retain_graph=True,
            )
        )
        expected_gradient, = torch.autograd.grad(
            expected, gradient_students, retain_graph=True)
        torch.testing.assert_close(
            actual_gradient,
            expected_gradient,
            rtol=3.0e-5,
            atol=3.0e-6,
        )
        self.assertIsNone(reference_gradient)
        self.assertIsNone(teacher_gradient)
        self.assertGreater(float(actual_gradient.norm()), 0.0)
        self.assertTrue(torch.isfinite(actual_gradient).all())

        full_repulsion_gradient, = torch.autograd.grad(
            kernel_ss.sum() / float(256 * 256),
            gradient_students,
            retain_graph=True,
        )
        rectangular_repulsion = rbf_frobenius_kernel(
            gradient_students, student_reference.detach(), objective.sigma
        ).sum() / float(128 * 256)
        rectangular_repulsion_gradient, = torch.autograd.grad(
            rectangular_repulsion, gradient_students)
        torch.testing.assert_close(
            full_repulsion_gradient,
            rectangular_repulsion_gradient,
            rtol=3.0e-5,
            atol=3.0e-6,
        )

    def test_inputs_are_detached_and_converted_to_fp32_for_kernels(self):
        objective = self._matched_objective(chunk_size=2)
        generator = torch.Generator().manual_seed(9102)
        gradient_students = torch.randn(
            3, 1, 2, generator=generator, requires_grad=True)
        student_reference = torch.randn(
            5,
            1,
            2,
            generator=generator,
            dtype=torch.float64,
            requires_grad=True,
        )
        teacher = torch.randn(
            3,
            1,
            2,
            generator=generator,
            dtype=torch.float64,
            requires_grad=True,
        )

        loss, _ = objective(
            gradient_students=gradient_students,
            student_reference=student_reference,
            current_teacher=teacher,
            stage=False,
        )
        student_gradient, reference_gradient, teacher_gradient = (
            torch.autograd.grad(
                loss,
                (gradient_students, student_reference, teacher),
                allow_unused=True,
            )
        )
        self.assertEqual(loss.dtype, torch.float32)
        self.assertIsNotNone(student_gradient)
        self.assertIsNone(reference_gradient)
        self.assertIsNone(teacher_gradient)

    def test_legacy_interface_preserves_full_symmetric_formula_and_schema(self):
        objective = self._matched_objective(chunk_size=2)
        generator = torch.Generator().manual_seed(9103)
        generated = torch.randn(
            4, 1, 2, generator=generator, requires_grad=True)
        teacher = torch.randn(4, 1, 2, generator=generator)

        loss, metadata = objective(
            generated,
            current_teacher=teacher,
            stage=False,
        )
        kernel_gg = rbf_frobenius_kernel(
            generated, generated, objective.sigma)
        fixed_attraction = (
            rbf_frobenius_kernel(
                generated, objective.landmarks, objective.sigma)
            @ objective.alpha
        ).mean()
        kernel_gt = rbf_frobenius_kernel(
            generated, teacher, objective.sigma)
        kernel_tt = rbf_frobenius_kernel(
            teacher, teacher, objective.sigma)
        expected = (
            kernel_gg.mean()
            - 2.0
            * (
                (2.0 / 3.0) * fixed_attraction
                + (1.0 / 3.0) * kernel_gt.mean()
            )
            + (2.0 / 3.0) * objective.k_rr
            + (1.0 / 3.0) * kernel_tt.mean()
        )
        torch.testing.assert_close(loss, expected)
        actual_gradient, = torch.autograd.grad(
            loss, generated, retain_graph=True)
        expected_gradient, = torch.autograd.grad(expected, generated)
        torch.testing.assert_close(
            actual_gradient,
            expected_gradient,
            rtol=2.0e-5,
            atol=2.0e-6,
        )
        self.assertEqual(int(metadata["global_samples"]), 4)
        for partial_key in (
            "partial_student_reference_enabled",
            "gradient_student_samples",
            "student_reference_samples",
            "student_reference_detached",
            "student_repulsion_denominator",
            "fixed_teacher_attraction_denominator",
            "current_teacher_attraction_denominator",
            "current_teacher_self_kernel_denominator",
            "full_student_partial_gradient",
        ):
            self.assertNotIn(partial_key, metadata)



if __name__ == "__main__":
    unittest.main()
