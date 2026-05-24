import unittest

import numpy as np

from utils.ground_alignment import (
    GroundAlignmentConfig,
    estimate_ground_alignment,
    transform_camera_rt,
    transform_points,
)


class GroundAlignmentTests(unittest.TestCase):
    def test_estimates_tilted_ground_plane(self):
        rng = np.random.default_rng(3)
        xy = rng.uniform(-2.0, 2.0, size=(800, 2))
        z = 0.2 * xy[:, 0] - 0.1 * xy[:, 1] + 0.7
        ground = np.column_stack([xy, z])
        ground += rng.normal(scale=0.002, size=ground.shape)

        clutter = rng.uniform(-2.0, 2.0, size=(120, 3))
        clutter[:, 2] += 2.5
        points = np.vstack([ground, clutter])

        config = GroundAlignmentConfig(
            voxel_size=0.0,
            sor_k=0,
            ransac_iters=300,
            ransac_threshold=0.02,
            low_percentile=85.0,
            min_inlier_ratio=0.4,
            seed=1,
        )
        result = estimate_ground_alignment(points, config)
        transformed_ground = transform_points(ground, result.transform)

        self.assertGreater(result.inlier_count, 700)
        self.assertLess(abs(transformed_ground[:, 2]).mean(), 0.01)
        np.testing.assert_allclose(
            result.transform[:3, :3] @ result.normal,
            np.array([0.0, 0.0, 1.0]),
            atol=1e-6,
        )

    def test_camera_transform_preserves_world_to_camera_projection(self):
        world_to_ground = np.array(
            [
                [0.0, -1.0, 0.0, 1.0],
                [1.0, 0.0, 0.0, -2.0],
                [0.0, 0.0, 1.0, 0.5],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        camera_R = np.eye(3)
        camera_T = np.array([0.2, -0.3, 1.1])
        point_world = np.array([1.7, -0.4, 2.2])

        new_R, new_T = transform_camera_rt(camera_R, camera_T, world_to_ground)
        point_ground = transform_points(point_world[None, :], world_to_ground)[0]

        old_camera_point = camera_R.T @ point_world + camera_T
        new_camera_point = new_R.T @ point_ground + new_T
        np.testing.assert_allclose(new_camera_point, old_camera_point, atol=1e-9)


if __name__ == "__main__":
    unittest.main()
