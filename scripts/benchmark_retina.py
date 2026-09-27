"""Compare warmed retinal samplers, excluding rendering and JIT compilation.

Run from a source checkout: PYTHONPATH=src python scripts/benchmark_retina.py
"""

import json
import platform
import statistics
import time

import numba
import numpy as np

from flygym.vision.retina import Retina


def main():
    retina = Retina()
    image = np.random.default_rng(42).integers(
        0, 256, (retina.nrows, retina.ncols, 3), dtype=np.uint8
    )

    def reference():
        return retina.raw_image_to_hex_pxls(retina.correct_fisheye(image))

    def fused():
        return retina.fisheye_image_to_hex_pxls(image)

    np.testing.assert_array_equal(reference(), fused())
    samples = {"reference": [], "fused": []}
    for repeat in range(30):
        order = [("reference", reference), ("fused", fused)]
        if repeat % 2:
            order.reverse()
        for name, sample in order:
            start = time.perf_counter()
            sample()
            samples[name].append(time.perf_counter() - start)
    medians = {name: statistics.median(values) for name, values in samples.items()}
    print(
        json.dumps(
            {
                "platform": platform.platform(),
                "python": platform.python_version(),
                "numpy": np.__version__,
                "numba": numba.__version__,
                "numba_threads": numba.get_num_threads(),
                "image_shape": image.shape,
                "exact": True,
                "median_seconds": medians,
                "speedup": medians["reference"] / medians["fused"],
                "samples_seconds": samples,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
