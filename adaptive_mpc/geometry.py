"""ContactSDF-inspired support-plane geometry for the disclosed box-union task.

These are smooth effective contact gaps, not an exact Euclidean distance oracle.
All public gaps use FULL separation and matching velocity Jacobians.
"""
from itertools import product

import casadi as ca
import numpy as np

from .task import PEG, WALLS, rotation, skew, velocity_map


def smooth_max(values, sharpness):
    # Explicit stable logsumexp also supports CasADi 3.6.5 SX expansion.
    scaled = sharpness * ca.vertcat(*values)
    shift = ca.mmax(scaled)
    return (shift + ca.log(ca.sum1(ca.exp(scaled - shift)))) / sharpness


def smooth_min(values, sharpness):
    return -smooth_max([-x for x in values], sharpness)


def box_field(point, center, halfsize, sharpness):
    d = point - ca.DM(center)
    return smooth_max([sign * d[j] - halfsize[j]
                       for j in range(3) for sign in (1, -1)], sharpness)


def point_jacobian(rot, local):
    return ca.horzcat(ca.DM.eye(3), -skew(rot @ local) @ rot, ca.DM.zeros(3, 9))


class ContactGeometry:
    contact_count = 11
    row_count = 44

    def __init__(self, sharpness=800.):
        if sharpness <= 0:
            raise ValueError("sharpness must be positive")
        q = ca.MX.sym("configuration", 16)
        rot, b = rotation(q[3:7]), velocity_map(q)
        gaps, normals, tangents, labels = [], [], [], []

        def append(gap, tangent, label):
            gaps.append(gap)
            normals.append(ca.jacobian(gap, q) @ b)
            tangents.append(tangent)
            labels.append(label)

        # Eight object-part/environment pairs, with 26 samples on each box.
        for part, (center, half) in enumerate(PEG):
            samples = [ca.DM(np.asarray(center) + np.asarray(half) * np.array(v))
                       for v in product((-1., 0., 1.), repeat=3) if any(v)]
            points = [q[:3] + rot @ s for s in samples]
            for wall in range(4):
                if wall == 0:
                    values, axes = [p[2] for p in points], (0, 1)
                else:
                    wc, wh, axes = WALLS[wall - 1]
                    values = [box_field(p, wc, wh, sharpness) for p in points]
                minimum = smooth_min(values, sharpness)
                # The nine coplanar bottom-face samples otherwise shift table gap.
                gap = minimum + (np.log(9.) / sharpness if wall == 0 else 0.)
                weights = ca.exp(-sharpness * (ca.vertcat(*values) - minimum))
                effective = ca.horzcat(*samples) @ (weights / ca.sum1(weights))
                jp = point_jacobian(rot, effective)
                t1, t2 = jp[axes[0], :], jp[axes[1], :]
                append(gap, ca.vertcat(t1, t2, -t1, -t2), f"part{part}:wall{wall}")

        # Three sphere/peg pairs. The peg is a smooth union of its two boxes.
        query = ca.MX.sym("body_query", 3)
        field = smooth_min([box_field(query, c, h, sharpness) for c, h in PEG], sharpness)
        query_fn = ca.Function("peg_field", [query], [field, ca.gradient(field, query)])
        for finger in range(3):
            local = rot.T @ (q[7 + 3*finger:10 + 3*finger] - q[:3])
            distance, n = query_fn(local)
            n = n / ca.sqrt(ca.sumsqr(n) + 1e-12)
            normal_world = rot @ n
            t1 = ca.cross(normal_world, ca.DM([0., 0., 1.]))
            t1 = t1 / ca.sqrt(ca.sumsqr(t1) + 1e-12)
            t2 = ca.cross(normal_world, t1)
            jf = ca.DM.zeros(3, 15)
            jf[:, 6 + 3*finger:9 + 3*finger] = ca.DM.eye(3)
            relative = jf - point_jacobian(rot, local - distance * n)
            rows = ca.vertcat(t1.T, t2.T, -t1.T, -t2.T) @ relative
            append(distance - .010, rows, f"peg:finger{finger}")

        self.labels = tuple(labels)
        phi = ca.vertcat(*(ca.repmat(g, 4, 1) for g in gaps))
        jn = ca.vertcat(*(ca.repmat(n, 4, 1) for n in normals))
        jt = ca.vertcat(*tangents)
        self.function = ca.Function("contact_geometry", [q], [phi, jn, jt])
        self.gap_jacobian = ca.Function("gap_jacobian", [q], [ca.jacobian(phi, q)])

    def evaluate(self, q):
        phi, jn, jt = self.function(q)
        return np.asarray(phi).ravel(), np.asarray(jn), np.asarray(jt)
