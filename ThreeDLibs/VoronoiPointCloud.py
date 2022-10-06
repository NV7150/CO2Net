import numpy as np
from scipy.spatial import Delaunay, ConvexHull
import open3d as o3d
# from sympy.combinatorics import Polyhedron
# from sympy.geometry import Point
from ThreeDLibs.Util import softmax


A = 10

class Surface:
    def __init__(self, ps):
        self.ps = np.array(ps)
        v1, v2 = -self.ps[[1, 2]] + self.ps[0]
        self.norm = np.cross(v1, v2)
        self.norm /= np.linalg.norm(self.norm)

    def projection_p(self, p):
        return p - np.dot(self.norm, p - self.ps[0]) * self.norm


class VoronoiPointCloud:
    def __init__(self, mother_points, outer_amount=20):
        mother_points = np.array(mother_points)
        d = Delaunay(mother_points)
        self.mother_points = np.array(d.points)
        self.hull_idxs = d.simplices.tolist()
        hulls_np = d.points[np.array(self.hull_idxs)].tolist()

        self.normal_th = len(hulls_np) - 1
        self.base_mp_num = len(mother_points)

        convexes = np.array(d.convex_hull)
        idxs = list(set(convexes.flatten()))
        hull_v = np.array(d.points[idxs])
        g = np.mean(hull_v, axis=0)

        outer_dict = {}
        for i in idxs:
            p = d.points[i]
            norm = p - g
            norm /= np.linalg.norm(norm)
            outer_dict.setdefault(i, norm)
        # for c in convexes:
        #     vs = d.points[c]
        #     v1 = vs[1] - vs[0]
        #     v2 = vs[2] - vs[0]
        #     norm = np.cross(v1, v2)
        #     dir_vec = g - np.mean(vs, axis=0)
        #     norm *= 1 if np.dot(norm, np.transpose(dir_vec)) < 0 else -1
        #
        #     norm /= np.linalg.norm(norm)
        #
        #     for v in c:
        #         if v not in outer_dict.keys():
        #             outer_dict.setdefault(v, np.zeros(3))
        #         outer_dict[v] += norm

        # for k in outer_dict.keys():
        #     outer_dict[k] /= np.linalg.norm(outer_dict[k])

        self.hull_surfaces = []

        for hull_surface in d.convex_hull:
            hull = []
            hull_idxs = []
            for v_i in hull_surface:
                hull.append(v_i)
                self.mother_points = np.concatenate([self.mother_points, [outer_dict[v_i] * outer_amount + d.points[v_i]]], axis=0)
                hull.append(len(self.mother_points) - 1)
                hull_idxs.append(v_i)
            self.hull_idxs.append(list(set(hull_idxs)))
            hulls_np.append(self.mother_points[hull].tolist())
            self.hull_surfaces.append(Surface(d.points[hull_surface]))

        self.hulls = []
        for hull in hulls_np:
            self.hulls.append(ConvexHull(hull))

    def get_weights(self, point):
        p = np.array(point)
        hit_idxs = []
        hit = False
        is_outer = False
        surface_i = -1
        for i, h in enumerate(self.hulls):
            if not np.array_equal(h.vertices, ConvexHull(np.concatenate([h.points, [p]], axis=0)).vertices):
                continue
            hit_idxs = self.hull_idxs[i]
            hit = True

            if i > self.normal_th:
                is_outer = True
                surface_i = i - self.normal_th - 1

        weight_map = np.zeros(self.base_mp_num)
        if not hit:
            return weight_map

        if is_outer:
            p = self.hull_surfaces[surface_i].projection_p(p)

        ps = self.mother_points[hit_idxs]
        dist_vec = ps - p
        dists = np.linalg.norm(dist_vec, axis=1).flatten()
        dists = A / np.power(dists, 2)
        # min_dist = np.min(dists)
        # weights = ((np.max(dists) - (dists - min_dist)) / (np.max(dists) - min_dist)) + 1e-5

        weights = softmax(dists)

        weight_map[hit_idxs] = weights
        return weight_map

    def export_sm(self):
        vs = []
        lines = []
        for h in self.hulls:
            base_i = len(vs)
            vs.extend(h.points)
            for i in range(len(h.vertices)):
                for j in range(i + 1, len(h.vertices)):
                    lines.append((i + base_i, j + base_i))
        lineset = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(vs),
            lines=o3d.utility.Vector2iVector(lines)
        )
        return lineset


class SensorVPcd:
    def __init__(self, sensor_points, outer_amount=20):
        self.n2idx = {}
        mother_points = []
        for i, (n, p) in enumerate(sensor_points.items()):
            self.n2idx.setdefault(n, i)
            mother_points.append(p)
        self.vpcd = VoronoiPointCloud(mother_points, outer_amount)

    def dict2ndarray(self, sensor_val_dict):
        vals = np.zeros(len(sensor_val_dict))

        for n, v in sensor_val_dict.items():
            vals[self.n2idx[n]] = v

        return vals

    def get_weighted(self, sensor_val_dict, p):
        vals = self.dict2ndarray(sensor_val_dict)
        weight = self.vpcd.get_weights(p)

        return vals * weight

    def export_sm(self):
        return self.vpcd.export_sm()




