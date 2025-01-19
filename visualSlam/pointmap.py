import open3d as o3d
import numpy as np

class Map(object):
    def __init__(self):
        self.frames = []
        self.points = []
        self.state = None
        self.vis = None

    def create_viewer(self):
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window()
        # Global coordinate frame
        self.global_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)
        self.vis.add_geometry(self.global_frame)

        # Ego vehicle position (represented as a sphere)
        self.ego_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.1)
        self.ego_sphere.paint_uniform_color([1, 0, 0])  # Red color
        self.vis.add_geometry(self.ego_sphere)

        # Point cloud
        self.pcd = o3d.geometry.PointCloud()
        self.vis.add_geometry(self.pcd)
        self.ctr = self.vis.get_view_control()
        self.ctr.set_zoom(-10) 

    def viewer_refresh(self,poses, pts):
        # Update ego vehicle position
        translations = poses[:, :3, 3]
        print(translations,translations[translations.shape[0]-1,:])
        self.ego_sphere.translate(translations[translations.shape[0]-1,:], relative=True)

        # Update point cloud
        #self.pcd.points = o3d.utility.Vector3dVector(pts)
        self.pcd.paint_uniform_color([0, 1, 0])  # Green color

        # Update geometries
        self.vis.update_geometry(self.ego_sphere)
        self.vis.update_geometry(self.pcd)

        # Render
        self.vis.poll_events()
        self.vis.update_renderer()

    def display(self):
        poses = np.array([frame.pose for frame in self.frames])
        pts = np.array([point.pt for point in self.points])

        self.viewer_refresh(poses, pts)

    def run_viewer(self):
        self.create_viewer()
        while True:
            self.viewer_refresh()

class Point(object):
    def __init__(self, mapp, loc):
        self.frames = []
        self.pt = loc
        self.idxs = []
        self.id = len(mapp.points)
        mapp.points.append(self)

    def add_observation(self, frame, idx):
        self.frames.append(frame)
        self.idxs.append(idx)
